"""Readout-only continuous costs/timing and exact experiment parameter counts.

This does not estimate unexecuted Mamba kernels or infer full-volume speed from
small CPU tests. Every attention projection/matmul is continuous, including
when a particular continuous memory happens to contain only zeros.
"""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import statistics
import time

import torch


def linear_attention_cost(channels, sites):
    return {
        'attention_projection_macs': sites*256*channels,
        'attention_aggregation_macs': sites*3*32*32,
        'attention_retrieval_macs': sites*32*32,
        'attention_normalization_dot_macs': sites*32,
        'attention_kernel_elements': sites*4*32,
        'attention_key_embedding_additions': sites*3*32,
        'attention_key_reduction_additions_upper_bound': sites*3*32,
        'attention_output_divisions': sites*32,
        'attention_denominator_epsilon_additions': sites,
    }


def experiment_models(path):
    """Build real baseline/attention modules without executing selective scan."""
    from snn_nnunet.run_model import _config
    from snn_nnunet.prepare_plans import DEFAULT_SNN_CONFIG
    from snn_nnunet.network_adapter import SNNConfig,build_core
    config=_config(path)
    if config['snn']['variant']!='orig':
        raise ValueError('This experiment report expects the supplied orig BraTS configuration')
    snn=config['snn']
    kwargs=deepcopy(DEFAULT_SNN_CONFIG['model_kwargs'])
    spatial=deepcopy(snn['spatial_mamba'])
    kwargs['spatial_output_spiking']=spatial.pop('output_spiking',False)
    kwargs.update(spatial)
    kwargs['output_spiking']=snn['output_spiking']
    kwargs['temporal_mamba']=deepcopy(snn['temporal_mamba'])
    models={}
    for key,model_name in [('baseline','selective_multitimescale_lif'),('attention','MultiMemoryLinearAttentionLIF')]:
        values=deepcopy(DEFAULT_SNN_CONFIG)
        values.update(model_name='orig',model_kwargs=deepcopy(kwargs),use_fptt=snn['use_fptt'],k=snn['k'],
                      temporal_axis=snn['temporal_axis'])
        values['model_kwargs']['recurrent_neuron']['model']=model_name
        torch.manual_seed(snn['seed'])
        models[key]=build_core(SNNConfig.from_plans({'snn_config':values}))
    return config,models


def measure_readout(neuron,features,memories,*,backward,warmup,repeats):
    device=features.device
    def synchronize():
        if device.type=='cuda':torch.cuda.synchronize(device)
    def step():
        neuron.zero_grad(set_to_none=True)
        for x in [features,*memories]:x.grad=None
        with torch.set_grad_enabled(backward):
            output=neuron._aggregate_memories(features,memories)[0]
            if backward:output.square().mean().backward()
    for _ in range(warmup):step()
    synchronize()
    baseline=None
    if device.type=='cuda':
        baseline=torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    durations=[]
    for _ in range(repeats):
        synchronize()
        start=time.perf_counter()
        step()
        synchronize()
        durations.append(time.perf_counter()-start)
    seconds=statistics.median(durations)
    return {'device':str(device),'shape_bchw':list(features.shape),'dtype':'float32',
            'forward_backward':backward,'warmup':warmup,'repeats':repeats,
            'median_seconds':seconds,'min_seconds':min(durations),'max_seconds':max(durations),
            'samples_per_second':features.shape[0]/seconds,
            'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
            'peak_gpu_above_baseline_bytes':torch.cuda.max_memory_allocated(device)-baseline if baseline is not None else None,
            'scope':'readout only; excludes memory updates, W_R, soma, backbone and optimizer; not training pipeline throughput'}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('brats24_config.yaml'))
    parser.add_argument('--output',type=Path,default=Path('analysis_results/linear_attention/report.json'))
    parser.add_argument('--warmup',type=int,default=2)
    parser.add_argument('--repeats',type=int,default=7)
    parser.add_argument('--timing-size',type=int,default=16)
    parser.add_argument('--measure-cuda',action='store_true')
    args=parser.parse_args(argv)
    if min(args.warmup,args.repeats,args.timing_size)<1:parser.error('warmup, repeats and timing-size must be positive')
    from snn_nnunet.recurrent_neuron import named_selective_neurons,make_selective_neuron
    from snn_nnunet.profile_paper_unet import selective_cost,estimate_cost
    from snn_nnunet.paper_unet import PaperUNet
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config,models=experiment_models(args.config)
    snn=config['snn']
    z,h,w=snn['training_patch_size']
    rows=[]
    cells=dict(named_selective_neurons(models['baseline']))
    new_cells=dict(named_selective_neurons(models['attention']))
    # Actual configured spatial geometry; odd patches are padded by the core.
    multiple=models['baseline'].encoder_scale
    h,w=(math.ceil(n/multiple)*multiple for n in (h,w))
    sizes=[]
    for stride in models['baseline'].downsample_strides:
        h,w=math.ceil(h/stride),math.ceil(w/stride)
        sizes.append((h,w))
    for name,old in cells.items():
        stage=int(name.split('.')[0].replace('conv_block',''))-1
        stage_h,stage_w=sizes[stage]
        sites=z*stage_h*stage_w
        new=new_cells[name]
        before,after=selective_cost(old,sites),selective_cost(new,sites)
        row={'name':name,'channels':old.channels,'shape_bzchw':[1,z,old.channels,stage_h,stage_w],
             'baseline_continuous_cost':before,'attention_continuous_cost':after,
             'readout_delta_macs':after['continuous_linear_macs']-before['continuous_linear_macs'],
             'baseline_neuron_parameters':sum(p.numel() for p in old.parameters()),
             'attention_neuron_parameters':sum(p.numel() for p in new.parameters())}
        device=torch.device('cuda' if args.measure_cuda and torch.cuda.is_available() else 'cpu')
        torch.manual_seed(126)
        features=torch.randn(1,old.channels,args.timing_size,args.timing_size,device=device,requires_grad=True)
        memories=[torch.randn_like(features,requires_grad=True) for _ in range(3)]
        # Fresh real cells avoid moving live parent models or their buffers.
        benchmark_cells={}
        for label,source in [('baseline',old),('attention',new)]:
            candidate=make_selective_neuron(source.channels,source.config).to(device)
            candidate.load_state_dict(source.state_dict(),strict=True)
            benchmark_cells[label]=candidate
        row['readout_timings']={label:{
            'inference':measure_readout(cell,features,memories,backward=False,warmup=args.warmup,repeats=args.repeats),
            'forward_backward':measure_readout(cell,features,memories,backward=True,warmup=args.warmup,repeats=args.repeats)}
            for label,cell in benchmark_cells.items()}
        rows.append(row)
    # Fully accounted auxiliary backbone, explicitly distinct from BraTS orig.
    paper={}
    for label,selector in [('baseline','selective_multitimescale_lif'),('attention','MultiMemoryLinearAttentionLIF')]:
        overrides={f'conv_block{i}':selector for i in (1,2,3)}
        core=PaperUNet(neurons={'overrides':overrides})
        ledger=estimate_cost(core,(81,128,128))
        paper[label]={
            'parameters':ledger['actual_parameters'],
            'continuous_GMAC':(ledger['continuous_input_macs']+ledger['continuous_ssn_linear_macs'])/1e9,
            'spike_driven_GAC_at_unit_activity':ledger['spike_eligible_dense_macs']/1e9,
            'scope':'auxiliary unidirectional paper_unet, B=1,Z81,H=W128, encoder SiLIF and decoder PLIF; theoretical binary event upper bound at evaluation, not measured energy',
            'ledger':ledger}
    report={'torch':str(torch.__version__),'cuda_available':torch.cuda.is_available(),
            'experiment_config':str(args.config),'training_patch_zyx':snn['training_patch_size'],
            'primary_backbone':snn['variant'],'threads':1,
            'parameters':{k:{'total':sum(p.numel() for p in m.parameters()),
                             'trainable':sum(p.numel() for p in m.parameters() if p.requires_grad)} for k,m in models.items()},
            'silif_blocks':rows,
            'primary_neuron_continuous_GMAC':{label:sum(row[f'{label}_continuous_cost']['continuous_linear_macs'] for row in rows)/1e9
                                            for label in ('baseline','attention')},
            'primary_whole_network_continuous_GMAC':'unmeasured: Mamba/SS2D and complete backbone work not estimated by this readout ledger',
            'primary_whole_network_spike_GAC':'unmeasured: requires per-layer input activity and scaled-dropout classification; no new spike synapses added',
            'primary_readout_spike_GAC':0,
            'full_volume_inference':'unmeasured: no real MRI/predictor/GPU benchmark executed',
            'full_training_throughput':'unmeasured: readout-only forward/backward timing is reported separately',
            'peak_full_model_gpu_memory':'unmeasured: no full-model GPU benchmark executed',
            'auxiliary_paper_unet':paper,
            'count_convention':'one MAC=one multiply-accumulate; Q/K/V/O and attention statistics/retrieval/normalizer, spatial mixing and W_R are continuous MACs; binary-input synapses are potential ACs at evaluation; inverted dropout-scaled inputs require continuous/scaled arithmetic',
            'elementwise_scope':'partial tensor-site counts, not complete scalar FLOPs; cross-memory statistic/vector additions and the internal ELU+1 arithmetic are not individually itemized',
            'exclusions':['normalization','PLIF threshold/reset/surrogate','coefficient functions','kernel normalization',
                          'biases and scalar arithmetic (separate fields where available)','backward/optimizer in forward MAC tables'],
            'accuracy':'No new training or accuracy experiment performed.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print('Wrote',args.output)
    print('Parameters:',report['parameters'])
    print('Primary SiLIF continuous GMAC:',report['primary_neuron_continuous_GMAC'])


if __name__=='__main__':main()
