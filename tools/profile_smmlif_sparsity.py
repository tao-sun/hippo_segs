"""Layerwise MAC * nonzero fraction on a real validation inference tile.

Reuses the repository activity_record and the independent convolution audit.
All functional convolutions inside selective neurons are measured separately.
No training sources/checkpoints are modified.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import blosc2
import numpy as np
import torch
from nnunetv2.inference.sliding_window_prediction import compute_steps_for_sliding_window
from snn_nnunet.network_adapter import SNNConfig, SNNnnUNetAdapter
from snn_nnunet.profile_paper_unet import activity_record, estimate_cost
from profile_smmlif_gmac import OperationAudit, read_logged_plans


class SparseAudit(OperationAudit):
    def __init__(self, model):
        super().__init__()
        self.layers = {}
        self.modules_by_name = dict(model.named_modules())
        self.weight_names = {p.data_ptr(): n.removesuffix('.weight')
                             for n, p in model.named_parameters() if n.endswith('.weight')}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = super().__torch_dispatch__(func, types, args, kwargs)
        if str(func) != 'aten.convolution.default':
            return result
        x, weight = args[:2]
        scope = self.scopes[-1]
        # Functional gates/projection bypass module forward. Their FP32 weights
        # retain parameter storage; normalized depthwise weights use module scope.
        name = self.weight_names.get(weight.data_ptr(), scope)
        layer = self.modules_by_name[name]
        record = activity_record(layer, x)
        row = self.layers.setdefault(name, dict(
            input_shape=list(x.shape), output_shape=list(result.shape), calls=0,
            dense_macs=0, nonzero_inputs=0, input_elements=0,
            weighted_operation_estimate=0.0, binary_input=True,
            min_step_density=1.0, max_step_density=0.0))
        assert row['input_shape'] == list(x.shape)
        row['calls'] += 1
        row['dense_macs'] += record['dense_macs']
        row['nonzero_inputs'] += record['nonzero_inputs']
        row['input_elements'] += record['input_elements']
        row['weighted_operation_estimate'] += record['nonzero_weighted_operation_estimate']
        row['binary_input'] &= record['binary_input']
        row['min_step_density'] = min(row['min_step_density'], record['nonzero_fraction'])
        row['max_step_density'] = max(row['max_step_density'], record['nonzero_fraction'])
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path, default=Path('SMMLIF_training_DS_01_29077508.out'))
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--case-id', default='BraTS-GLI-00020-100')
    parser.add_argument('--tile-index', type=int, default=0)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output-dir', type=Path, default=Path('analysis_results/smmlif_29077508/sparsity'))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    plans = read_logged_plans(args.log)
    config = SNNConfig.from_plans(plans)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(config.deterministic)
    model = SNNnnUNetAdapter(config).to(args.device).eval()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    assert checkpoint['snn_config'] == plans['snn_config']
    model.load_state_dict(checkpoint['network_weights'], strict=True)
    epoch = checkpoint['current_epoch']
    del checkpoint
    summary = json.loads(args.summary.read_text())
    assert args.case_id in {r['case'] for r in summary['cases']}, 'Not in validation cases'
    settings = summary['settings']
    patch = tuple(settings['patch_size'])
    assert patch == config.validation_patch_size
    data_file = args.data_dir / (args.case_id + '.b2nd')
    data = blosc2.open(str(data_file), mode='r')
    spatial = list(data.shape[1:])
    padded_spatial = [max(n, p) for n, p in zip(spatial, patch)]
    steps = compute_steps_for_sliding_window(padded_spatial, patch, settings['tile_step_size'])
    starts = list(itertools.product(*steps))
    if not 0 <= args.tile_index < len(starts):
        parser.error(f'--tile-index must be in [0, {len(starts)-1}]')
    start = starts[args.tile_index]
    # Match centered zero padding used by pad_nd_image in native inference.
    before = [(p-n)//2 for n, p in zip(spatial, padded_spatial)]
    slices = tuple(slice(max(0,s-b), min(n,s-b+p))
                   for s,b,n,p in zip(start,before,spatial,patch))
    patch_data = np.asarray(data[(slice(None), *slices)], dtype=np.float32)
    pad = [(0,0)] + [(max(0,b-s), max(0,s-b+p-n))
                        for s,b,n,p in zip(start,before,spatial,patch)]
    patch_data = np.pad(patch_data, pad)
    assert patch_data.shape == (4,*patch)
    x = torch.from_numpy(np.ascontiguousarray(patch_data)).unsqueeze(0).to(args.device)
    audit = SparseAudit(model)
    handles = []
    for name, module in model.named_modules():
        handles.append(module.register_forward_pre_hook(audit.enter(name)))
        handles.append(module.register_forward_hook(audit.leave))
    print(f'Measuring {args.case_id}, tile {args.tile_index}/{len(starts)}, start {start}, shape {tuple(x.shape)}', flush=True)
    try:
        with torch.no_grad(), audit:
            output = model(x)
    finally:
        for handle in handles:
            handle.remove()
    assert torch.isfinite(output).all()
    ledger = estimate_cost(model.core, patch, batch=1)
    assert sum(r['dense_macs'] for r in audit.layers.values()) == ledger['dense_equivalent_macs']
    assert sum(r['calls'] for r in audit.layers.values()) == 2673
    assert len(audit.layers) == 33
    binary_weighted = continuous_weighted = dense = 0
    for name, row in audit.layers.items():
        row['density'] = row['nonzero_inputs']/row['input_elements']
        row['sparsity'] = 1-row['density']
        assert math.isclose(row['weighted_operation_estimate'], row['dense_macs']*row['density'], rel_tol=1e-12, abs_tol=1e-5)
        row['zero_weighted_operation_estimate'] = row['dense_macs']*row['sparsity']
        row['zero_weighted_giga_operations'] = row['zero_weighted_operation_estimate']/1e9
        row['dense_gmac'] = row['dense_macs']/1e9
        row['weighted_giga_operations'] = row['weighted_operation_estimate']/1e9
        row['operation_type'] = 'AC_estimate' if row['binary_input'] else 'MAC_estimate'
        dense += row['dense_macs']
        if row['binary_input']:
            binary_weighted += row['weighted_operation_estimate']
        else:
            continuous_weighted += row['weighted_operation_estimate']
    report = dict(
        case=args.case_id, preprocessed_file=str(data_file), volume_shape=list(data.shape),
        tile_index=args.tile_index, total_tiles=len(starts), tile_start=list(start),
        input_shape=list(x.shape), patch_sha256=hashlib.sha256(patch_data.tobytes()).hexdigest(),
        checkpoint=str(args.checkpoint), checkpoint_epoch=epoch,
        torch_version=torch.__version__, device=args.device, eval=True, no_grad=True,
        scope='ONE real validation tile, identity orientation; 81 steps from zero state; not a dataset average or full-volume activity measurement',
        formula='weighted_l = dense_MAC_l * (nonzero_input_elements_l / input_elements_l) = dense_MAC_l * (1 - input_sparsity_l)',
        convention='Exact zeros only; approximate density-weighted operation estimate, not exact fanout/padding-weighted zero-skipped products; does not measure acceleration of dense PyTorch',
        dense_gmac=dense/1e9, estimated_continuous_gmac=continuous_weighted/1e9,
        estimated_binary_gac=binary_weighted/1e9,
        combined_giga_operation_estimate=(continuous_weighted+binary_weighted)/1e9,
        zero_weighted_giga_operation_estimate=(dense-continuous_weighted-binary_weighted)/1e9,
        activity_weighted_fraction=(continuous_weighted+binary_weighted)/dense,
        non_mac_operations='Excluded: elementwise state updates, bias, nonlinearities, normalization, pooling and data movement; original report records these separately',
        layers=audit.layers)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'profile.json').write_text(json.dumps(report,indent=2)+'\n')
    fields=['layer','dense_MAC','dense_GMAC','input_elements','nonzero_inputs','input_sparsity_percent','input_activity_percent','MAC_times_sparsity','GMAC_times_sparsity','weighted_operations','weighted_giga_operations','operation_type','calls']
    with (args.output_dir/'layers.csv').open('w') as f:
        writer=csv.writer(f); writer.writerow(fields)
        for name,r in audit.layers.items():
            writer.writerow([name,r['dense_macs'],r['dense_gmac'],r['input_elements'],r['nonzero_inputs'],100*r['sparsity'],100*r['density'],r['zero_weighted_operation_estimate'],r['zero_weighted_giga_operations'],r['weighted_operation_estimate'],r['weighted_giga_operations'],r['operation_type'],r['calls']])
    lines=[f'**Sparsità misurata: {args.case_id}, tile {args.tile_index}, start {list(start)}.**',
           '', 'Un forward reale FP32, batch 1, eval/no_grad, checkpoint_best del run, 81 slice consecutive da stato azzerato; identità senza mirroring. Risultato specifico di questa patch, non media sul dataset.',
           '', '`quota_teoricamente_saltabile = MAC_densi × sparsità_ingresso`; `operazioni_pesate = MAC_densi × attività_ingresso = MAC_densi × (1 − sparsità_ingresso)`.',
           '', '| Layer | GMAC densi | Sparsità ingresso % | Attività ingresso % | MAC × sparsità / 1e9 (saltabili) | MAC × attività / 1e9 | Tipo |',
           '|---|---:|---:|---:|---:|---:|---|']
    for name,r in audit.layers.items():
        lines.append(f"| {name.removeprefix('core.')} | {r['dense_gmac']:.9f} | {100*r['sparsity']:.6f} | {100*r['density']:.6f} | {r['zero_weighted_giga_operations']:.9f} | {r['weighted_giga_operations']:.9f} | {r['operation_type']} |")
    lines += ['', f"Totale denso: **{report['dense_gmac']:.9f} GMAC**. Stima pesata: **{report['estimated_continuous_gmac']:.9f} GMAC continui + {report['estimated_binary_gac']:.9f} GAC su ingressi binari**.",
              '', f"Quota teoricamente saltabile MAC×sparsità: {report['zero_weighted_giga_operation_estimate']:.9f} miliardi. Somma aritmetica attiva, con tipi di operazione distinti: {report['combined_giga_operation_estimate']:.9f} miliardi di operazioni, pari al {100*report['activity_weighted_fraction']:.4f}% del conteggio denso.",
              '', 'Le densità sono misurate direttamente sugli ingressi di ogni convoluzione, anche dopo pooling e concatenazioni e nei kernel/gate/proiezioni funzionali del neurone. La stima riusa activity_record del profiler del repository; è una media pesata per cardinalità, non un conteggio esatto delle connessioni attive ai bordi. Nessuna soglia numerica è usata: nonzero significa valore diverso da zero.',
              '', 'Le memorie continue possono avere zeri al primo passo; questi sono misurati direttamente e non dedotti dal firing rate del soma. Il backend PyTorch rimane denso, quindi il totale eseguito resta 413.769793536 GMAC. Le operazioni elementwise e gli altri costi esclusi nel rapporto originale non sono scontati.',
              '', 'Non moltiplicare questa stima per il numero di tile/mirror per presentarla come costo misurato di un volume: ogni ingresso e traiettoria di stato può produrre attività diversa.']
    (args.output_dir/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:report[k] for k in ('dense_gmac','estimated_continuous_gmac','estimated_binary_gac','combined_giga_operation_estimate','activity_weighted_fraction')},indent=2))
    print(f'Wrote {args.output_dir}/profile.json, layers.csv, report.md', flush=True)


if __name__=='__main__':
    main()
