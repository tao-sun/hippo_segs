"""Exercise Arch2 through the same plans, CLI and trainer paths as paper_unet."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from snn_nnunet import cli, run_model
from snn_nnunet.network_adapter import SNNConfig, SNNnnUNetAdapter
from snn_nnunet.prepare_plans import DEFAULT_SNN_CONFIG, update_snn_config
from test_arch2 import core, one_thread, options, reference_scan


def settings():
    result = deepcopy(DEFAULT_SNN_CONFIG)
    result.update(model_name='arch2', model_kwargs=options(), k=2, use_fptt=False)
    return result


def experiment():
    return dict(family='snn', dataset_id=1, fold=0, gpus=1, run_dir='/tmp/arch2-run',
                snn=dict(variant='arch2', arch2=options(), temporal_axis=0, k=2,
                         use_fptt=False, training_patch_size=[4,16,24]))


def test_yaml_resolves_arch2_without_injecting_legacy_controls():
    resolved = run_model._validate_experiment(experiment())
    assert resolved['snn']['arch2']['patch_size'] == 2
    assert 'spatial_mamba' not in resolved['snn']
    assert 'paper_unet' not in resolved['snn']


def test_vss_output_switch_roundtrips_through_yaml_plans_and_weights(monkeypatch):
    import snn_nnunet.spikmamba as mixers
    from torch import nn
    monkeypatch.setattr(mixers, 'selective_scan_fn', reference_scan())
    raw = experiment()
    raw['snn']['arch2']['encoder']['vss_output_spiking'] = False
    resolved = run_model._validate_experiment(raw)
    value = settings()
    value['model_kwargs'] = resolved['snn']['arch2']
    config = SNNConfig.from_plans({'snn_config': value})
    restored = SNNConfig.from_plans({'snn_config': json.loads(json.dumps(config.to_dict()))})
    assert restored.model_kwargs['encoder']['vss_output_spiking'] is False
    network = SNNnnUNetAdapter(config).eval()
    copy = SNNnnUNetAdapter(restored).eval()
    copy.load_state_dict(network.state_dict(), strict=True)
    assert all(isinstance(block.post_plif, nn.GELU) for block in copy.core.encoder)
    data = torch.randn(1,4,2,16,16)
    torch.testing.assert_close(network(data), copy(data), rtol=0, atol=0)


@pytest.mark.parametrize('extra', [
    {'output_spiking': False}, {'encoder': {}}, {'paper_unet': {}},
    {'temporal_mamba': {'enabled': False}}, {'spatial_mamba': {}},
])
def test_yaml_rejects_other_architecture_options(extra):
    config = experiment()
    config['snn'].update(extra)
    with pytest.raises(ValueError):
        run_model._validate_experiment(config)


def test_arch2_section_requires_arch2_variant():
    config = experiment()
    config['snn']['variant'] = 'orig'
    with pytest.raises(ValueError, match='arch2'):
        run_model._validate_experiment(config)


@pytest.mark.parametrize('old_model', ['orig', 'paper_unet'])
def test_fresh_arch2_replaces_saved_model_and_persists_exact_options(tmp_path, old_model):
    old = deepcopy(DEFAULT_SNN_CONFIG)
    if old_model == 'paper_unet':
        old.update(model_name=old_model, model_kwargs={})
    path = tmp_path/'SNNPlans.json'
    path.write_text(json.dumps({'snn_config': old,
                               'configurations': {'3d_fullres': {'patch_size': [128]*3, 'batch_size': 2}}}))
    args = cli.build_parser().parse_args([
        'train', '--dataset-id', '1', '--fold', '0', '--model', 'arch2',
        '--arch2', json.dumps(options()), '--training-patch-size', '4', '16', '24', '--no-fptt'])
    config = cli._requested_config(args, path)
    assert config['model_name'] == 'arch2'
    assert config['model_kwargs']['encoder']['channels'] == [4,6,8]
    assert 'hierarchical_encoder' not in config['model_kwargs']
    update_snn_config(path, config, tmp_path/'results', training_patch_size=[4,16,24])
    saved = json.loads(path.read_text())
    assert SNNConfig.from_plans(saved).to_dict() == config
    # Verify the result-folder compatibility path also accepts this model.
    results = tmp_path/'results'
    results.mkdir()
    (results/'plans.json').write_text(path.read_text())
    update_snn_config(path, config, results)


def test_cli_resume_uses_saved_arch2_and_rejects_mixed_flags(tmp_path):
    path = tmp_path/'plans.json'
    path.write_text(json.dumps({'snn_config': settings()}))
    args = cli.build_parser().parse_args(['train', '--model', 'arch2', '--dataset-id', '1', '--fold', '0', '--c'])
    assert cli._requested_config(args, path)['model_kwargs']['encoder']['channels'] == [4,6,8]
    args = cli.build_parser().parse_args(['train', '--model', 'arch2', '--dataset-id', '1', '--fold', '0',
                                         '--no-output-spiking'])
    with pytest.raises(ValueError, match='incompatible'):
        cli._requested_config(args, path)


def test_launcher_dispatches_exact_arch2_options(tmp_path, monkeypatch):
    config = experiment()
    path = tmp_path/'config.yaml'
    config['run_dir'] = str(tmp_path/'run')
    path.write_text(yaml.safe_dump(config))
    (tmp_path/'preprocessed'/'Dataset001_Test').mkdir(parents=True)
    monkeypatch.setenv('nnUNet_raw', str(tmp_path/'raw'))
    monkeypatch.setenv('nnUNet_preprocessed', str(tmp_path/'preprocessed'))
    monkeypatch.setattr(run_model, '_require_preprocessed_configuration', lambda *args: None)
    calls = []
    monkeypatch.setattr(cli, 'main', calls.append)
    run_model.run(path)
    args = cli.build_parser().parse_args(calls[0])
    assert args.model == 'arch2'
    assert args.arch2 == run_model._config(path)['snn']['arch2']
    assert args.training_patch_size == [4,16,24]


@pytest.mark.parametrize('axis', [0,1,2])
def test_plans_reconstruction_chunking_and_checkpoint_roundtrip(monkeypatch, tmp_path, axis):
    import snn_nnunet.spikmamba as mixers
    monkeypatch.setattr(mixers, 'selective_scan_fn', reference_scan())
    value = settings()
    value['temporal_axis'] = axis
    config = SNNConfig.from_plans({'snn_config': value})
    network = SNNnnUNetAdapter(config).eval()
    data = torch.randn(1,4,8,9,10)
    expected = network(data, chunk_size=10)
    torch.testing.assert_close(network(data), expected, rtol=0, atol=0)
    path = tmp_path/'weights.pth'
    torch.save({'snn_config': config.to_dict(), 'network_weights': network.state_dict()}, path)
    checkpoint = torch.load(path, weights_only=False)
    restored = SNNnnUNetAdapter(SNNConfig.from_plans({'snn_config': checkpoint['snn_config']})).eval()
    restored.load_state_dict(checkpoint['network_weights'])
    assert restored(data).shape == (1,3,8,9,10)
    torch.testing.assert_close(restored(data), expected, rtol=0, atol=0)


@pytest.mark.parametrize('fptt', [False, True])
def test_native_trainer_loss_updates_weights_and_online_metrics(monkeypatch, fptt):
    import snn_nnunet.spikmamba as mixers
    from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss
    from snn_nnunet.trainer import nnUNetTrainerSNNFPTT
    monkeypatch.setattr(mixers, 'selective_scan_fn', reference_scan())
    value = settings()
    value['use_fptt'] = fptt
    config = SNNConfig.from_plans({'snn_config': value})
    network = SNNnnUNetAdapter(config).train()
    trainer = object.__new__(nnUNetTrainerSNNFPTT)
    trainer.snn_config, trainer.network = config, network
    trainer.device, trainer.current_epoch = torch.device('cpu'), 0
    trainer.initial_lr, trainer.weight_decay, trainer.num_epochs = .01, 3e-5, 10
    trainer.enable_deep_supervision = False
    trainer.optimizer, trainer.lr_scheduler = trainer.configure_optimizers()
    if fptt:
        from snn_nnunet import fptt as fptt_module
        fptt_module.init_running_params(network)
    trainer.loss = DC_and_BCE_loss({}, {'batch_dice': False, 'do_bg': True, 'smooth': 1e-5, 'ddp': False})
    trainer.label_manager = SimpleNamespace(has_regions=True, has_ignore_label=False)
    before = network.core.class_conv.conv.weight.detach().clone()
    batch = {'data': torch.randn(2,4,4,16,16), 'target': torch.randint(0,2,(2,3,4,16,16)).float()}
    metrics = trainer.train_step(batch)
    assert torch.isfinite(torch.tensor(metrics['loss']))
    assert not torch.equal(before, network.core.class_conv.conv.weight)
    network.eval()
    with torch.no_grad():
        validation = trainer.validation_step(batch)
    assert validation['tp_hard'].shape == (3,)
    assert torch.isfinite(torch.tensor(validation['loss']))


def checkpoint_trainer(config):
    from nnunetv2.training.logging.nnunet_logger import LocalLogger
    from snn_nnunet.trainer import nnUNetTrainerSNNFPTT
    from snn_nnunet.fptt import init_running_params
    trainer = object.__new__(nnUNetTrainerSNNFPTT)
    trainer.snn_config = config
    trainer.network = SNNnnUNetAdapter(config)
    trainer.device = torch.device('cpu')
    trainer.local_rank, trainer.is_ddp = 0, False
    trainer.disable_checkpointing, trainer.was_initialized = False, True
    trainer.grad_scaler, trainer._best_ema, trainer.current_epoch = None, .6, 3
    trainer.initial_lr, trainer.weight_decay, trainer.num_epochs = .01, 3e-5, 10
    trainer.optimizer, trainer.lr_scheduler = trainer.configure_optimizers()
    trainer.logger = LocalLogger()
    trainer.my_init_kwargs = {'plans': {'snn_config': config.to_dict()}}
    trainer.inference_allowed_mirroring_axes = (1,2)
    trainer.reproducibility = {'seed': config.seed}
    trainer.print_to_log_file = lambda *args, **kwargs: None
    if config.use_fptt:
        init_running_params(trainer.network)
    return trainer


@pytest.mark.parametrize('fptt', [False, True])
def test_actual_trainer_checkpoint_save_and_resume(monkeypatch, tmp_path, fptt):
    import snn_nnunet.spikmamba as mixers
    monkeypatch.setattr(mixers, 'selective_scan_fn', reference_scan())
    value = settings()
    value['use_fptt'] = fptt
    config = SNNConfig.from_plans({'snn_config': value})
    trainer = checkpoint_trainer(config)
    data = torch.randn(1,4,2,16,16)
    trainer.network(data).square().mean().backward()
    trainer.optimizer.step()
    trainer.optimizer.zero_grad()
    if fptt:
        from snn_nnunet.fptt import update_running_params
        update_running_params(trainer.network, config.fptt_alpha, config.fptt_beta)
    trainer.network.eval()
    expected = trainer.network(data)
    path = tmp_path/'checkpoint_latest.pth'
    trainer.save_checkpoint(str(path))
    saved = torch.load(path, weights_only=False)
    assert saved['snn_config'] == config.to_dict()
    restored = checkpoint_trainer(SNNConfig.from_plans(saved['init_args']['plans']))
    restored.load_checkpoint(str(path))
    assert restored.current_epoch == 4
    assert restored._best_ema == .6
    original_optimizer = trainer.optimizer.state_dict()
    restored_optimizer = restored.optimizer.state_dict()
    assert restored_optimizer['param_groups'] == original_optimizer['param_groups']
    assert original_optimizer['state']
    for key, state in original_optimizer['state'].items():
        for name, tensor in state.items():
            torch.testing.assert_close(restored_optimizer['state'][key][name], tensor)
    restored.network.eval()
    torch.testing.assert_close(restored.network(data), expected, rtol=0, atol=0)
    if fptt:
        for name in trainer.network.avg_weights:
            torch.testing.assert_close(restored.network.avg_weights[name], trainer.network.avg_weights[name])
            torch.testing.assert_close(restored.network.lambdas[name], trainer.network.lambdas[name])


@pytest.mark.parametrize('field', ['dropout', 'threshold', 'axis'])
def test_trainer_rejects_same_shape_architecture_changes_before_loading(tmp_path, field):
    config = SNNConfig.from_plans({'snn_config': settings()})
    trainer = checkpoint_trainer(config)
    path = tmp_path/'checkpoint.pth'
    trainer.save_checkpoint(str(path))
    changed = config.to_dict()
    if field == 'dropout':
        changed['model_kwargs']['dropout'] = .4
    elif field == 'threshold':
        changed['model_kwargs']['neurons']['v_threshold'] = 1.5
    else:
        changed['temporal_axis'] = 1
    restored = checkpoint_trainer(SNNConfig.from_plans({'snn_config': changed}))
    with pytest.raises(ValueError, match='arch2 architecture'):
        restored.load_checkpoint(str(path))


def test_actual_example_config_is_launchable():
    path = Path(__file__).resolve().parents[1]/'arch2_config.yaml'
    config = run_model._config(path)
    assert config['snn']['variant'] == 'arch2'
    assert config['snn']['arch2']['patch_size'] == 2
    assert config['data_partition'] == 'data_partitions/brats24_patient_seed42'


def test_native_validation_temporarily_uses_arch2_validation_geometry(monkeypatch):
    from contextlib import nullcontext
    import snn_nnunet.trainer as training
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    trainer = checkpoint_trainer(SNNConfig.from_plans({'snn_config': settings()}))
    trainer.configuration_manager = SimpleNamespace(configuration={'patch_size': [2,16,16]})
    observed = []
    monkeypatch.setattr(training, 'native_fp32_predictor', nullcontext)
    monkeypatch.setattr(training, 'native_empty_dice_evaluation', nullcontext)
    monkeypatch.setattr(nnUNetTrainer, 'perform_actual_validation',
                        lambda self, **kwargs: observed.append(self.configuration_manager.configuration['patch_size']))
    trainer.perform_actual_validation()
    assert observed == [[128,128,128]]
    assert trainer.configuration_manager.configuration['patch_size'] == [2,16,16]


@pytest.mark.parametrize('supervised', [False, True])
def test_actual_fp32_sliding_window_predictor_with_arch2(monkeypatch, supervised):
    import snn_nnunet.spikmamba as mixers
    from snn_nnunet.inference import FP32nnUNetPredictor
    monkeypatch.setattr(mixers, 'selective_scan_fn', reference_scan())
    value = settings()
    value['model_kwargs']['deep_supervision'] = {'enabled': supervised}
    network = SNNnnUNetAdapter(SNNConfig.from_plans({'snn_config': value})).eval()
    predictor = FP32nnUNetPredictor(tile_step_size=.5, use_gaussian=True, use_mirroring=False,
                                    perform_everything_on_device=False, device=torch.device('cpu'),
                                    allow_tqdm=False)
    predictor.manual_initialization(
        network, SimpleNamespace(get_label_manager=lambda data: SimpleNamespace(num_segmentation_heads=3)),
        SimpleNamespace(patch_size=[2,16,16]), None, {}, 'nnUNetTrainerSNNFPTT', ())
    data = torch.randn(4,3,17,19)
    first = predictor.predict_sliding_window_return_logits(data)
    second = predictor.predict_sliding_window_return_logits(data)
    assert first.shape == (3,3,17,19)
    assert first.dtype == torch.float32
    assert torch.isfinite(first).all()
    torch.testing.assert_close(first, second, rtol=0, atol=0)
