from pathlib import Path
import subprocess

import pytest
import torch
import yaml

from model import SNNBraTS
import snn_fptt as training
import evaluate_snn_fold_view as evaluation


class SelectiveScanIdentity:
    def __call__(self, scan_input, *args, **kwargs):
        return scan_input


@pytest.mark.parametrize('enabled', [False, True])
def test_skip_forward_backward_and_checkpoint_spec(enabled):
    model = SNNBraTS(out_channels=3, patch_size=2,
                     selective_scan=SelectiveScanIdentity(), input_skip=enabled)
    model.eval()
    x = torch.rand(1, 2, 4, 9, 11, requires_grad=True)
    result = model(x)
    assert result.shape == (1, 3, 2, 9, 11)
    result.square().mean().backward()
    assert torch.isfinite(result).all()
    if enabled:
        assert torch.isfinite(x.grad).all()
        assert model.input_proj.weight.grad.abs().sum() > 0
        assert model.input_fuse.weight.grad.abs().sum() > 0
    else:
        legacy = SNNBraTS(out_channels=3, patch_size=2, selective_scan=SelectiveScanIdentity())
        legacy.load_state_dict(model.state_dict(), strict=True)
    cfg = dict(model='orig', patch_size=2, linear_projection=True,
               residual_connections=True, dwconv2d_spiking=True,
               patch_embedding_spiking=False, input_skip=enabled)
    _, kwargs = evaluation.read_model_spec(dict(model=model.state_dict(), config=cfg))
    assert kwargs['input_skip'] is enabled
    restored = training.build_model(**kwargs)
    restored.load_state_dict(model.state_dict(), strict=True)


def test_skip_config_validation_and_default(tmp_path):
    cfg = yaml.safe_load(Path('experiments_snn_fptt.yaml').read_text())
    cfg.pop('input_skip', None)
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(cfg))
    assert training.load_experiment_from_yaml(str(path))['input_skip'] is False
    cfg['input_skip'] = 'true'
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='input_skip'):
        training.load_experiment_from_yaml(str(path))
    cfg.update(input_skip=True, model='shallow')
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='orig'):
        training.load_experiment_from_yaml(str(path))


def test_skip_resume_mismatch_rejected_before_metadata_write(tmp_path, monkeypatch):
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    torch.save({'config': {'use_fptt': False}}, tmp_path / 'checkpoint_last.pt')
    metadata = tmp_path / 'hyperparameters.yaml'
    metadata.write_text('original')
    with pytest.raises(ValueError, match='input_skip'):
        training.run_experiment(dict(name='test', resume_from=str(tmp_path),
                                     use_fptt=False, resume_scheduler=False, input_skip=True))
    assert metadata.read_text() == 'original'
    assert not (tmp_path / 'train.out').exists()


def test_tmux_name_queries_launch_pane(monkeypatch):
    monkeypatch.setenv('TMUX', '/tmp/tmux-test/default,123,4')
    monkeypatch.setenv('TMUX_PANE', '%7')
    def run(command, **kwargs):
        assert command == ['tmux', '-S', '/tmp/tmux-test/default', 'display-message',
                           '-p', '-t', '%7', '#{session_name}']
        assert kwargs['timeout'] > 0
        return subprocess.CompletedProcess(command, 0, 'training-session\n', '')
    monkeypatch.setattr(training.subprocess, 'run', run)
    assert training.get_tmux_session_name() == 'training-session'


def test_tmux_unavailable_is_nonfatal(monkeypatch):
    monkeypatch.delenv('TMUX', raising=False)
    assert training.get_tmux_session_name() is None
    monkeypatch.setenv('TMUX', '/tmp/default,1,0')
    monkeypatch.setenv('TMUX_PANE', '%1')
    def fail(*args, **kwargs):
        raise FileNotFoundError('tmux')
    monkeypatch.setattr(training.subprocess, 'run', fail)
    assert training.get_tmux_session_name() is None
