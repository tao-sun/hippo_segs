"""Exercise the real nnU-Net epoch hook without allocating a training network."""

from copy import deepcopy
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from nnunetv2.training.logging.nnunet_logger import LocalLogger

from snn_nnunet.trainer import _EPOCH_METRICS, nnUNetTrainerSNNFPTT


def _trainer(distributed=False):
    trainer = nnUNetTrainerSNNFPTT.__new__(nnUNetTrainerSNNFPTT)
    trainer.snn_config = SimpleNamespace(use_fptt=False, k=1)
    trainer.current_epoch = 0
    trainer.is_ddp = distributed
    local_logger = LocalLogger()
    for key in _EPOCH_METRICS:
        local_logger.my_fantastic_logging[key] = []
    trainer.logger = SimpleNamespace(local_logger=local_logger, log=local_logger.log)
    return trainer


def _outputs(sample_diagnostics=True):
    key = 'recurrent/encoder/state/mean_abs'
    return [
        {'loss': 2., 'task_loss': 1., 'fptt_regularization': 1., 'total_loss': 2.,
         'window_count': 2, 'optimizer_updates': 2, 'k': 1, 'fptt_mode': False,
         'recurrent_diagnostics': {key: .2} if sample_diagnostics else {},
         'recurrent_diagnostic_counts': {key: 1} if sample_diagnostics else {}},
        {'loss': 6., 'task_loss': 4., 'fptt_regularization': 2., 'total_loss': 6.,
         'window_count': 6, 'optimizer_updates': 6, 'k': 1, 'fptt_mode': False,
         'recurrent_diagnostics': {key: .6} if sample_diagnostics else {},
         'recurrent_diagnostic_counts': {key: 3} if sample_diagnostics else {}},
    ]


def _assert_metrics(trainer, updates, sample_diagnostics=True):
    history = trainer.logger.local_logger.my_fantastic_logging
    assert history['train_losses'] == pytest.approx([4.])
    assert history['task_losses'] == pytest.approx([3.25])
    assert history['fptt_regularization'] == pytest.approx([1.75])
    assert history['total_losses'] == pytest.approx([5.])
    assert history['optimizer_updates'] == [updates]
    assert history['k'] == [1]
    assert history['fptt_mode'] == [False]
    key = 'recurrent/encoder/state/mean_abs'
    if sample_diagnostics:
        assert history[key] == pytest.approx([.5])
    else:
        assert key not in history


@pytest.mark.parametrize('sample_diagnostics', [False, True])
def test_epoch_end_preserves_loss_and_nested_diagnostics(sample_diagnostics):
    trainer = _trainer()
    outputs = _outputs(sample_diagnostics)
    before = deepcopy(outputs)
    trainer.on_train_epoch_end(outputs)
    _assert_metrics(trainer, 8, sample_diagnostics)
    assert outputs == before


def _distributed_epoch_worker(rank, rendezvous):
    dist.init_process_group('gloo', init_method=f'file://{rendezvous}', rank=rank,
                            world_size=2, timeout=timedelta(seconds=45))
    try:
        trainer = _trainer(distributed=True)
        trainer.on_train_epoch_end([_outputs()[rank]])
        _assert_metrics(trainer, 4)
    finally:
        dist.destroy_process_group()


def test_epoch_end_aggregates_loss_and_diagnostics_across_real_workers(tmp_path):
    torch.multiprocessing.spawn(_distributed_epoch_worker,
        args=(str(tmp_path / 'rendezvous'),), nprocs=2, join=True)
