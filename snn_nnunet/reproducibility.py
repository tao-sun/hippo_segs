"""Strict fresh-run reproducibility and ordered native nnU-Net augmentation."""
import os
import platform
import random

import numpy as np
import torch


POLICY_FIELDS = ('seed', 'deterministic', 'augmentation_workers')


def validate_policy(seed, deterministic, augmentation_workers):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be an integer in [0, 2**32)')
    if type(deterministic) is not bool:
        raise ValueError('deterministic must be a boolean')
    if type(augmentation_workers) is not int or not 1 <= augmentation_workers <= 64:
        raise ValueError('augmentation_workers must be an integer between 1 and 64')


def training_environment(seed, deterministic, environment=None):
    """Set interpreter/cuBLAS options before the fresh Accelerate worker starts."""
    result = dict(os.environ if environment is None else environment)
    if deterministic:
        result['PYTHONHASHSEED'] = str(seed)
        result['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    return result


def configure_reproducibility(seed, deterministic, rank=0):
    """Call before model initialization; disabled mode preserves legacy behavior."""
    effective_seed = (seed + rank) % 2**32
    if deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        random.seed(effective_seed)
        np.random.seed(effective_seed)
        torch.manual_seed(effective_seed)
        torch.cuda.manual_seed_all(effective_seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    return {
        'seed': seed, 'effective_seed': effective_seed if deterministic else None,
        'deterministic': deterministic, 'rank': rank,
        'world_size': int(os.environ.get('WORLD_SIZE', '1')),
        'python': platform.python_version(), 'torch': str(torch.__version__),
        'numpy': np.__version__, 'cuda': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(),
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_tf32': torch.backends.cudnn.allow_tf32,
        'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'pythonhashseed': os.environ.get('PYTHONHASHSEED'),
    }


class _SeededWorkerLoader:
    """Seed every CPU RNG after fork/spawn, not just NumPy in the producer."""
    def __init__(self, loader, seeds):
        self.loader, self.seeds = loader, seeds

    def set_thread_id(self, worker):
        seed = self.seeds[worker]
        random.seed(seed)
        np.random.seed(seed)
        # Workers do CPU augmentation; avoid touching an inherited CUDA context.
        torch.random.default_generator.manual_seed(seed)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True, warn_only=False)
        if hasattr(self.loader, 'rs'):
            self.loader.rs.seed(seed)
        self.loader.set_thread_id(worker)

    def __next__(self):
        return next(self.loader)


def ordered_augmenter(loader, *, seed, rank, stream, num_workers, pin_memory):
    from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
    offsets = {'train': 10000, 'validation': 20000}
    base = seed + 1000003 * rank + offsets[stream]
    seeds = [(base + worker) % 2**32 for worker in range(num_workers)]
    if hasattr(loader, 'indices'):
        loader.indices = sorted(loader.indices)
    return MultiThreadedAugmenter(
        _SeededWorkerLoader(loader, seeds), transform=None, num_processes=num_workers,
        num_cached_per_queue=2, seeds=seeds, pin_memory=pin_memory, wait_time=.002,
    )


def deterministic_transforms(transform):
    """Keep augmentation, but choose blur convolution without timing benchmarks."""
    from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
    if isinstance(transform, GaussianBlurTransform):
        transform.benchmark = False
    for child in getattr(transform, 'transforms', ()):
        deterministic_transforms(child)
    child = getattr(transform, 'transform', None)
    if child is not None:
        deterministic_transforms(child)
    return transform


def deterministic_dataloaders(trainer):
    """Same transforms and patches as nnU-Net 2.8.1, with ordered seeded queues."""
    from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
    from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
    if trainer.dataset_class is None:
        trainer.dataset_class = infer_dataset_class(trainer.preprocessed_dataset_folder)
    patch = trainer.configuration_manager.patch_size
    scales = trainer._get_deep_supervision_scales()
    rotation, dummy_2d, initial_patch, mirror = trainer.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
    common = dict(is_cascaded=trainer.is_cascaded,
                  foreground_labels=trainer.label_manager.foreground_labels,
                  regions=trainer.label_manager.foreground_regions if trainer.label_manager.has_regions else None,
                  ignore_label=trainer.label_manager.ignore_label)
    train_transform = trainer.get_training_transforms(
        patch, rotation, scales, mirror, dummy_2d,
        use_mask_for_norm=trainer.configuration_manager.use_mask_for_norm, **common)
    val_transform = trainer.get_validation_transforms(scales, **common)
    deterministic_transforms(train_transform)
    deterministic_transforms(val_transform)
    datasets = trainer.get_tr_and_val_datasets()
    config = trainer.snn_config
    workers = (config.augmentation_workers, max(1, config.augmentation_workers // 2))
    generators = []
    try:
        for stream, dataset, initial_size, transform, count in zip(
                ('train', 'validation'), datasets, (initial_patch, patch),
                (train_transform, val_transform), workers):
            loader = nnUNetDataLoader(
                dataset, trainer.batch_size, initial_size, patch, trainer.label_manager,
                oversample_foreground_percent=trainer.oversample_foreground_percent,
                sampling_probabilities=None, pad_sides=None, transforms=transform,
                probabilistic_oversampling=trainer.probabilistic_oversampling)
            generators.append(ordered_augmenter(
                loader, seed=config.seed, rank=trainer.local_rank, stream=stream,
                num_workers=count, pin_memory=trainer.device.type == 'cuda'))
        # Match native startup warmup; the same batches are discarded every run.
        for generator in generators:
            next(generator)
    except BaseException:
        for generator in generators:
            generator._finish()
        raise
    return tuple(generators)
