# hippo_segs

## Learning-rate scheduling and resume

Training schedulers advance once per epoch, regardless of the number of GPUs.
`reduce_plateau` monitors training loss with PyTorch's defaults: ten tolerated
epochs without improvement and a factor of 0.1 at each reduction.

On resume, `resume_scheduler: false` resets scheduler history while preserving
the checkpoint's optimizer state and learning rate. To explicitly restart at a
different learning rate, also set `resume_lr: 0.0001`. Set `resume_lr: null` to
keep the checkpoint learning rate, restore the scheduler with
`resume_scheduler: true`, or start a new run with `resume_from: null`.
An explicit `resume_lr` applies on every restart until it is cleared.

## Evaluation throughput

`eval_batch_subjects` controls the number of validation subjects per GPU,
independently of `batch_size_subjects` used for training. The example config
uses 2; older configs that omit it keep 1. Try 4 if GPU and host memory permit.
The same setting applies to the optional training-set evaluation loader.

`eval_batch_slices` remains the sequential slice window length, not a parallel
image batch. Neuron states reset at the start of each subject batch and remain
independent between subjects. Evaluation accumulates integer intersection and
voxel counts on the device, then computes each subject's ET/TC/WT Dice with the
existing threshold and epsilon. Only the small Dice table is copied to the CPU.
The reported score remains the mean of subject Dice, including incomplete
batches; Accelerate removes distributed padding duplicates before averaging.

These settings are read at startup; an already running process keeps its loaded
code and configuration. Measure evaluation time and memory on the target GPUs
before assuming a speedup from a larger batch.

## BraTS24 preprocessing and cache

BraTS24 preprocessing writes only the subject cache, directly from
the raw NIfTI volumes in memory. No intermediate PNGs or NIfTI masks are saved:

```bash
.venv/bin/python data/preprocess_brats24.py \
  --input /home/aurora/data/BRATS2024 \
  --cache-root /home/aurora/data/BRATS2024_subject_cache \
  --view sagittal \
  --workers 4
```

The selected view is saved under `<cache-root>/<view>/<fold>/<subject>.pt`,
with fold and source manifests alongside the cache. Existing valid cache files
are skipped; pass
`--overwrite-cache` to rebuild them. By default it includes both BraTS24
training folders. Pass `--exclude-additional` to use only `training_data1_v2`.

For training without a preprocessing directory, configure:

```yaml
data_root: /home/aurora/data/BRATS2024_subject_cache
cache_root: /home/aurora/data/BRATS2024_subject_cache
cache_required: true
label_format: brats24
```

The loader uses the cache manifests to discover subjects and checks for missing
files. Fold assignments still follow `FOLD_NAMES` in `data/preprocess_brats24.py`.
Keep the raw input volumes to regenerate other views or rebuild the cache.

## BraTS subject cache

`snn_fptt.py` can read one lossless uint8 `.pt` file per subject instead of
opening every modality PNG and the NIfTI segmentation at every epoch. Build
the cache once, separately from training:

```bash
.venv/bin/python build_brats_cache.py \
  --data-root /gpfs/scratch1/shared/apiaghiardelli/BRATS2023_preprocessed/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData \
  --cache-root /gpfs/scratch1/shared/apiaghiardelli/BRATS2023_subject_cache \
  --view sagittal \
  --workers 4
```

The command is restartable. It validates and skips complete cache files,
rebuilds incompatible files, writes each subject atomically, and creates
`<cache-root>/<view>/manifest.json`. Use `--overwrite` to rebuild every
subject deliberately.

Enable the completed cache in `experiments_snn_fptt.yaml`:

```yaml
cache_root: /gpfs/scratch1/shared/apiaghiardelli/BRATS2023_subject_cache
cache_required: true
loader_workers: 2
loader_prefetch_factor: 1
```

`loader_workers` is per Accelerate process. With four GPUs, two workers means
eight DataLoader workers in total. Cached and original loading return the
same float32 images, ET/TC/WT labels, and metadata, so existing training
checkpoints remain compatible.
