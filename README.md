# hippo_segs

## Running `snn_fptt.py`

### 1. Create the environment

The project is managed with [uv](https://docs.astral.sh/uv/). From the
repository root, create the Python 3.9 environment and install the locked
dependencies:

```bash
uv sync --frozen --python 3.9
```

### 2. Download and preprocess the BraTS data

Download the BraTS dataset and place its original ZIP archives in a dedicated
input directory. The archives do not need to be extracted: the preprocessing
script reads subjects directly from the ZIP files, processes one subject at a
time, and writes training-ready `.pt` cache files.

Run the preprocessing separately for every dataset or preprocessing variant.
Keep different BraTS releases in dedicated input directories. The same input
directory may be reused to compare preprocessing variants, but every dataset
and normalization method must have its own `--cache-root`. Multiple anatomical
views may share one cache root because each view is stored in its own
subdirectory.

For the current BraTS24 experiments, we are testing foreground Z-score
normalization instead of the previous min-max normalization. The normalized
values are kept as `float32` rather than being quantized to `uint8`:

```bash
uv run python data/preprocess_brats24.py \
  --input /path/to/folder-containing-brats24-zips \
  --input-mode archives \
  --cache-root /path/to/BRATS2024_cache_zscore \
  --preprocessing-normalization zscore \
  --preserve-float32 \
  --view sagittal \
  --workers 1
```

Replace `sagittal` with `axial` or `coronal`, or pass multiple views after
`--view`, when required by the experiment. Normalization is applied to each
3D modality before view extraction. The preprocessing command is restartable:
completed subjects are validated and skipped when it is run again with the
same configuration.

The options in the training YAML must match the cache. For the legacy
min-max/`uint8` baseline, use
`--preprocessing-normalization minmax`, omit `--preserve-float32`, and write to
a different cache directory.

For already preprocessed legacy datasets, `build_brats_cache.py` can still be
used to convert the existing files to the subject-cache format. The direct
`data/preprocess_brats24.py` ZIP-to-cache workflow above is recommended for
BraTS24.

### 3. Configure the experiment

Edit `experiments_snn_fptt.yaml` before starting the training. At minimum,
choose the experiment name, validation fold, anatomical view, cache location,
model, batch size, learning rate, and number of epochs.

For the Z-score cache created above, the relevant data options are:

```yaml
data_root: null
cache_root: /path/to/BRATS2024_cache_zscore
cache_required: true
label_format: brats24
view: sagittal
val_fold: 1

preprocessing_normalization: zscore
preserve_float32: true
```

Keeping `cache_required: true` is recommended because it prevents training
from silently falling back to slower source-file loading. `cache_root` must
point to the output of the matching preprocessing run. With a required cache,
`data_root` is not used as a separate image source and may be set to `null`;
the loader then uses `cache_root` for dataset discovery as well. A real
`data_root` is only needed when cache fallback is allowed or when using an
uncached legacy dataset.

The main experimental switches are:

```yaml
# Weight the probability of selecting training patients according to lesion
# volume. Sampling changes which patients are seen during an epoch, but not
# the total number of draws per epoch.
lesion_volume_sampling: true
lesion_sampling_classes: [ET, TC]
lesion_sampling_gamma: 0.5
lesion_sampling_max_weight: 4.0

# true enables FPTT; false uses plain TBPTT and ignores the FPTT parameters.
use_fptt: true

# Postprocess thresholded prediction volumes during evaluation.
apply_postprocessing: true

# Minimum connected-component sizes, in voxels, in ET, TC, WT order.
# Use non-zero values to remove components smaller than these thresholds.
min_component_sizes: [50, 50, 50]
closing_radius: 1
```

Set `lesion_volume_sampling: false`, `use_fptt: false`, or
`apply_postprocessing: false` to disable the corresponding feature. When
postprocessing is disabled, `min_component_sizes` and `closing_radius` do not
affect predictions. Postprocessing is applied only to predictions; the ground
truth is never modified.

`loader_workers` is configured per Accelerate process. For example, four GPU
processes with `loader_workers: 2` create eight DataLoader workers in total.
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

### 4. Start the training

Run a single-process experiment from the repository root with:

```bash
uv run accelerate launch --num_processes 1 \
  snn_fptt.py --config experiments_snn_fptt.yaml
```

For multi-GPU training, set the number of processes to the number of GPUs and
add `--multi_gpu`, for example:

```bash
uv run accelerate launch --multi_gpu --num_processes 4 \
  snn_fptt.py --config experiments_snn_fptt.yaml
```

The supplied `snn_fptt.job` provides the equivalent SLURM workflow. Its first
argument may be used to select a YAML file other than the default:

```bash
sbatch snn_fptt.job /path/to/experiment.yaml
```
