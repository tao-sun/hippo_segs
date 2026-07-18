# Accelerate multi-GPU training design

## Goal

Make `snn_fptt.py` run using only the public Hugging Face Accelerate API for
data-parallel training on one Slurm node with one or more GPUs. The project
code must not import or configure `torch.distributed`,
`DistributedDataParallel`, or `DistributedSampler` directly; Accelerate owns
the distributed backend. Preserve the existing SNN,
TBPTT, FPTT, validation, metrics CSV, and checkpoint format. A normal
single-process Python launch must remain usable.

## Considered approaches

1. **Hugging Face Accelerate only (selected).** One process runs per GPU.
   Project code interacts only with Accelerate, which internally shards data
   loaders and synchronizes gradients. This is the smallest maintainable
   change and matches the requested API.
2. **PyTorch DistributedDataParallel directly.** It offers more low-level
   control, but requires custom process setup, samplers, collectives, and
   checkpoint coordination that Accelerate already supplies.
3. **`torch.nn.DataParallel`.** It is simpler to invoke but uses one process
   as a bottleneck and is not the recommended basis for scalable training.

## Training architecture

`run_experiment` creates one `Accelerator` per process. Model, optimizer,
training loader, validation loader, and scheduler are passed through
`accelerator.prepare`. Tensor placement uses `accelerator.device`.

The configured `batch_size_subjects` remains the per-process batch size. With
four GPUs, the effective global batch is therefore four times the configured
value. This gives conventional DDP semantics and avoids silently changing the
amount of data processed by each GPU.

The model returned by `accelerator.prepare` performs forward passes, while
the model returned by `accelerator.unwrap_model` is used for FPTT state
(`avg_weights` and `lambdas`), custom state detachment, firing-rate hooks, and
serialization. Project code does not access or depend on the underlying
distributed wrapper. This prevents wrapper-specific parameter names from
changing the keys used by the FPTT dictionaries.

The training loop uses `accelerator.backward(loss)` and
`accelerator.clip_grad_norm_`. Epoch loss is reduced across processes using
the total loss numerator and update/sample denominator rather than averaging
already averaged process values.

## Validation and outputs

Accelerate shards validation subjects across processes. Each process computes
per-subject Dice values; per-class Dice sums and subject counts are reduced
across processes to produce one global result. Duplicate tail samples caused
by an uneven validation split must not affect the metric; validation will use
Accelerate's gathered metric handling or an equivalent deduplication-aware
reduction.

Only the main process will:

- create and copy run metadata;
- redirect output to `train.out`;
- write `hyperparameters.yaml` and `epoch_metrics.csv`;
- print global progress and metrics;
- save the best checkpoint.

All processes wait at synchronization points around shared filesystem setup
and checkpointing. The checkpoint continues to contain `model`, `epoch`,
`dice_mean`, and `config`, and the saved state dict comes from
`accelerator.unwrap_model(model)`.

## Slurm launch

`snn_fptt.job` requests four GPUs on one A100 node and launches four local
processes with `accelerate launch`. It derives the process count from
`SLURM_GPUS_ON_NODE`, with a safe numeric fallback, and binds one process to
each visible GPU. No multi-node support is included in this change.

`requirements.txt` records the Accelerate dependency already present in the
project virtual environment.

## Failure handling and compatibility

All ranks must enter training, validation, and synchronization in the same
order. Main-process-only file handles use a no-op path on worker processes so
workers never truncate shared files. The implementation will avoid mixed
precision by default because the numerical behavior of the custom SNN/FPTT
operations has not yet been validated under FP16 or BF16.

Launching with `python snn_fptt.py --config ...` remains a supported
single-process path. Launching with `accelerate launch --num_processes N`
activates distributed training without requiring a pre-generated Accelerate
configuration file.

## Verification

Tests will cover process-safe output decisions, unwrapped-model FPTT access,
and globally reduced loss/metrics using lightweight fakes where practical.
Existing tests must continue to pass. A CPU single-process smoke run will
validate imports and orchestration; the final Slurm command will be checked
with a short multi-GPU/debug configuration when cluster resources are
available.
