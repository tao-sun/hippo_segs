# Accelerate Multi-GPU Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the existing SNN FPTT training on one or more GPUs using only Hugging Face Accelerate's public API.

**Architecture:** `run_experiment` owns one `Accelerator`, prepares the model, optimizer, scheduler, and loaders, and delegates distributed-safe training and validation to the existing loops. Small reduction helpers isolate global loss and Dice aggregation; only the main process writes shared artifacts and checkpoints.

**Tech Stack:** Python 3.9, PyTorch, Hugging Face Accelerate 1.10.1, pytest, Slurm.

## Global Constraints

- Do not import or configure `torch.distributed`, `DistributedDataParallel`, or `DistributedSampler` directly.
- Preserve SNN, TBPTT, FPTT, checkpoint keys, and single-process execution.
- Keep mixed precision disabled by default.
- Treat `batch_size_subjects` as a per-process batch size.
- Support one Slurm node only.

---

### Task 1: Distributed-safe training utilities

**Files:**
- Create: `tests/test_accelerate_training.py`
- Modify: `snn_fptt.py`

**Interfaces:**
- Produces: `reduce_loss_totals(accelerator, loss_sum: torch.Tensor, update_count: torch.Tensor) -> float`
- Produces: `reduce_dice_totals(accelerator, dice_sum: torch.Tensor, subject_count: torch.Tensor) -> Dict[str, float]`
- Produces: `train_epoch_snn_tbptt(..., accelerator: Accelerator) -> float`
- Produces: `evaluate_3d_snn(..., accelerator: Accelerator) -> Dict[str, float]`

- [ ] **Step 1: Write failing unit tests for global reductions**

Create a fake accelerator whose `reduce` returns predefined global sums. Verify that loss uses global numerator/denominator and Dice returns ET, TC, WT, mean, and integer subject count from global totals.

- [ ] **Step 2: Run the focused test and confirm failure**

Run: `pytest -q tests/test_accelerate_training.py`

Expected: collection fails because the reduction helpers do not exist.

- [ ] **Step 3: Add Accelerate reduction helpers**

Use `accelerator.reduce(tensor, reduction="sum")`; clamp denominators to at least one and return ordinary Python values. Keep tensors on `accelerator.device`.

- [ ] **Step 4: Adapt training and validation loops**

Replace `.to(device)` with `.to(accelerator.device)`, `loss.backward()` with `accelerator.backward(loss)`, and `nn.utils.clip_grad_norm_` with `accelerator.clip_grad_norm_`. Use `accelerator.unwrap_model(model)` for FPTT dictionaries and custom state methods. Disable worker progress bars and firing-rate reports with `accelerator.is_local_main_process`. Reduce loss and per-class Dice totals globally before returning.

- [ ] **Step 5: Run focused and existing tests**

Run: `pytest -q tests/test_accelerate_training.py tests/test_ssm_module.py tests/test_subject_limit.py tests/test_overfit_subset.py`

Expected: all tests pass.

### Task 2: Accelerate orchestration and process-safe artifacts

**Files:**
- Modify: `snn_fptt.py`
- Modify: `tests/test_accelerate_training.py`

**Interfaces:**
- Consumes: reduction and loop interfaces from Task 1.
- Produces: `run_experiment(exp_cfg: Dict, config_path: Optional[str] = None)` runnable under either Python or `accelerate launch`.

- [ ] **Step 1: Add a failing single-process orchestration smoke test**

Mock expensive dataset/model construction and verify an `Accelerator` is constructed, `prepare` is called with model, optimizer, both loaders, and scheduler, and checkpoint state comes from `unwrap_model`.

- [ ] **Step 2: Run the smoke test and confirm failure**

Run: `pytest -q tests/test_accelerate_training.py -k orchestration`

Expected: failure because `run_experiment` does not construct or use `Accelerator`.

- [ ] **Step 3: Integrate Accelerator into `run_experiment`**

Construct `Accelerator()` before filesystem setup. Use `accelerator.device`, prepare all trainable/runtime objects, initialize FPTT state on the unwrapped model, and call the adapted loops. Guard directory creation, source copying, log/CSV/YAML writes, console summaries, and checkpointing with `accelerator.is_main_process`; bracket shared setup and finalization with `wait_for_everyone()`.

- [ ] **Step 4: Preserve scheduler and checkpoint correctness**

All processes call `scheduler.step` with the globally reduced training loss. Save only on the main process using `accelerator.save` and `accelerator.unwrap_model(model).state_dict()`, retaining keys `model`, `epoch`, `dice_mean`, and `config`.

- [ ] **Step 5: Run tests and compile check**

Run: `pytest -q && python -m py_compile snn_fptt.py`

Expected: all tests pass and compilation exits zero.

### Task 3: Dependency and Slurm launch

**Files:**
- Modify: `requirements.txt`
- Modify: `snn_fptt.job`

**Interfaces:**
- Consumes: Accelerate-compatible `snn_fptt.py` from Task 2.
- Produces: one-node launch with one process per allocated GPU.

- [ ] **Step 1: Add Accelerate dependency**

Add `accelerate>=1.10,<2` to `requirements.txt`.

- [ ] **Step 2: Request four GPUs and launch through Accelerate**

Set `#SBATCH --gpus=4`. Validate `SLURM_GPUS_ON_NODE` as a positive integer with fallback `1`, then run:

```bash
accelerate launch --num_machines 1 --num_processes "$NUM_PROCESSES" \
  snn_fptt.py --config "$SLURM_SUBMIT_DIR/experiments_snn_fptt.yaml"
```

- [ ] **Step 3: Verify dependency, shell syntax, and CLI availability**

Run: `bash -n snn_fptt.job && source .venv/bin/activate && python -c 'import accelerate; print(accelerate.__version__)' && accelerate launch --help >/dev/null`

Expected: shell syntax succeeds, Accelerate reports version 1.10.1, and help exits zero.

- [ ] **Step 4: Run complete verification**

Run: `pytest -q && python -m py_compile snn_fptt.py && git diff --check`

Expected: tests pass, compilation succeeds, and no whitespace errors are reported.
