# Temporal weight decay, memory duration, and slice context

This note documents the current implementation and saved checkpoints inspected on 2026-11-03. The user confirmed the comparison pair via `slurm_snn_brats24_f0_train_val_28780460.out` and `slurm_snn_brats24_f0_train_val_28848785.err`. The numerical comparison below uses `SNN_paper_unet_sel_fold0` (job 28780460; temporal decay enabled) and `SNN_paper_unet_sel3_fold0` (job 28848785; a verified zero-decay temporal parameter group), at 50 completed epochs. This comparison is observational, not a controlled weight-decay ablation.

## What the switch controls

`temporal_weight_decay: false` exempts only each selective neuron's trainable `temporal_raw` tensor from optimizer weight decay. It does not freeze these parameters, remove memory leakage, or disable regularization of the whole network. Spatial kernels, readout gates, memory projections, soma parameters, and other model weights retain their normal optimizer decay. The saved optimizer uses SGD with momentum 0.99, Nesterov acceleration, and normal weight decay 3e-5. The verified no-decay run has a second parameter group with weight_decay=0.

The temporal memory buffers themselves are transient states, not trainable parameters. They reset for an independent patch and are not stored as patient memories in checkpoints. `temporal_fptt_regularization` is a separate switch; it affects training only when FPTT is enabled.

Implementation: `snn_nnunet/trainer.py::configure_optimizers`, `snn_nnunet/recurrent_neuron.py::SelectiveMultiTimescaleLIF`, and `snn_nnunet/recurrent_config.py::RECURRENT_DEFAULTS`.

## From raw parameters to memory duration

For compartment r=1,2,3 and channel c, let a[r,c] be `temporal_raw`. The implementation defines

    tau[r,c] = 1e-6 + sum_{j=1}^r softplus(a[j,c])
    lambda[r,c] = exp(-1 / tau[r,c])

The cumulative positive increments enforce ordered time constants and retention coefficients. A raw parameter for an earlier compartment also affects every later compartment's tau. The parameters are learned per channel; mean values conceal channel variation.

For one compartment, the actual slice recurrence is

    m[t] = lambda * A m[t-1] + (1-lambda) * x[t]
    A = (1-eta) I + eta K

where K is the bounded depthwise spatial convolution. Selective gates read the three memories and a learned projection adds their aggregate to the somatic input. Thus temporal retention is only one component of the network's use of context.

For A=I and zero initial memory,

    m[t] = (1-lambda) sum_{d=0}^{t-1} lambda^d x[t-d]

The contribution from a particular past slice is proportional to (1-lambda)*lambda^d. Its relative attenuation after d transitions is

    R(d) = lambda^d = exp(-d/tau)
    H_10% = tau * ln(10)

Tau is an e-folding distance: a pre-existing memory perturbation retains 36.8% of its amplitude after tau slice transitions. H_10% is the distance at which it retains 10%; neither is a hard cutoff or a count of slices the model necessarily uses. With spatial mixing enabled, the perturbation propagates as (lambda A)^d. For the configured nonexpansive A, lambda^d is an infinity-norm upper envelope, rather than an exact same-pixel retention curve. Readout gates, soma dynamics, and the rest of the network further change the influence on predictions.

Distances refer to consecutive slices along the preprocessed temporal axis, not automatically original acquisition slices or millimetres. This setup processes 81-slice independent patches; an intrinsic horizon cannot confer context beyond the patch. To express a horizon in millimetres, multiply by the corresponding preprocessed slice spacing.

## Why weight decay can shorten these memories

PyTorch SGD adds rho*a to the loss gradient before its momentum/Nesterov update:

    g = dL/da + rho*a

Ignoring momentum only for illustration,

    a_next = (1 - learning_rate*rho)*a - learning_rate*dL/da

The actual runs use momentum and Nesterov, so this simplified update is not their full trajectory. Weight decay acts on raw coordinates a, not directly on lambda, tau, or memory activations.

If regularization dominates and a tends to zero, then

    tau[r] -> r*ln(2) + 1e-6
    lambda[r] -> exp(-1 / (r*ln(2) + 1e-6))

| Compartment | Initial lambda | Initial tau (slices) | Initial H_10% (slices) | Tau at raw=0 | Lambda at raw=0 |
|---|---:|---:|---:|---:|---:|
| Fast | 0.50 | 1.443 | 3.322 | 0.693 | 0.2363 |
| Medium | 0.80 | 4.481 | 10.319 | 1.386 | 0.4861 |
| Slow | 0.95 | 19.496 | 44.891 | 2.079 | 0.6182 |

These are the coordinates' zero-penalty attractor, not a guaranteed trained endpoint. The initialization uses positive raw increments, so shrinking them shortens the initial memories substantially. If raw values were negative, shrinking them toward zero could instead lengthen memory. Consequently, 'weight decay always makes memory shorter' is not a general mathematical statement.

Removing temporal decay lets task gradients determine these coordinates without that explicit pull toward zero. It does not guarantee long memory or prevent task gradients from shortening it. The parameter sensitivity is

    d lambda[r] / d a[j] = lambda[r] * sigmoid(a[j]) / tau[r]^2, for j <= r

and zero for j>r. Long time constants can therefore have small lambda sensitivity to a unit raw-parameter change. Remaining near initialization is not evidence that a parameter is frozen, nor proof that long context is being used effectively.

## Checkpoint observations at matched epoch 50

Values below are channel means computed directly from each checkpoint's `temporal_raw` tensors. Tau is transformed per channel before averaging; it is not reconstructed from mean lambda.

| Run | Block | Fast lambda | Medium lambda | Slow lambda | Slow tau (slices) | Slow H_10% (slices) |
|---|---|---:|---:|---:|---:|---:|
| Decay enabled (`sel`) | conv_block1 | 0.2683 | 0.5279 | 0.7080 | 2.902 | 6.682 |
| Decay enabled (`sel`) | conv_block2 | 0.2557 | 0.5207 | 0.7042 | 2.852 | 6.567 |
| Decay enabled (`sel`) | conv_block3 | 0.2558 | 0.5206 | 0.7042 | 2.851 | 6.565 |
| Temporal decay disabled (`sel3`) | conv_block1 | 0.4941 | 0.7988 | 0.9499 | 19.466 | 44.823 |
| Temporal decay disabled (`sel3`) | conv_block2 | 0.4998 | 0.7999 | 0.9500 | 19.494 | 44.887 |
| Temporal decay disabled (`sel3`) | conv_block3 | 0.5004 | 0.8001 | 0.9500 | 19.498 | 44.897 |

For illustration, a scalar lambda=0.704 retains about 3% after 10 transitions and 0.09% after 20; lambda=0.95 retains about 60% and 36%, respectively. These describe direct memory attenuation, not segmentation accuracy or an end-to-end attribution measurement.

At epoch 100, the enabled run's conv_block3 means are lambda=(0.2389, 0.4909, 0.6307) and tau=(0.699, 1.405, 2.169) slices, close to the raw=0 attractor. This pattern is consistent with substantial regularization-driven contraction. Existing `analysis_results/run_28780460/summary.json` also records a decay/FPTT-only simulation closely matching that block; it is a simulation, not an actual no-decay training run.

## Why the two runs do not isolate weight decay

| Setting | `SNN_paper_unet_sel_fold0` | `SNN_paper_unet_sel3_fold0` |
|---|---|---|
| Temporal optimizer decay | 3e-5 | 0 |
| Other optimizer decay | 3e-5 | 3e-5 |
| Training method | FPTT | Full-patch BPTT |
| k | 1 slice | 81 slices |
| Patch | 81x128x128 | 81x128x128 |
| Planned epochs | 200 | 200 |
| Initial learning rate | 0.001 | 0.001 |
| Saved explicit seed/determinism | Not recorded in the inspected fields | Seed 42; deterministic=true |
| Epoch-50 raw full-fold mean Dice | 0.6500 | 0.6259 |
| Epoch-50 postprocessed full-fold mean Dice | 0.6723 | 0.7112 |

Both validation histories contain 270 cases at epoch 50. The disabled run is higher after postprocessing by 0.0389 absolute Dice, but lower before postprocessing by 0.0241. Reporting only the postprocessed improvement would omit a material difference.

The training loop takes an optimizer step per temporal window. For an 81-slice patch, k=1 therefore gives 81 updates, versus one update with k=81. This changes exposure to weight decay and momentum per epoch, as well as task gradients and FPTT regularization. A matched epoch does not imply a matched number of optimizer updates.

Crucially, k is a gradient-window length, not the memory time constant. Detaching states preserves their numerical values while truncating the gradient graph. With k=1, earlier slices can still influence later predictions through the carried state; standard backpropagation through those earlier state updates is truncated. With k=81, gradients can cross the whole patch. FPTT additionally changes the optimization objective and parameter handling.

These observations support: 'The no-temporal-decay/full-patch-BPTT configuration preserved much longer intrinsic memory timescales than the decay-enabled/FPTT configuration.' They do not establish: 'Disabling weight decay alone caused the Dice improvement.'

## A naming/configuration pitfall

`SNN_paper_unet_sel_notempdecay_fold0/experiment.yaml` declares temporal_weight_decay=false, but its inspected best checkpoint (7 completed epochs) has only one optimizer group, with weight_decay=3e-5, and no zero-decay group. Its name and YAML are therefore insufficient evidence that the intended exemption was active. This note does not treat that checkpoint as the verified no-decay comparison.

## A controlled comparison

Run two fresh experiments with identical architecture, initialization, seed, data split, batch/patch sizes, k, FPTT setting, number of updates, learning-rate schedule, validation, and checkpoint-selection rule. Change only temporal_weight_decay. Verify the optimizer groups at startup and save that audit in each run. Replicate with several seeds and preferably more folds.

Measure per-channel raw parameters, lambda and tau throughout training, task-gradient and decay magnitudes, full-fold raw and postprocessed Dice, and sensitivity to removing distant slice context. A frozen-temporal-parameter reference would help distinguish preserving the initialization from learning useful timescales. Any such context intervention must keep preprocessing and evaluation consistent.

## Sources

- Local implementation: `snn_nnunet/recurrent_neuron.py`, `snn_nnunet/recurrent_config.py`, `snn_nnunet/trainer.py`.
- Saved checkpoints: `checkpoint_epoch_0050.pth` in each compared run; `checkpoint_epoch_0100.pth` for the enabled run. All under `/gpfs/scratch1/shared/apiaghiardelli/<run>/Dataset001_BraTS24GLI/nnUNetTrainerSNNFPTT__SNNPlans__3d_fullres/fold_0/`.
- Saved `experiment.yaml` and `validation_periodic/dice_history.csv` from those runs.
- [PyTorch SGD documentation](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD): coupled weight decay, momentum, and Nesterov update.
