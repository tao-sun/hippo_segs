# hippo_segs

This repository trains the existing sequential SNN/SpikMamba models with nn-U-Net v2.8.1. nnU-Net handles dataset planning and preprocessing, sampling, augmentation, region loss, cross-validation, checkpointing, full-volume validation, inference, and postprocessing. The project supplies an external trainer and a thin CLI.

## Set up

Use Python 3.10 and [uv](https://docs.astral.sh/uv/) from the repository root. The project pins `nnunetv2==2.8.1` and PyTorch 2.5.1; its Mamba wheel targets Python 3.10, CUDA 12, and PyTorch 2.5. Production model execution needs a compatible NVIDIA GPU environment.

~~~bash
uv sync --python 3.10

export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export SNN_RESULTS_BASE=/path/to/nnUNet_results
export nnUNet_results="$SNN_RESULTS_BASE"
export nnUNet_extTrainer="$PWD/snn_nnunet"
export nnUNet_compile=false
~~~

Set the three nnU-Net data paths to writable, persistent directories. The project CLI sets `nnUNet_extTrainer` and defaults `nnUNet_compile=false` itself. Keep these exports when invoking native nnU-Net commands directly: the installed package discovers the external `nnUNetTrainerSNNFPTT` through `nnUNet_extTrainer`. Compilation defaults to false because the SNN has mutable neuron state and custom Mamba operations.

Run the following commands from the repository root in a shell with these exports. Dataset ID `501` and name `BraTS24GLI` are examples; choose an unused ID and use it consistently.

## Prepare extracted BraTS24 data

Extract the BraTS24 GLI archive first. `--dataset-root` must point to an extracted directory, not a ZIP file. The converter finds subject directories recursively. Each case needs one each of `*t1n*.nii.gz`, `*t1c*.nii.gz`, `*t2w*.nii.gz`, and `*t2f*.nii.gz`, plus one `*seg*.nii.gz` for a labeled training case.

~~~bash
export BRATS24_EXTRACTED=/path/to/extracted/BraTS24-GLI
uv run --python 3.10 python -m snn_nnunet.cli prepare \
  --dataset-root "$BRATS24_EXTRACTED" \
  --dataset-id 501 --dataset-name BraTS24GLI
~~~

The converter writes `Dataset501_BraTS24GLI` under `nnUNet_raw`. Channels `_0000` through `_0003` are T1, T1ce, T2, and FLAIR. Only segmentation value `4` is changed to background `0`; image intensities and geometry are preserved. Unlabeled cases go to `imagesTs`. An already formatted `Dataset501_BraTS24GLI` raw directory can also be passed as `--dataset-root`.

Preparation runs native nnU-Net fingerprinting, planning, and preprocessing with dataset integrity checks. It derives `SNNPlans.json` from stock plans, retaining the stock preprocessed data identifier. The SNN `3d_fullres` plan uses a **[128, 128, 128] voxel patch** and **batch size 4**. Its `model_kwargs.patch_size: 4` is the internal 2D SpikMamba patch-embedding factor, not the nnU-Net spatial patch size.

## Train and resume

To select between the stock 3D U-Net and the SNN without changing either
trainer, edit `brats24_fold0_model.yaml`. Set `family` to `unet3d` or
`snn`, and give each experiment a fresh `run_dir`. The launcher reads the
appropriate plans (`nnUNetPlans` or `SNNPlans`) and keeps checkpoints in that
run directory. The optional `snn` section is used only when `family: snn`.
Run it from the repository root after exporting the existing data paths:

~~~bash
export nnUNet_raw=/home/aurora/nnUNet_raw
export nnUNet_preprocessed=/home/aurora/nnUNet_preprocessed
uv run --python 3.10 python -m snn_nnunet.run_model \
  --config brats24_fold0_model.yaml
~~~

The launcher refuses a nonempty `run_dir` and saves a copy of the YAML as
`experiment.yaml` inside the new run. Use a different run directory when
switching model family. For `unet3d` it calls nnU-Net's standard `nnUNetTrainer`
with `3d_fullres`; for `snn` it calls this repository's existing training CLI.

Use a distinct `nnUNet_results` root for each experiment. Keep its path: resume, validation, and prediction must point to that same root. The native result folders and checkpoints live beneath it. `full-cv` trains folds 0 through 4 sequentially with the same configuration:

~~~bash
export SNN_RUN="$SNN_RESULTS_BASE/brats24_orig_axis0_k16_fptt"
export nnUNet_results="$SNN_RUN"
uv run --python 3.10 python -m snn_nnunet.cli full-cv \
  --dataset-id 501
~~~

To train one fold, point `nnUNet_results` at a different fresh root:

~~~bash
export SNN_SINGLE_RUN="$SNN_RESULTS_BASE/brats24_fold0"
export nnUNet_results="$SNN_SINGLE_RUN"
uv run --python 3.10 python -m snn_nnunet.cli train \
  --dataset-id 501 --fold 0
~~~

Defaults are model `orig`, temporal axis `0`, `k=16`, and FPTT enabled. Select another model (`orig`, `shallow`, `medium`, or `deep`), axis (`0`, `1`, or `2`), window length, or plain TBPTT with `--model`, `--temporal-axis`, `--k`, or `--no-fptt`. These choices must match for every fold in a run.

Resume with `--continue` after restoring the original `nnUNet_results` root. Repeat every nondefault configuration flag: omitted flags revert to defaults and can conflict with the saved plans. For example, a run originally started with `--model shallow --temporal-axis 2 --k 8 --no-fptt` resumes as follows:

~~~bash
export nnUNet_results=/path/to/existing/shallow_axis2_k8_results
uv run --python 3.10 python -m snn_nnunet.cli full-cv \
  --dataset-id 501 \
  --model shallow --temporal-axis 2 --k 8 --no-fptt --continue
~~~

For the default example run:

~~~bash
export nnUNet_results="$SNN_RUN"
uv run --python 3.10 python -m snn_nnunet.cli full-cv \
  --dataset-id 501 --continue
~~~

The CLI delegates checkpoint continuation to native nnU-Net. A fresh results root keeps experiments separate; reuse the same root for a continuation or for reading its checkpoints.

For multiple GPUs, pass `--gpus` to training; this example starts a separate four-GPU results root:

~~~bash
export nnUNet_results="$SNN_RESULTS_BASE/brats24_4gpu"
uv run --python 3.10 python -m snn_nnunet.cli full-cv \
  --dataset-id 501 --gpus 4
~~~

Each fold uses native nnU-Net distributed training, while `full-cv` runs the folds one after another.

## Validate and predict

After training, native full-volume validation uses the final checkpoint by default; `--best` selects `checkpoint_best.pth`:

~~~bash
export nnUNet_results="$SNN_RUN"
uv run --python 3.10 python -m snn_nnunet.cli validate \
  --dataset-id 501 --fold 0

uv run --python 3.10 python -m snn_nnunet.cli validate \
  --dataset-id 501 --fold 0 --best
~~~

The training log's native pseudo-Dice is computed on sampled validation patches. For fold-level Dice reporting, use the true full-volume native validation output.

Prediction input is a folder of raw nnU-Net channels named like `case_0000.nii.gz` through `case_0003.nii.gz`. For example, use the prepared dataset's `imagesTs` if it contains unlabeled cases. One-fold prediction:

~~~bash
export nnUNet_results="$SNN_RUN"
uv run --python 3.10 python -m snn_nnunet.cli predict \
  --dataset-id 501 \
  --input "$nnUNet_raw/Dataset501_BraTS24GLI/imagesTs" \
  --output /path/to/predictions_fold0 --folds 0
~~~

Once all five folds have checkpoints, omit `--folds` to use the native five-fold prediction ensemble. `--checkpoint best` selects best instead of final checkpoints:

~~~bash
export nnUNet_results="$SNN_RUN"
uv run --python 3.10 python -m snn_nnunet.cli predict \
  --dataset-id 501 \
  --input "$nnUNet_raw/Dataset501_BraTS24GLI/imagesTs" \
  --output /path/to/predictions_5fold
~~~

Optional postprocessing uses nnU-Net's selected connected-component rules. To request it, start a full five-fold run with `--select-best` so native configuration and postprocessing selection runs after training, then predict with the same results root and `--postprocess`:

~~~bash
export SNN_POST_RUN="$SNN_RESULTS_BASE/brats24_with_selection"
export nnUNet_results="$SNN_POST_RUN"
uv run --python 3.10 python -m snn_nnunet.cli full-cv \
  --dataset-id 501 --select-best

uv run --python 3.10 python -m snn_nnunet.cli predict \
  --dataset-id 501 \
  --input "$nnUNet_raw/Dataset501_BraTS24GLI/imagesTs" \
  --output /path/to/predictions_5fold_selected --postprocess
~~~

Postprocessed results appear in `/path/to/predictions_5fold_selected_postprocessed`. The selection needs completed cross-validation outputs and applies no project-specific thresholds or morphology.

## Sequential trainer behavior

The adapter accepts native `[B, 4, X, Y, Z]` patches and returns `[B, 3, X, Y, Z]` region logits. `--temporal-axis` chooses X, Y, or Z in **preprocessed nnU-Net coordinates**, which need not match original NIfTI orientation. Slices are traversed in order. Neuron state continues across windows within one patch and resets for the next independent patch or inference tile. Native augmentation and inference mirroring exclude the temporal axis: axes 0, 1, and 2 allow `(1, 2)`, `(0, 2)`, and `(0, 1)` respectively.

The trainer uses native region Dice/BCE loss. For each `k`-slice window, it takes one optimizer update and detaches neuron state: truncated backpropagation through time (TBPTT). FPTT adds its regularizer and updates auxiliary running parameters once per window; `--no-fptt` retains TBPTT without FPTT operations. A temporal length of 128 at `k=16` gives 8 updates per minibatch. At 250 minibatches per epoch this is **2,000 optimizer updates per epoch across 32,000 temporal positions per batch element** (128 × 250), or 600,000 updates over the configured 300 epochs. Batch size 4 therefore processes 128,000 individual slice instances per epoch. These are job-level update counts, not multiplied by GPU count; they describe the schedule, not measured production throughput.

## Troubleshooting

- If native nnU-Net reports missing paths, export `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results` in the current shell. Native commands also need `nnUNet_extTrainer` to find the custom trainer.
- If conversion rejects a ZIP or a case, extract the archive and check for exactly one file of each modality per subject. An existing raw destination with different contents is deliberately rejected.
- If resume or validation reports incompatible plans or results, restore the original `nnUNet_results` root and matching dataset ID, model, temporal axis, `k`, and FPTT mode. Repeat all nondefault training flags with `--continue`. The shared preprocessed `SNNPlans.json` must also match the selected run for validation.
- The nnU-Net 2.8.1 native Python training API expects the dataset argument as a **string**. This CLI converts it; direct API calls should use `str(dataset_id)` (for example, `"501"`).
- Native prediction needs checkpoints for each requested fold. Its five-fold default needs all five; use `--folds 0` for a completed single fold.

## Legacy workflow

The earlier 2D slice-cache experiment remains available through `data/preprocess_brats24.py`, `experiments_snn_fptt.yaml`, and `snn_fptt.py`. It uses separate preprocessing and Accelerate commands.
