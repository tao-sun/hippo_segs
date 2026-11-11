# Patient-wise BraTS24 training and testing

The active `unet_config.yaml` now uses a separate patient partition and a new run directory. Architecture, deep supervision, training hyperparameters, 50 patch-validation minibatches per epoch, and full-volume validation every 50 epochs are preserved.

## Protocol

Five outer folds, seed 42. In each outer fold, one fifth of base IDs is the test set. Of the remaining patient groups, 10% (rounded up) is reserved for internal validation, using seed `42 + fold`. All sessions of a base ID stay together. The approximately 72%/8%/20% shares refer to groups; scan counts vary.

`BraTS-GLI-00020-100` and `BraTS-GLI-00020-101` map to `BraTS-GLI-00020`. The dataset contains 1,350 case IDs and 613 such base IDs. This is grouping by the available identifier; an independent clinical patient-ID mapping was not supplied.

| Outer fold | Training: groups / scans | Validation: groups / scans | Test: groups / scans |
|---|---:|---:|---:|
| 0 | 441 / 966 | 49 / 107 | 123 / 277 |
| 1 | 441 / 970 | 49 / 111 | 123 / 269 |
| 2 | 441 / 975 | 49 / 108 | 123 / 267 |
| 3 | 441 / 957 | 50 / 113 | 122 / 280 |
| 4 | 441 / 989 | 50 / 104 | 122 / 257 |

Patch validation, periodic full-volume validation, `checkpoint_best.pth` selection (native EMA pseudo Dice), and `checkpoint_best_postprocessed.pth` selection use **internal validation only**. No test inference runs during training. The native file `splits_final.json` contains only `train` and internal `val`; the outer test lives in `partition.json` and `test_splits.json`.

## Start training

From the repository root, for fold 0:

```bash
sbatch train_patient_brats24.sbatch
```

For another outer fold, for example fold 1:

```bash
sbatch --export=ALL,FOLD=1 train_patient_brats24.sbatch
```

Repeat with FOLD=0,1,2,3,4 for the full experiment. Each run has its own results directory. The five files `configs/patient_wise/fold_N.yaml` are matching experiment snapshots; keep architecture and training settings consistent across folds when editing them. To choose another configuration, pass `CONFIG_PATH` using `--export=ALL,CONFIG_PATH=/absolute/path/config.yaml`.

The existing `sbatch train_validate_snn_brats24_fold0.sbatch` also uses the new patient partition through the updated `unet_config.yaml`.

## Evaluate the held-out test after training

The default checkpoint is `checkpoint_best.pth`, selected on internal validation. The command requires `checkpoint_final.pth` in the same fold directory and checks its completed epoch against the currently saved training target. A final checkpoint from an earlier, shorter training is rejected while an extension is incomplete.

```bash
sbatch test_patient_brats24.sbatch
sbatch --export=ALL,FOLD=1 test_patient_brats24.sbatch
```

Results are written to `<run_dir>/test_fold_N/summary.json` and `metrics.csv`, with raw and postprocessed Dice for WT, TC and ET.

Each test computes two separate averages for every region and for their equally weighted WT/TC/ET mean:

- **Mean Dice per volume**: average the Dice of all test volumes, giving every volume equal weight. JSON: `mean_per_volume`. The legacy `mean` key retains this same value.
- **Mean Dice per patient**: first average each patient's volume Dice, then average these patient means, giving every base ID equal weight regardless of its number of scans. JSON: `mean_per_patient`.

Both averages are calculated before (`raw`) and after (`postprocessed`) postprocessing. The JSON also contains `counts` (volumes/patients), a `patients` list with each patient's volume IDs and regional means, and `delta_per_patient` for the patient-weighted postprocessing gain. The existing `delta` remains the volume-weighted gain. Empty-empty Dice remains 1.

Additional test files are `aggregate_metrics.csv` (both weightings, raw/postprocessed, WT/TC/ET/mean) and `patient_metrics.csv` (one row per patient per processing stage, including volume count, case IDs and regional means). The job log explicitly prints **Mean Dice per volume** and **Mean Dice per patient**. These additions apply both to the final test and to full-volume internal validation for patient-partition runs, including periodic validation every 50 epochs. Checkpoint selection retains its existing volume-average criterion.
 The evaluator restores the run's saved split and validation patch automatically. Fixed postprocessing settings are unchanged: probability threshold 0.5, ET/TC/WT minimum component sizes 161/166/50 voxels, closing radius 1. Choose checkpoint and processing settings using internal validation before running the test; do not choose them by comparing test scores.

The test module can also be called directly:

```bash
.venv/bin/python -m snn_nnunet.custom_full_validation \
  --dataset-id 1 --fold 0 --configuration 3d_fullres \
  --trainer nnUNetTrainerSNNFPTT --plans SNNPlans \
  --results-root /gpfs/scratch1/shared/apiaghiardelli/SNN_2D_Unet_SMMLif_DS_patientwise_seed42/outer_fold_0 \
  --output-dir /gpfs/scratch1/shared/apiaghiardelli/SNN_2D_Unet_SMMLif_DS_patientwise_seed42/outer_fold_0/test_fold_0 \
  --checkpoint-name checkpoint_best.pth --subset test
```

The SNN jobs export the required nnU-Net paths. Direct module invocation requires `nnUNet_raw` (the existing dataset root); the saved run restores `nnUNet_preprocessed` and `nnUNet_results`.

## Two Dice averages during periodic validation

At epochs 50, 100, 150, etc. (or the configured validation interval), full-volume **internal validation** computes both volume-weighted and patient-weighted Dice, before and after postprocessing. It uses only the internal `val` cases; the outer test remains held out until the explicit final test command.

Each `validation_periodic/epoch_NNNN/` contains `summary.json`, `metrics.csv`, `aggregate_metrics.csv` and `patient_metrics.csv`. The combined `validation_periodic/dice_history.csv` keeps its original columns and adds patient counts plus explicit columns such as `raw_per_volume_mean`, `postprocessed_per_volume_mean`, `raw_per_patient_mean` and `postprocessed_per_patient_mean`, with corresponding WT/TC/ET columns.

When rebuilding the history on resume, older summaries from patient-partition runs are upgraded from their saved per-volume Dice without rerunning inference. Legacy runs without a patient partition retain their previous aggregation behavior. Online patch validation at every epoch continues to compute the native pseudo Dice; these patient metrics concern the full-volume evaluations.

## Data isolation and resume

At launch, the YAML option `data_partition` creates `<run_dir>/nnUNet_preprocessed/Dataset001_BraTS24GLI`: arrays are symlinked, while dataset metadata, plans and splits are private copies. `<run_dir>/data_partition.json` stores the complete partition. Resume, CLI validation and custom evaluation restore that private root and verify split integrity. Old runs without this snapshot keep their previous behavior.

Shared preprocessing and its original split are preserved. Existing checkpoints must be resumed with their original split; a new patient-wise run starts from fresh model weights. No job was submitted as part of this implementation.

The current shared `gt_segmentations` directory contains no original NIfTI masks. For these new SNN runs, final validation uses the existing custom evaluator directly from preprocessed image/segmentation arrays, as periodic validation already does, avoiding a redundant native pass that requires those missing NIfTIs. Its Dice remains on the preprocessed volume grid, consistent with the existing custom pipeline. Raw preprocessing/planning has been reused; this change does not refit dataset preprocessing separately inside each outer fold.

The original active YAML is backed up at `analysis_results/patient_partition_2026-11-11/before/unet_config.yaml`.

## Reproduce and inspect the partition

- `data_partitions/brats24_patient_seed42/partition.json`: all three groups, seed, counts and provenance.
- `data_partitions/brats24_patient_seed42/splits_final.json`: internal training/validation lists consumed by nnU-Net.
- `data_partitions/brats24_patient_seed42/test_splits.json`: held-out outer cases.
- `data_partitions/brats24_patient_seed42/patient_assignments.csv`: per-base-ID membership across all five outer folds.

To generate a separate new partition directory:

```bash
.venv/bin/python -m snn_nnunet.partition_runtime \
  --source /gpfs/scratch1/shared/apiaghiardelli/nnUNet_preprocessed/Dataset001_BraTS24GLI \
  --output data_partitions/another_patient_partition \
  --seed 42 --validation-fraction 0.1
```

The generator refuses to overwrite existing partition directories. It checks the preprocessed inventory against `dataset.json.numTraining`, all case IDs and paired preprocessed arrays, disjoint groups in every split, complete case coverage and exactly one outer test appearance per case. Every model compared should use the same partition and test protocol. Report per-fold volumetric test Dice and the mean/standard deviation across the five folds; scan-wise and patient-wise averages are different when subjects have multiple scans.
