# DNN 2D baseline

Standalone DNN baseline for 2D BraTS slice training and 3D evaluation.

- `dnn_3ch.py`: model, datasets, training, and volume evaluation.
- `dnn_evaluation.py`: per-view and ensemble evaluation.
- `dnn_unet.job`: SLURM training array over five folds.
- `snn_evaluation.job`: SLURM DNN evaluation, retaining its original filename.
- `run_dnn_eval_sag_fold1.sh`: local evaluation launcher for fold 1.

Run these commands from the repository root with the project environment active:

```bash
python dnn_2D_baseline/dnn_3ch.py --help
python dnn_2D_baseline/dnn_evaluation.py --help
bash dnn_2D_baseline/run_dnn_eval_sag_fold1.sh
```

Submit cluster jobs from the repository root:

```bash
sbatch dnn_2D_baseline/dnn_unet.job
sbatch dnn_2D_baseline/snn_evaluation.job
```

Data, checkpoint, output, and cluster settings retain their existing values.
Relative paths are resolved from the working directory where the command or
job is launched. Update the settings in the launchers for the intended run.
