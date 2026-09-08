# hippo_segs

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
