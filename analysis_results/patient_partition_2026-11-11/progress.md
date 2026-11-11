# SDD ledger — plan: docs/superpowers/plans/2026-11-11-patient-partition.md
Task 1: complete — RED missing module, GREEN 15 passed.
Pre-flight: Task 2 consumes Task 1 base-ID grouping; nested splits must keep the standard nnU-Net val key for internal validation and store test separately.
Ruling: implement in the current baseline_UNET working tree and preserve unrelated existing changes — the user requested changes to this workspace; new run metadata isolates data splits without switching or resetting the active branch.
Ruling: no commits or job submissions — neither requested by the user; source backups allow feature-only review of already dirty files.
Protocol clarification approved: retain patch validation but split outer development/test patient-wise; internal validation uses 10% of the development patients, as recommended immediately before approval.
Task 2: complete — nested folds, private metadata, launcher, resume and explicit val/test evaluator; targeted tests RED→GREEN, full suite 104 passed.
Ruling: place persistence/lifecycle helpers in partition_runtime.py — separates pure group algorithms from filesystem and CLI operations; no behavioral cost.
Ruling: new SNN runs use custom final validation directly, skipping the duplicate native validation pass — shared gt_segmentations is empty; the existing custom evaluator reads available preprocessed labels. Periodic validation and its metric grid are preserved.
Task 3: generated all 1,350 cases/613 base IDs; activated new run in unet_config.yaml and generated five matching YAML snapshots; added separate train/test sbatch jobs and PATIENT_PARTITION.md. Full test suite 104 passed; real trainer preflight and independent review pending.
Final review: 2 Important findings fixed in one RED→GREEN pass; no Critical or Minor findings.
Final: fixed partial dataset acceptance — test_partition_generation_rejects_a_wholly_missing_case RED→GREEN.
Final: fixed stale final checkpoint during extension — test_test_command_requires_final_checkpoint_to_complete_current_training_target RED→GREEN (100/200 rejected,200/200 accepted).
Final: scope rulings — GPU job execution not requested; grouping uses approved base ID with clinical mapping caveat; dataset-wide preprocessing reused and documented; unrelated existing changes preserved.
Task 3: complete — 107 tests passed; both real trainer classes verified on all5folds; shared metadata SHA256 unchanged; model/training settings preserved; Python and shell syntax checks passed.
