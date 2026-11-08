# Fall detection revision: quantitative reproduction

This directory contains the five-seed revision results and CPU scripts for independently recomputing the reported metrics. It supplements the existing implementation; the scope of each command is explicit below.

## Recompute tables and plots

From the repository root, with Python 3.10, NumPy, pandas, scikit-learn and matplotlib installed:

```bash
python revision_20261108/reproduce.py
python revision_20261108/bootstrap.py
```

The first command regenerates per-run precision/recall/F1/AUC, mean and sample SD, paired ablation differences, four-class metrics, ROC curves and confusion matrices. The second reproduces the 20,000-draw participant-cluster and class-stratified video confidence intervals, conditional on the five already fitted model pairs. Outputs are written to `generated/`. The scripts check results against fixed expected values. `manuscript_tables/` contains the separately audited manuscript tables used to cross-check the export.

`results/external_predictions.csv` contains 6,670 rows: five configurations × five seeds × 230 videos, plus four single-run comparators × 230 videos. Rejected DSTE inputs remain no-alert. Missing ROC scores use a shared score below every valid score for that method and dataset. Repeated seeds do not create additional independent videos. Source confusion matrices cover 50 configuration/seed/split combinations.

## Fixed model inference on pooled features

The separately supplied `revision_trained_heads.zip` contains 15 state-dictionary files under `weights/seed_0` through `weights/seed_4`. `weight_manifest.json` provides SHA256 checksums. The archive includes author-trained Raw heads and Full/No-FU adapters and J1/G0 heads; it does not contain a DSTE backbone or third-party frontend weights.

```bash
python revision_20261108/infer_features.py \
  --features pooled_features.npy --weights weights \
  --condition full_g0 --seed 0 --output logits.npy
```

Input is float32 `[N, 2048]` pooled DSTE features. This command is CPU inference on fixed features, not RGB decoding or pose estimation. It supports all five evaluated configurations. Exported tensors were checked against the selected training checkpoints; all 25 seed/configuration routes were compared with stored predictions on real external feature windows within the recorded CPU/GPU numerical tolerance.

## Feature-level retraining

`train_features.py` calls the same training core used for the revision. It requires the original split ordering and corrected preprocessing: no feature scaling, SAFER reference-frame window labels, and FU mean-window clip features. `split_lists/` records SAFER sequence membership, all 993 FU clips and external video membership. `configs/revision_training.json` fixes the optimizer, losses, batching, schedules and the 21/12/50 epoch budgets. Full and No-FU share SAFER sample ordering and dropout RNG consumption; No-FU never accesses FU examples.

Prepare `train_features.npy` (609183×2048), `train_labels.npy`, `val_features.npy` (106680×2048), and `val_labels.npy` in a cache directory. Full additionally requires `fu_features.npy` (993×2048) and binary `fu_labels.npy` in the order of `fu_clips.csv`.

```bash
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python revision_20261108/train_features.py \
  --cache cache --condition full --seed 0 --output runs/seed_0/full
```

Repeat for seeds 0–4 and conditions `raw`, `no_fu`, `full`. This feature-level command does not download datasets or regenerate the multi-gigabyte corrected features. It hashes supplied inputs and selects G0 using source validation only. Test/OOD and external labels are never opened during fitting. The revision used PyTorch 2.7.1+cu128 on RTX 5080, FP32 with AMP/TF32 disabled and deterministic operations. Exact results across different devices or software versions are not guaranteed. The original full-pipeline requirements are included for historical context; their older torch pins are not the RTX 5080 environment.

## Additional evidence

- `calibration_*`: source-validation threshold selection and the fixed external NTU60 comparison.
- `rgb_trials.csv` / `rgb_summary.json`: four fixed videos × three fresh-process RGB trials, including startup and intermediate I/O in cold wall time. CPU RSS and PyTorch GPU allocation have different accounting boundaries.
- `le2i_collections.csv`: annotation availability in the selected release, with all 130 annotated videos retained.
- `person_snapshot_review.*`: preliminary AI-assisted visual screening at five fixed frames per video. This is not human exhaustive video annotation and does not establish absence of additional people between snapshots.

## Reproduction boundaries

The numeric tables and feature-level code can be examined without downloading videos. Raw dataset videos, pose arrays and third-party weights are not redistributed here; obtain them from their original providers under the applicable terms. The original frontend and comparator implementations remain in the repository. This addition does not claim a freshly validated, one-command reconstruction of all RGB inputs and all comparator training. FLASH source split/pose provenance and the video-level conversion of its impact task remain limitations described in the manuscript. The included outputs enable auditing the reported comparison without silently substituting a new reproduction.

The repository license applies to the added code. Dataset and third-party model licenses remain with their respective providers. This author repository is public and is not an anonymous review repository.
