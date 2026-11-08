# Fall-Specific Adaptation of a Frozen Skeleton Action Model

Code and reproducibility inputs for the revised fall-detection study. A frozen
DSTE encodes skeleton sequences; a residual adapter and four-class head provide
fall-specific predictions. Raw denotes the independently trained raw-feature head
without an adapter, No-FU denotes adaptation on SAFER only, and Full G0 denotes
the proposed SAFER+FU adapter followed by the refit classifier.

## Reproduce the reported results

Use Python 3.10. The following CPU commands regenerate the repeated-run metrics,
ablation contrasts, ROC/confusion figures and conditional paired bootstrap intervals
from the released predictions. They check the results against fixed reference values.

```bash
python -m pip install -r requirements-revision.txt
python revision_20261108/reproduce.py
python revision_20261108/bootstrap.py
```

Outputs go to `revision_20261108/generated/`. The fixed predictions, labels, split
lists, configurations and checksums are reproduction inputs, so they remain in the
repository. See [the reproduction guide](revision_20261108/README.md) for details.

## Run the trained fall heads

Install PyTorch separately for your hardware. The revision used PyTorch 2.7.1,
torchvision 0.22.1 and CUDA 12.8. Obtain `revision_trained_heads.zip` from the
[fixed release](https://github.com/lab-sehyeon/fall_detect_sehyeon/releases/tag/revision-20261109-r2)
and extract its `weights/` directory. The archive contains 15 selected state files
for seeds 0–4; [weight_manifest.json](revision_20261108/weight_manifest.json)
records their checksums.

```bash
python revision_20261108/infer_features.py \
  --features pooled_features.npy --weights weights \
  --condition full_g0 --seed 0 --output logits.npy
```

The input is float32 `[N, 2048]` pooled DSTE features. Supported configurations are
`raw`, `no_fu_j1`, `no_fu_g0`, `full_j1`, and `full_g0`. This command applies the
fixed heads; it does not generate poses from RGB.

## Train from source feature caches

The [reproduction guide](revision_20261108/README.md#feature-level-retraining)
specifies source array shapes, ordering, label meanings, and the command for
`train_features.py`. The training core, selected settings and split lists are
retained. No test labels are used for fitting or checkpoint selection.

## Source preprocessing and comparison implementations

| Path | Purpose |
| --- | --- |
| `model/`, `tools.py` | DSTE architecture and shared utilities |
| `fall_pipeline/revision/`, `fall_pipeline/joint/` | Revision training and adapter/head components |
| `data_gen/`, `fall_pipeline/safer/`, `fall_pipeline/fu/` | Skeleton conversion, source preprocessing and feature helpers |
| `fall_pipeline/external/` | RGB frontend, lifting and external inference components |
| `scripts/`, `configs/` | Retained preparation/evaluation entry points, comparator implementations and their dependencies |
| `revision_20261108/` | Verified feature-level entry points and quantitative reproduction inputs |
| `tests/` | Existing checks for retained components |

The retained RGB/comparator scripts use separately obtained datasets, third-party
checkouts and weights, and historical manifests. They are not a newly validated
one-command reconstruction of every original pipeline. The verified release scope
is fixed-feature inference and quantitative regeneration; raw RGB preprocessing
and all comparator training have not been independently reconstructed end to end.
The older full-pipeline requirements are preserved for those components and are
not the RTX 5080 installation instructions.

## Evaluation scope

Five-run video F1 was 89.13 ± 1.70% on CAUCAFall and 91.50 ± 0.88% on Le2i for
Full G0, versus 86.15 ± 2.42% and 91.30 ± 0.94% for Raw. Values are mean ± sample
SD, not uncertainty intervals for new populations. Full−Raw mean differences
were +2.98 and +0.20 percentage points; both participant-cluster 95% intervals
include zero. These are offline, selected-person-track results with failed inputs
retained as no-alert outcomes. The source/transfer trade-offs and simpler variants
are included in the released results.

## Assets and attribution

Obtain original videos, pose data, the pretrained DSTE and frontend weights from
their providers under the corresponding terms. They are not redistributed here.
The release contains author-trained adapter/head weights only. The earlier fixed
release remains available for provenance; normal Git history is preserved.

This project builds on [FoundSkelModel](https://github.com/wengwanjiang/FoundSkelModel).
Keep its [LICENSE](LICENSE), the [third-party notices](THIRD_PARTY_NOTICES.md), and
component-specific notices when redistributing code. Dataset and model assets have
their own terms. This is the public author repository, not an anonymous repository.
