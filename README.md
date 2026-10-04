# Skeleton-Based Fall Detection

A research implementation of fall detection from RGB video using pretrained 3D
skeleton representations. The project builds on
[FoundSkelModel](https://github.com/wengwanjiang/FoundSkelModel) and adds a residual
feature adapter and a fall classifier while keeping the skeleton encoder frozen.

The repository includes training and evaluation code, experiment configurations,
and research reports, including comparisons on Le2i and CAUCAFall.

## Method

The pipeline preserves the temporal order of video frames through pose estimation,
3D reconstruction, and skeleton feature extraction.

```text
RGB video
  -> Person detection and 2D pose estimation
  -> Pose quality screening
  -> 3D skeleton reconstruction
  -> Temporal windows
  -> Frozen pretrained skeleton encoder
  -> Residual feature adapter
  -> Four-class classifier
  -> Fall event decoding
```

- **Video processing:** YOLOv8 detects people, ViTPose estimates 2D joints, and
  MotionAGFormer reconstructs 3D skeleton sequences.
- **Representation learning:** the pretrained DSTE encoder from FoundSkelModel
  provides fixed skeleton features and is excluded from fall-training optimization.
- **Feature adaptation:** a residual adapter is trained with SAFER-Activities and
  FU-Kinect-Fall, using dataset-specific supervision.
- **Fall classification:** the final classifier is trained on SAFER-Activities to
  distinguish other activity, falling, the motion of lying down, and a lying posture.
- **Event decoding:** consecutive fall predictions are converted into fall alarms
  for evaluation.

The evaluated pipeline uses whole-video quality checks and future temporal context
in 3D reconstruction. The reported results are offline evaluations; real-time
latency and deployment performance have not been established.

## External Evaluation

The following results come from local evaluations on the same **130 Le2i videos**
and **100 CAUCAFall videos** for each model. All values are **video-level percentages**,
with fall as the positive class. Processing failures remain in the evaluation
population and are treated as producing no alarm.

| Method | Le2i Precision | Le2i Recall | Le2i F1 | CAUCAFall Precision | CAUCAFall Recall | CAUCAFall F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Ours** | **95.56** | **86.87** | **91.01** | **97.50** | **78.00** | **86.67** |
| Pretrained encoder + NTU60 classifier | 98.63 | 72.73 | 83.72 | 90.91 | 40.00 | 55.56 |
| HFD 3D-CNN+SVM (reproduction) | 86.21 | 25.25 | 39.06 | 51.61 | 96.00 | 67.13 |
| Privacy X3D-UDA (RGB inference) | 98.21 | 55.56 | 70.97 | 60.26 | 94.00 | 73.44 |
| FLASH (reproduction) | 76.15 | 100.00 | 86.46 | 49.49 | 98.00 | 65.77 |

These are measurements of the evaluated implementations, rather than performance
numbers copied from their papers. Training sources and input processing differ:

- The encoder baseline uses the author's pretrained encoder and a 60-class linear
  classifier trained in this project on NTU60. Its falling action class supplies
  the fall prediction.
- HFD uses the author's C3D feature extractor and a newly trained SVM on GMDCSA.
  FLASH uses the official architecture retrained on the available UP-Fall sequences.
- Privacy X3D-UDA uses an author-provided checkpoint trained with RGB and depth,
  evaluated here with RGB input.

No target-domain fitting or threshold tuning was performed in these evaluation
runs. The target datasets had been inspected in earlier analyses, and overlap with
all upstream pretraining data has not been fully audited. These comparisons do not
isolate the effect of the adapter alone.

See the [evaluation report](docs/shared/2026-10-04_external_comparison_summary_shared.md)
and [results CSV](docs/shared/2026-10-04_external_comparison_summary_selected_results.csv)
for training provenance, processing coverage, confusion counts, and event-level
results where available.

## Repository Layout

| Path | Contents |
| --- | --- |
| `model/` | Pretrained skeleton model architecture |
| `fall_pipeline/` | Feature adaptation, classification, and evaluation modules |
| `data_gen/` | Dataset preparation and skeleton preprocessing |
| `configs/` | Model and experiment configurations |
| `scripts/` | Data preparation, training, evaluation, and reporting tools |
| `tests/` | Tests for model components, preprocessing, and evaluation logic |
| `docs/shared/` | Research methods, evaluation reports, and result tables |

## Reproducibility and Assets

This source repository provides code, configurations, tests, and research reports.
Dataset files, model checkpoints, third-party repository checkouts, private experiment
records, and illustrations extracted from dataset videos are excluded from the
publication package. Obtain external assets from their original distributors under
the applicable terms.

Some experiment tools require separately prepared manifests and execution records.
The repository alone is therefore insufficient to rerun every historical experiment
without preparing those inputs. Dependency specifications are recorded in
[requirements-fall-detect.txt](requirements-fall-detect.txt) and
[requirements-recovered.txt](requirements-recovered.txt).

The [active configuration](configs/active_fall_model.json) identifies the current
model components and evaluation entry point.

## Documentation

- [Methods](docs/shared/2026-09-29_paper_methods_main_shared.md)
- [Research rationale and evidence](docs/shared/2026-10-04_research_strengths_paper_rationale_shared.md)
- [External comparison results](docs/shared/2026-10-04_external_comparison_summary_shared.md)
- [Research documentation index](docs/shared/README.md)
- [Contribution guide](CONTRIBUTING.md)

Detailed research reports are currently primarily written in Korean.

## Acknowledgments

This project builds on the official
[FoundSkelModel implementation](https://github.com/wengwanjiang/FoundSkelModel) and
[Foundation Model for Skeleton-Based Human Action Understanding](https://arxiv.org/abs/2508.12586).
The RGB processing pipeline also uses YOLOv8, ViTPose, and MotionAGFormer.
Their contributions remain attributed to the original authors.

## License

The upstream FoundSkelModel code is distributed under **Apache-2.0**, and its
[original license](LICENSE) is preserved. See
[Third-Party Notices](THIRD_PARTY_NOTICES.md) for component-specific licenses,
attributions, and redistribution considerations.

Ultralytics YOLOv8 is subject to **AGPL-3.0 or a separate Enterprise agreement**.
The applicable obligations for a combined distribution require consideration of
its scope; the root Apache license does not provide Apache-only permission for the
entire pipeline. See the [Ultralytics licensing guidance](https://www.ultralytics.com/license).

Model weights, datasets, source videos, and extracted images have separate terms.
Preserve the applicable copyright notices and license texts when redistributing
third-party material.
