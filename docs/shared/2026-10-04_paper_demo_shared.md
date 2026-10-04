# Paper-ready qualitative figures for fall recognition

문서 ID: DOC-20261004-paper-demo-R1

Date: 2026-10-04. Status: `completed` for figure preparation and validation. Venue-specific formatting and image-publication checks remain pending.

Three main figures and twenty supplementary cases illustrate the existing Le2i and CAUCAFall evaluations. All figures use actual video frames and saved model outputs. No predictions, training settings or decision rules were changed. The proposed method is labeled **Ours** throughout the figures.

Figure gallery (공개 사본 미포함) · Combined 23-page PDF (공개 사본 미포함) · Complete figure package (공개 사본 미포함) · English captions (공개 사본 미포함) · LaTeX insertion snippets (공개 사본 미포함)

## Main-text organization

| Figure | Content | Suggested placement |
| --- | --- | --- |
| 1. Inference pathway (공개 사본 미포함) | Actual RGB frame, detected person and 2D pose, estimated normalized skeleton, model modules and four-class output | Methods |
| 2. Le2i comparison (공개 사본 미포함) | One Ours TP and one Ours FN, with all five models evaluated on the same two videos | Qualitative results |
| 3. CAUCAFall comparison (공개 사본 미포함) | One Ours TP and one Ours FN, with all five models evaluated on the same two videos | Qualitative results |

Figure 1 displays a representative endpoint frame, but the proposed classifier uses the full 64-frame skeleton window. Its output is selected by four-class argmax, not a universal fall-probability threshold. “Posture” denotes the lying posture class; “Transition” denotes the lie-down transition class. Estimated, normalized skeleton coordinates are not metric world coordinates.

Figures 2 and 3 each use four shared RGB frames per video and aligned ground truth. The comparison includes Ours, USDRL with an NTU60 classifier, HFD, Privacy X3D-UDA using RGB input, and FLASH. HFD and FLASH are marked with an asterisk to identify project-retrained configurations. USDRL uses the official backbone with a project-trained NTU60 head, not an author-provided fall classifier. The [external comparison summary](2026-10-04_external_comparison_summary_shared.md) provides the broader experimental context.

## How to interpret the comparisons

The black GT bar marks the annotated fall interval. Vertical ticks mark actual fall-positive outputs at stored timestamps, and open triangles mark alarm onsets. Output frequency differs between window-based, clip-based and frame-based methods; tick density is not a comparable confidence measure. Empty processed rows contain no fall-positive outputs. X3D provides a video-level result only, so no temporal trajectory or alarm onset is invented for it.

The **Video** column reports video-level TP or FN. **E-FP** counts unmatched event alarms under the existing event-matching interval, from 0.5 seconds before GT onset to 3 seconds after GT end. A video TP does not imply correct event timing. For example, HFD detects the CAUCAFall `FallBackwardsS1` video as positive but misses its annotated event under this timing rule. FLASH's Le2i success example includes an additional unmatched alarm. These are offline evaluation timestamps, not measurements of online processing latency.

The four main comparison videos were selected post hoc from the previously fixed Ours success/failure cases. Every model is shown on these same videos, but selection remains conditional on Ours. The examples are illustrative, not a random or representative sample; they cannot establish overall superiority. Cases missed by Ours but detected by other models are retained.

## Supplementary material

The twenty supplementary figures retain the earlier five-model × two-dataset × two-outcome selection. Each contains three real frames and model-specific saved scores or a video-only decision. These cases were selected independently by model, so different supplementary pages are not necessarily paired comparisons of the same video. Their success/failure balance is not an accuracy estimate.

Two FLASH exceptions are explicitly identified. The Le2i failure example is a non-fall false positive because that evaluation contains no fall-video false negative for FLASH. The CAUCAFall failure is a pose-extraction failure across all 122 frames: the classifier did not run, and no negative score is fabricated. SVM margins and FLASH logits are not described as calibrated probabilities; connecting lines are visual aids, not additional predictions. No feature summary is presented as attention or causal explanation.

## Manuscript insertion and limitations

Main figures are designed for approximately 182 mm two-column width; supplementary figures use 88.9 mm single-column width. Labels are at least 8.5 pt at those sizes. Use the individual PDFs for manuscript insertion: text and lines remain vector objects with embedded fonts. SVGs are provided for editing, and PNGs are exported at 600 dpi. Export resolution does not increase the information present in the original RGB frames.

English captions and LaTeX figure snippets are included. The snippets are not a complete manuscript; adapt their paths and placement to the journal or conference template. Final figure numbering, column widths and caption length should follow that venue's requirements.

The figures and outcomes were checked against their underlying frames and evaluation outputs. They explain existing results rather than replacing full quantitative evaluation. RGB examples are not de-identified, and dataset availability does not by itself establish permission to reproduce identifiable images in a paper. Confirm the applicable image-publication requirements before submission or public release.
