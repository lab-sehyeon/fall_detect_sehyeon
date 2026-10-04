# FU ZS0/ZS1 — SAFER 시간 모델의 무학습 전이

- 문서 ID: `DOC-20260920-fu-zs-reconstruction-R2`
- 기준일: 2026-09-20
- 연구 상태: `completed` — 전체 평가·독립 검산 완료

## 연구 질문과 방법

SAFER에 맞춰 학습한 temporal adapter가 FU-Kinect의 낙상과 intentional lying을 무학습으로
구분할 수 있는지, 전체 클립 시간 압축이나 FPS 차이가 전이에 영향을 주는지 확인한다.

Frozen DSTE/ADL 및 별도 재구현 F1 TS/sqrt epoch17을 그대로 사용한다. ZS0는 whole-clip64,
ZS1은 native30 또는 deterministic aligned25에서 contiguous64/stride8을 사용한다. 짧은 입력은
마지막 pose로 padding하되 해당 output은 판정에서 제외한다. 중첩 raw logits를 평균한 후 softmax한다.

미기재 조건은 사전에 별도 정의했다. End-anchor로 실제 전체 구간을 덮으며, aligned25 길이는
floor((T−1)×5/6)+1, 원본 index는 floor(6j/5+0.5)다. ZS0는 bilinear resize를 사용한다.
Clip prediction은 실제 frame 중 하나 이상 four-class argmax가 fall인 경우, ranking score는
max p(fall)이다. 임계값 fitting이나 후보 선택은 하지 않는다.

## 평가와 한계

FU993 clip labels로 binary F1·AUPRC/AUROC와 falling-vs-lying subset AUPRC, lying FP를 계산한다.
Clip label을 frame ground truth로 복제하지 않는다. 모델 불변성, sampling/coverage와 별도
집계의 독립 검산을 완료하기 전에는 수치를 확정하지 않는다. 원본 F1·전이 평가의 bit-exact 복원이
아니며, 새 결과를 역사적 수치와 동일 실험의 재현으로 주장하지 않는다.

## 검증된 결과

| 조건 | Clip F1 | 전체 AUPRC | Fall-vs-lying AUPRC | Lying FP/168 |
| --- | ---: | ---: | ---: | ---: |
| ZS0 resize64 | 23.705% | 13.759% | 42.277% | 128 |
| ZS1 native30 | 20.000% | 12.850% | 37.327% | 150 |
| ZS1 aligned25 | 20.449% | 12.878% | 37.516% | 138 |

각 조건에서993개 전체를 평가했고 입력·모델 불변성, 중첩 평균·clip 판정·지표의 독립 검산을
통과했다. 세 조건 모두 오탐이 많아 현재 무학습 전이 성능은 낮다. 역사적 all-non-fall 결과와
다른 양상이므로 같은 실패 기전을 재현했다고 주장하지 않는다. 결과를 보고 threshold나 시간
처리를 변경하지 않았다. 다음은 원본 특징·adapter hidden·concat의 동일 조건 probe 비교다.

[선행 F1](2026-09-20_f1_reconstruction_contract_shared.md) ·
[후속 전체 재구현 범위](2026-09-20_post_f1_execution_contracts_shared.md)
