# SAFER F1 temporal adapter 재현 계약

- 문서 ID: `DOC-20260920-f1-reconstruction-contract-R7`
- 기준일: 2026-09-20
- 연구 상태: `completed` — 두 번째 revision 전체 학습·고정 평가·독립 검산 완료

## 연구 질문과 보존할 방법

고정된 DSTE의 dense 시간 표현 위에 residual temporal adapter를 학습해 window-level linear
control보다 frame-level 낙상·lying 구분과 시간적 localization을 개선할 수 있는지 검증한다.

역사적 문서에서 확인되는 방법은 temporal-only/temporal+spatial × plain CE/sqrt inverse-frequency
CE의 네 후보 비교다. 시간 표현은 64×1024이며, spatial valid-token 평균을 각 frame에 결합한
후보는 64×2048이다. Frame logits를 중첩 구간에서 평균한 다음 softmax를 적용한다. DSTE와 ADL
분기는 고정하며 validation frame macro-F1, fall-vs-unstable AUPRC, lying_down F1 순으로 선택한다.

## 재현 범위와 한계

시간층 상세와 optimizer·학습률·batch·총 epoch 등은 문서만으로 확정되지 않는다. 누락값을 보충하는
경우 별도 문서 기반 재구현 실험으로 사전에 명시하며 원본과 완전히 동일한 실행으로 표현하지 않는다.
기존 V1 입력 재구성의 동일성 한계도 이어진다.

역사적 선택은 temporal+spatial/sqrt, epoch15였지만 새 실행에서 같은 후보·epoch를 강제로 고르지 않는다.
Fall-vs-unstable과 fall-vs-lie 지표도 정의 확인 없이 혼용하지 않는다. Event 평가의 구간 추출·matching
규칙이 확정되기 전에는 새 event F1을 역사적 값과 동일 기준으로 비교하지 않는다.

## 선행 입력 검증

원본에 없는 세부는 별도 재구현 조건으로 확정했다. Hidden 512의 residual temporal block 한 개,
kernel3 convolution 두 개와 dropout 0.1을 사용한다. 네 표현/loss 후보를 seed0, 20epochs,
batch128, AdamW(lr 0.001, weight decay 0.0001), 일정 학습률로 비교한다. Sqrt weight는
중첩을 제거한 train covered-frame 빈도로만 계산하고 loss pair는 같은 초기화·sample order를 사용한다.
기존 epoch15 결과에 맞추어 학습을 중단하거나 후보를 고르지 않는다.

Fall-vs-unstable AUPRC는 coarse unstable/fall frame에서 4-class p(fall)로 계산한다. Event는
연속 argmax-fall segment의 IoU≥0.1 greedy one-to-one matching으로 별도 보고하며, 역사적 event
지표와 동일하다고 주장하거나 모델 선택에 사용하지 않는다. 이 세부는 원본에서 복구한 값이 아니라
사전 고정한 재구현 정의다.

Train/validation의 중첩 window label 일관성과 subject 분리를 확인했다. Train 4,891,177 frames 중
4,828,896개, validation 857,774 frames 중 843,768개가 clean windows로 덮인다. 미커버 frame은
각각 62,281개와 14,006개이며 timeline 평가의 coverage로 명시해야 한다. 중첩 window의 frame
개수와 unique frame 개수가 다르므로 loss의 class-frequency 산정 단위도 사전 고정할 필요가 있다.

표현·loss·모델·timeline·selection과 중단 후 재개의 일관성을 검증했다. 제한 실행은 전체 성능 결과로
해석하지 않는다. 전체 특징 추출→네 후보 학습→validation 선택 고정→test/OOD 단일 평가→독립
검산 순서의 실험을 완료했다. 네 후보 모두 20 epochs를 마쳤고 validation에서
temporal+spatial/sqrt, epoch17을 선택했다. 선택 후 변경 없이 test/OOD에 평가했다.

| 평가 | Windows | 평가 covered frames | Macro-F1 | Fall F1 | Fall-vs-unstable AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Test | 216,768 | 1,743,832 | 75.278% | 79.063% | 92.575% |
| OOD | 71,964 | 577,392 | 56.516% | 54.271% | 94.671% |

미커버 frame은 test 21,056개, OOD 126개이며 지표에서 제외했다. 모든 후보의 validation 선택,
고정 모델 불변성, 저장 예측·중첩 평균·지표 재계산의 독립 검산을 통과했다. OOD에서 성능 하락이
남아 있으며 전체 후속 파이프라인의 검증 완료를 뜻하지 않는다. F0B의 window-level 지표와 F1의
unique-frame timeline 지표는 평가 단위가 달라 차이를 직접적인 성능 개선량으로 해석하지 않는다.

[자동 갱신 실행 기록](2026-09-20_f1_run_shared.md)에 epoch별 예비 진행과 최종 검증 결과를 구분한다.

전체 입력 검사에서 train6,440개·validation377개의 전좌표0 window가 확인됐다. 이를 새 제외 기준으로
사용하지 않고 기존 masked mean 관례에 맞춰 유효 spatial token이 없으면 context0으로 정의했다.
Temporal 표현은 DSTE의 출력을 유지한다. 이 입력 정의를 학습 시작 전에 별도 revision으로
명시했으며 표본·split·모델·optimizer·선택 기준은 변경하지 않았다. 성능을 보고 조정한 것이 아니다.

[선행 F0B 평가](2026-09-19_f0b_postselection_shared.md)와
[전체 재현 현황](2026-09-19_recovery_restart_shared.md)을 함께 따른다.
