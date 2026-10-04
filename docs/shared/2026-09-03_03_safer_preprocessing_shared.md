# E03 — SAFER 전처리 V1/V2/V3와 temporal baseline

- 문서 ID: `DOC-20260903-exp03-safer-preprocess-R25`
- 기준일: 2026-09-22
- 연구 상태: V1·F1·P1·V2 controls 완료, V3 pilot `not_selected`

## 최신 연구 진행 — 완료 결과와 남은 검증

[전체 진행](2026-09-22_recovery_execution_shared.md)에서 현재 상태를 구분한다.
[V2 대조 모델](2026-09-22_safer_v2_controls_shared.md)과
[J0/J1 V1/V2](2026-09-22_joint_reconstruction_shared.md)의 전체 평가·독립 검산을 완료했다.
V1/V2 J1의FU nestedF1은93.373/92.683%이며, V2에서는오탐과recall이함께줄었다.
[FU4분류기](2026-09-22_fu_classifier_reconstruction_shared.md)도nested검산을완료했지만V3reference는없다.
[V3pilot](2026-09-22_safer_v3_reconstruction_shared.md)은원본좌표0프레임620개로
별도정의한퇴화입력검사를미통과했다. full생성·V3후속학습은중단했고평가정의추가검토가필요하다.
원본과동일한수치복원이나후반연구전체완료를뜻하지않는다. 아래이전진행은당시이력이다.

## 이전 진행 — R24 당시 상태

[별도 V2 방법·검증](2026-09-20_safer_v2_reconstruction_shared.md)의 전체497sequences/
8,091,357frames와1,007,723windows 생성·독립 검산을 완료했다.
공식 confidence 순서의 의미 불일치와 proxy3D의 한계를 남기고 V1과 구분한다.
[F0A/F0B matched controls](2026-09-22_safer_v2_controls_shared.md)는 전체 실행 중이다.
[J0/J1](2026-09-22_joint_reconstruction_shared.md)은 방법·구현 검증, V3 pilot는 미실행이다.

## 완료 결과 — F1 별도 재구현

F1 이후 FU ZS0/ZS1과 D0/D1/D2 진단도 전체 실행·독립 검산을 완료했다.
[ZS 결과](2026-09-20_fu_zs_reconstruction_shared.md)와 [probe 결과](2026-09-20_fu_probe_reconstruction_shared.md)를
따른다. 개발 OOF 지표다. [P0](2026-09-20_primitive_reconstruction_shared.md)는 전체 검산 후 미채택,
P1은 전체 초기 동등성·10epoch 학습·독립 검산을 완료했다. epoch2 validation Macro-F1
75.313%, fall F1 77.419%이며 test/OOD는 미평가다. V2 후속 비교·V3와 최종 J1은 아직 남아 있다.

확인된 역사적 방법과 별도 명시한 미기재 설정을 구분한 F1 전체 실험과 독립 검산을 완료했다.
전좌표0 입력도 제외하지 않고 spatial context0으로 처리했다. 네 후보 20-epoch 학습 후 validation에서
temporal+spatial/sqrt epoch17을 선택했다. 고정 test/OOD Macro-F1은 75.278/56.516%,
fall F1은 79.063/54.271%다. F0B와 평가 단위가 달라 직접적인 개선량으로 비교하지 않는다.
[계약](2026-09-20_f1_reconstruction_contract_shared.md)과
[자동 실행 기록](2026-09-20_f1_run_shared.md)에 진행과 최종 검증을 구분한다.

## 최신 진행 — 고정 F0B 평가 완료

Validation에서 고정한 temporal+spatial epoch46의 전체 test/OOD 평가를 완료했다.
Test 216,768개에서 Macro-F1 71.505%, fall F1 79.228%, conditional AUPRC 95.573%였으며,
OOD 71,964개에서는 각각 53.452%, 52.744%, 85.659%였다. 모델 불변성·입력 순서·전체 범위와
독립 metric 재검산을 통과했다. 이는 원본과 동일한 실험 또는 과거 대비 인과적 개선의 증거가 아니다.

이하 이전 revision의 평가 대기 상태는 이 완료 기록으로 대체한다.
[F0B 상세 결과](2026-09-19_f0b_postselection_shared.md),
[다음 F1 재현 계약](2026-09-20_f1_reconstruction_contract_shared.md)을 따른다.

## 연구 질문과 데이터

SAFER의 frame label로 fall, lie-down, lying-down temporal 표현을 학습하고, legacy 3D 입력의
topology·timeline 문제를 교정했을 때 낙상 신호가 회복되는지 평가했다. 기존 기록은 497 sequences,
8,091,357 frames와 `other/fall/lie_down/lying_down` 4-class 설정을 사용했다.

## 전처리 교정

- V1에서 H36M17 출력을 COCO17 순서로 해석한 topology mismatch를 확인했다.
- 243-frame resampling 결과를 원래 timeline에 잘못 배치한 정렬 문제를 확인했다.
- V2는 원본 2D pose에서 native timeline으로 다시 lifting하고 H36M17→NTU25 mapping을 명시했다.
- V3는 243-frame, stride 121, triangular weighting과 edge floor 0.05의 overlap-add를 적용했다.

## 기존 기록 기준 결과

| 평가 | V1 | V2/V3 | 해석 |
| --- | ---: | ---: | --- |
| F0A Test F1 | 14.640% | 47.436% | 교정 후 신호 회복 |
| F0A OOD F1 | 4.834% | 22.378% | domain gap 지속 |
| F0B Test Macro-F1 | 55.584% | 55.306% | 일방적 성능 개선은 아님 |
| F0B OOD fall F1 | 21.775% | 25.578% | +3.803%p |

V3는 external-reference seam candidate rate를 79.490%에서 2.011%로 줄였다. V2와 분류 성능은
실질적으로 동등했으므로 V3는 정확도 향상보다 물리적으로 타당한 temporal preprocessing으로
채택됐다.

## 한계와 다음 연구

OOD 성능은 내부 test보다 낮아 domain gap이 남아 있다. 재현 시 V2/V3 계보를 분리하고 동일한
split·window·mapping 계약으로 matched comparison을 수행한다.

현재 normal, non-lab OOD의 pose 자료와 공식 CSV·split 최소 묶음을 확보하고 원본 감사를 완료했다.
In-lab 467개와 OOD 30개, 총 497 sequences·8,091,357 frames가 기존 기록과 일치했다. 새 wheelchair
subset은 역사적 497-sequence 재현에 섞지 않고 별도 연구로 분리한다.

모든 공식 2D 좌표는 finite였지만 legacy 3D에는 기존 기록과 같은 193 sequences·93,256 frames의
non-finite 값이 있었다. 따라서 과거 실험과 동일하게 첫 legacy 입력에서는 non-finite frame을 포함한
64-frame window를 제외하며 전체 궤적을 임의 보간하지 않는다.

다음은 역사적 순서대로 legacy `clean3d_v1`, F0A A043 zero-shot, F0B 4-class linear control과 F1
temporal adapter를 복구하는 것이다. 이후 알려진 topology·timeline 결함을 교정한 V2/V3를 별도
계보로 재현한다. 이 선행 control 전에는 J0/J1을 시작하지 않는다.

대용량 입력 생성 전 count-only 검증도 통과했다. 공식 subject split 안에서 validation은 subject
2, 12, 24, 25, 34, 35로 구성되며 train/validation/test는 각각 298/73/96 sequences다. 64-frame,
stride-8에서 꼬리창을 추가하지 않고 legacy 3D 비유한 frame 포함 window를 제외하면
train/validation/test/OOD가 정확히 599,986/104,589/216,768/71,964 windows가 된다. 비유한 값을
제외하지 않은 candidate 수 609,183/106,680/219,896/71,964도 기존 V2 기록과 일치한다.

과거 mapper와 산출물이 남아 있지 않아, 보존된 연구 기록과 공식 upstream을 근거로 legacy 호환 후보를
명시적으로 구성하는 실험 복원을 시작했다. 이는 원본 구현의 byte-identical 복구가 아니라
`experimental compatibility reconstruction`이다.

네 legacy 후보를 subject-disjoint validation 104,589 windows에서 비교했다. 과거 F0A validation
`7.989/8.249/61.446%`와 가장 가까운 후보도 `1.109/8.997/59.481%`였으며, 세 지표를 모두 충분히
재현하지 못했다. 또한 이번 평가에 사용한 ADL head가 과거 epoch-150 head가 아니라 이후 성능 선택으로
저장된 epoch-155 head임을 확인했다. 따라서 어떤 후보도 채택하지 않았고 test/OOD는 후보 선택에
사용하지 않았다.

동일한 공식 설정을 정확히 epoch 150에서 종료한 ADL head를 별도로 재생성해 과거 Top-1 85.285%,
Top-5 97.234%를 다시 확인했다. 이 head로 네 원래 후보와 좌표 크기 차이를 근거로 추가한 네 후보를
전체 validation에서 재평가했으나, 어떤 후보도 과거 세 지표 각각 1.0%p 이내라는 사전 gate를 통과하지
못했다. 가장 가까운 후보도 F1 8.826%, AUPRC 9.788%, fall-vs-lie AUPRC 59.096%였다.

모든 비교는 subject-disjoint validation만 사용했고 test/OOD는 열거나 후보 선택에 사용하지 않았다.
과거 mapper와 정확한 metric 구현 및 weight가 유실된 상태에서 validation 수치에 맞춰 후보를 계속
추가하는 것은 사후 과적합이므로 metric 기반 복구는 여기서 종료한다.

보존 연구 기록과 공식 전처리 계약에 가장 직접적인 COCO semantic proxy, framewise center, window 첫
frame full-3D shoulder alignment, scale 미적용 구성을 구조적 호환안으로 고정했다. 이는 원본 mapper의
완전 복원이 아니라 연구 근거 기반 실험 재구성이며 이 한계를 manifest에 남긴다. 전체 window 수와
label/order 무결성을 확인한 뒤 F0A, F0B, F1 순서로 진행한다.

이 구조적 호환안을 고정한 입력 생성기를 구현하고 공식 SAFER 자료를 사용한 소규모 검증을 완료했다.
네 split의 전체 적격 window 수를 다시 확인했으며, 각 split에서 실제 window를 생성해 좌표 유한값,
두 번째 사람 zero-fill, 중심·어깨 정렬, coarse-to-4-class label 관계와 sample order 검사를 통과했다.
소규모 산출물은 본 실험에 사용할 수 없도록 명시적으로 구분했다.

전체 993,307개 window 생성과 payload 검사를 완료했다. train/validation/test/OOD는 각각
599,986/104,589/216,768/71,964 windows이며 좌표, 두 번째 사람, label, sample order와 split 무결성
검사를 모두 통과했다. 이 입력은 구조적 호환 재구성 결과이며 과거 mapper의 완전 복원으로 간주하지
않는다.

다음 역사적 control인 F0A 평가기도 구현해 소규모 입력에서 검증했다. DSTE와 epoch-150 ADL head를
고정하고 학습이나 threshold 조정 없이 NTU A043 argmax만 fall로 판정한다. 입력·모델 불변성 검사는
통과했으며 소규모 prefix의 지표는 class coverage가 불완전하므로 성능 결과로 사용하지 않는다.

reconstructed V1의 validation/test/OOD 전체 F0A 평가를 완료했다. 학습이나 threshold 조정 없이 고정
A043 argmax를 사용한 결과는 다음과 같다.

| Split | Fall F1 | All-action AUPRC | Fall-vs-lie AUPRC |
| --- | ---: | ---: | ---: |
| Validation | 0.000% | 9.433% | 55.184% |
| Test | 1.784% | 14.334% | 69.560% |
| OOD | 1.925% | 2.872% | 54.271% |

모델·입력 불변성과 전체 split 평가 조건은 모두 통과했다. 고정 ADL decision의 fall F1은 매우 낮지만
fall-vs-lie 순위 신호는 남아 있어, SAFER supervision으로 새 분류 경계를 학습해야 한다는 negative
control 결론을 지지한다. 이 결과를 본 뒤 threshold를 조정하지 않았다.

다음은 Frozen DSTE 위에서 temporal-only 1024-D와 temporal+spatial 2048-D 표현에 각각 4-class linear
head를 학습하는 F0B다. validation macro-F1 우선, fall-vs-lie AUPRC tie-break로 후보를 고정하고
test/OOD는 선택 후에만 평가한다. 보존 기록에서 확인되지 않은 학습 세부값은 임의로 채우지 않고 계약
복구를 먼저 완료한 뒤 F1 temporal adapter로 이동한다.

F0B의 첫 단계로 frozen DSTE train/validation feature extractor를 구현하고 단위·회귀 검증을 완료했다.
각 64-frame window에서 temporal 1024-D와 spatial 1024-D max-pooled feature를 분리 저장하고,
temporal+spatial 후보는 이후 두 표현을 결합해 구성한다. 이 단계는 모델을 학습하거나 checkpoint를
선택하지 않는다.

추출기는 train/validation만 허용하며 test/OOD는 열지 않는다. 입력 label·sample order, feature shape와
유한값, DSTE/ADL 불변성을 모두 통과한 전체 cache만 다음 4-class head 실험에 사용한다. 구현 검증은
완료됐다.

train 599,986개와 validation 104,589개 전체에서 temporal/spatial 1024-D frozen feature 추출을
완료했다. 전체 payload, label·sample order, feature 유한값, DSTE/ADL 불변성과 train/validation-only
조건이 모두 통과했으며 test/OOD는 열지 않았다.

다음은 temporal-only와 temporal+spatial 4-class linear head 비교다. 50 epochs와 validation-only
selection 규칙은 확인됐지만 보존 기록에서 확인되지 않은 optimizer 계열 설정은 원래 실험과 동일한
값으로 간주하지 않는다. 추가 근거 또는 별도 recovery configuration을 고정한 뒤 head 학습을 시작한다.

원래 optimizer가 유실된 한계를 유지하면서 별도 recovery control 설정을 결과 전에 고정했다. 공식
FoundSkelModel linear evaluation의 SGD, plain cross-entropy, 초기화와 batch 규칙을 사용하고, 보존된
F0B 기록의 50 epochs, 두 representation과 validation 선택 규칙을 결합한다. 이 결과는 역사적 F0B의
exact reproduction으로 주장하지 않는다.

두 후보는 같은 minibatch 순서를 받으며 class weight나 early stopping을 사용하지 않는다. 매 epoch 전체
validation에서 4-class macro-F1을 우선하고 conditional fall-vs-lie AUPRC를 동률 기준으로 사용한다.
train/validation 전용 trainer 구현과 단기 GPU 실행 검증 후 전체 50-epoch 학습을 완료했다. 단기
실행 결과는 후보 성능에 사용하지 않았으며, 전체 validation에서 사전 고정한 기준으로 후보와 epoch를
선택했다.

## F0B recovery validation 결과

| Candidate | Best epoch | Accuracy | Macro-F1 | Fall F1 | All-action fall AUPRC | Fall-vs-lie AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Temporal only | 35 | 95.659% | 65.456% | 70.920% | 76.776% | 95.199% |
| Temporal + spatial | 46 | 96.140% | 69.913% | 74.345% | 78.983% | 95.814% |

primary selection metric인 4-class macro-F1에 따라 temporal+spatial epoch 46을 고정했다.
temporal-only보다 macro-F1은 4.456%p, fall F1은 3.425%p 높았다. 이는 reconstructed V1 feature와
결과 전에 고정한 SGD recovery configuration의 validation 결과이며, 유실된 역사적 optimizer를 정확히
복원한 결과로 주장하지 않는다.

validation selection 이전까지 test/OOD는 사용하지 않았다. 다음 단계에서는 선택된
temporal+spatial epoch-46 head를 변경하거나 추가 학습하지 않고 test와 OOD를 한 번 평가한다.

## 기존 F0B와의 validation 비교

| 결과 | 기존 기록 V1 | 현재 reconstructed V1 recovery | 차이 |
| --- | ---: | ---: | ---: |
| 선택 representation | Temporal + spatial | Temporal + spatial | 동일 |
| Macro-F1 | 53.952% | 69.913% | +15.961%p |
| Fall F1 | 54.186% | 74.345% | +20.159%p |
| Fall-vs-lie AUPRC | 96.477% | 95.814% | -0.663%p |

선택된 표현과 fall-vs-lie ranking 수준은 비슷하지만 Macro-F1과 Fall F1은 기존 기록보다 크게 높다.
따라서 전체 결과를 기존 실험과 수치적으로 비슷한 재현이라고 판단할 수 없다. 과거 optimizer 세부값과
원본 V1 mapper가 소실돼 차이 원인을 통제할 수 없으므로, 높은 현재 수치를 성능 개선으로 주장하지
않는다. 현재 test/OOD 평가는 아직 수행하지 않아 기존 test/OOD 결과와의 비교도 보류한다.

## Post-selection 평가 준비

Validation에서 고정한 temporal+spatial epoch-46 head만 test와 OOD에서 평가하는 전용 evaluator를
구현하고 실행 전 검증을 완료했다. 이 단계에는 학습, optimizer, threshold fitting, 후보 비교나
checkpoint 재선택 기능이 없다. 제한 표본 test/OOD metric을 먼저 보는 smoke도 수행하지 않는다.

평가는 test 216,768 windows와 OOD 71,964 windows 전체를 고정 순서로 한 번 처리한다. frozen DSTE
feature, 4-class logits와 source order를 보존해 결과를 감사할 수 있게 하며 DSTE·ADL·selected head가
평가 전후 동일한지도 검사한다. 실행 전 검증에서는 test/OOD payload를 열지 않았고, 전체 단일 평가는
아직 대기 중이다.
