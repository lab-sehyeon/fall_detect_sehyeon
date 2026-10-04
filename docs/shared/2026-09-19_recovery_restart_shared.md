# 전체 실험 재현 재개 현황

- 문서 ID: `DOC-20260919-recovery-restart-R12`
- 기준일: 2026-09-22
- 연구 상태: `paused` — V2 controls·V1/V2 joint 완료, V3 평가 정의 검토 필요

## 최신 연구 진행 — 완료 결과와 남은 검증

[전체 진행](2026-09-22_recovery_execution_shared.md)에서 현재 상태를 구분한다.
[V2 대조 모델](2026-09-22_safer_v2_controls_shared.md)과
[J0/J1 V1/V2](2026-09-22_joint_reconstruction_shared.md)의 전체 평가·독립 검산을 완료했다.
V1/V2 J1의FU nestedF1은93.373/92.683%이며, V2에서는오탐과recall이함께줄었다.
[FU4분류기](2026-09-22_fu_classifier_reconstruction_shared.md)도nested검산을완료했지만V3reference는없다.
[V3pilot](2026-09-22_safer_v3_reconstruction_shared.md)은원본좌표0프레임620개로
별도정의한퇴화입력검사를미통과했다. full생성·V3후속학습은중단했고평가정의추가검토가필요하다.
원본과동일한수치복원이나후반연구전체완료를뜻하지않는다. 아래이전진행은당시이력이다.

## 이전 진행 — R11 당시 상태

[SAFER V2](2026-09-20_safer_v2_reconstruction_shared.md)의 전체497개 sequence와
1,007,723windows 생성·전수 검산을 완료했다.
[Matched controls](2026-09-22_safer_v2_controls_shared.md)는 전체 실행 중이고
[J0/J1](2026-09-22_joint_reconstruction_shared.md)은 방법·구현 검증 단계다.
V3와 후반 연구는 남아 있으며 전체 복구 완료나 낙상 성능 개선을 뜻하지 않는다.

후속 전체에 확인된 역사적 조건과 새로 고정한 세부를 분리하는 별도 재구현 범위를 적용한다.
FU ZS0/native30/aligned25의 전체993clips 평가·독립 검산을 완료했으며 F1은
23.705/20.000/20.449%다. [ZS 결과](2026-09-20_fu_zs_reconstruction_shared.md)에 한계를
분리했다. [D0/D1/D2](2026-09-20_fu_probe_reconstruction_shared.md)는 동일 시간 처리의
전체5fold 개발 비교와 독립 검산도 완료했다. Native30 D0-TS/D1/D2 개발 OOF F1은
95.575/91.765/95.575%다. [P0 비교](2026-09-20_primitive_reconstruction_shared.md)도 완료했다.
결합 F1은95.808%지만 fall TP가162→160으로 줄어 채택 기준에 미달했다.
P1은 전체 초기 동등성·10epoch 학습·독립 검산을 완료했다. epoch2의 validation Macro-F1은
75.313%, fall F1은77.419%다. test/OOD는 미평가다. V2/V3·J0/J1·RGB/외부는 남아 있다.

삭제 전 완료·중단·미채택 실험을 각각의 데이터·전처리·split·학습·평가 조건에 맞춰 재현한다.
기존 [공유 통합본](2026-09-03_project_complete_summary_shared.md)의 연구 배경과 역사적 결과를
유지하면서, 이 문서에서 현재 확인된 진행 상태를 구분한다.

NTU60 ADL, FU F0와 reconstructed SAFER V1 F0B validation 결과는 앞선 재현 산출물에서 재확인했다.
후속 F0B test/OOD 전체 평가와 독립 검증도 완료했다. 삭제 전 원본 산출물의 완전 복원과는 구분한다.

## 기존 재현 결과 재검산

| 실험 | 검증 범위 | 재확인한 수치 |
| --- | --- | --- |
| NTU60 ADL epoch150 | validation 16,487개 저장 예측 | Top-1 85.285%, Top-5 97.234% |
| FU F0 recovery control | 993개, subject-disjoint 5-fold, OOF 1회 포함 | F1 94.611%, AUPRC 98.620% |
| SAFER F0B temporal-only | validation 104,589개 | epoch35, Macro-F1 65.456%, fall F1 70.920% |
| SAFER F0B temporal+spatial | 같은 validation | epoch46, Macro-F1 69.913%, fall F1 74.345% |

이 수치는 기존 예측에서 다시 계산한 검증값이며 새로운 학습 결과가 아니다. FU와 SAFER recovery
control은 유실된 원본 학습·전처리 세부를 완전히 복원한 실험으로 주장하지 않는다.

## 새로 완료한 F0B 고정 모델 평가

| Split | Windows | Macro-F1 | Fall F1 | Fall-vs-lie AUPRC |
| --- | ---: | ---: | ---: | ---: |
| Test | 216,768 | 71.505% | 79.228% | 95.573% |
| OOD | 71,964 | 53.452% | 52.744% | 85.659% |

Validation에서 선택한 TS epoch46을 그대로 적용했다. 전체 평가·입력 순서·모델 불변성·저장 예측
독립 재검산을 통과했으며 추가 학습이나 threshold 조정은 없었다.

## 실험별 현재 재현 상태

| 실험 | 현재 상태 | 다음 연구 단계 |
| --- | --- | --- |
| E01 NTU60 ADL | 생존 결과 재검산 완료 | 별도 F0-NTU sanity/binary control 계약 확인 |
| E02 FU-Kinect | F0·ZS/D0/D1/D2 완료, P0 완료·미채택 | J0/J1·nested 재현 |
| E03 SAFER | V1 controls·F1·P1 완료, V2 생성 중 | V2 전체 검산 → matched controls·V3 pilot |
| E04 J0/J1 | 역사적 완료, 현재 재현 대기 | 계보별 development → locked/nested → Final J1 |
| E05 LaDy | 역사적 미채택, 현재 재현 대기 | L0/L1/L1-C/L2 및 진단의 설정·선행 모델 확인 |
| E06 Global Motion/RGB | 일부 모듈 재구성, 성능 재현 대기 | 특징 정의·scaler·G0/G1/G2 계약 검증 |
| E07 CAUCAFall | 역사적 진단 결과만 보존 | E0/E1/E2·RGB·ADL matched 재현 |
| E08 OOPS/OmniFall | 역사적 진단 결과만 보존 | coverage·front-end variant·D0/D1·잔여 실패 재현 |
| E09 상태·회복 | 일부 모델·decoder 구조 재구성 | S0 계열과 event protocol별 재현 |
| E10 Le2i | 역사적 engineering 결과만 보존 | 동일 적격성·decoder·primary/post-hoc 조건 재현 |
| E11 외부 봉인 평가 | `planned` | core 재현 및 protocol 고정 후 실행 |
| E12 VLM·TrackMemory·YAMNet | standalone 일부 기록, 통합 `planned` | 완료된 진단과 미실행 확장 설계 구분 |
| E13 비교 모델 | 역사적 결과와 provenance 기록 보존 | classifier controls 및 모델별 동일 modality 계약 확인 |

F1, ZS0/ZS1, D0/D1/D2, P0/P1, V1/V2/V3별 모델과 미채택 ablation도 복구 범위에 포함한다.
과거 계획에만 있던 VLM/audio/봉인 실험을 이미 완료된 연구로 표현하지 않는다.

## 동일성 한계와 다음 단계

SAFER V1의 원본 mapper와 과거 F0B optimizer 설정이 유실돼, 직전 재구성 실험을 이어가는 것과 삭제
전 실험의 정확한 재현은 구분해야 한다. 완료된 평가는
[F0B 결과 기록](2026-09-19_f0b_postselection_shared.md)을 따른다.

F1은 dense temporal 표현, loss 후보와 선택 지표는 확인되지만 temporal block과 학습 세부가 부족하다.
문서만으로 확정하지 못한 값은 원본 설정인 것처럼 보충하지 않고 별도 재구현 계약으로 구분한다.
[F1 계약 정리](2026-09-20_f1_reconstruction_contract_shared.md)를 시작점으로 이후 입력 교정, 공동학습,
외부 진단 순으로 진행하며 각 단계의 결과와 한계를 별도 실험 문서로 갱신한다.

F1은 별도 재구현 조건을 고정하고 네 후보 20-epoch 학습·고정 평가·독립 검산을 완료했다.
전좌표0 window를 배제하지 않고 masked spatial context0을 사용하는 두 번째 revision이다.
선택된 temporal+spatial/sqrt epoch17의 test/OOD Macro-F1은 75.278/56.516%, fall F1은
79.063/54.271%다. [F1 실행 결과](2026-09-20_f1_run_shared.md)와
[재현 계약·coverage](2026-09-20_f1_reconstruction_contract_shared.md)를 따른다.
원본과 동일한 실험은 아니며 F0B window 지표와 평가 단위가 달라 직접적인 개선량으로 비교하지 않는다.
후속 ZS와 probe는 완료했다. Primitive와 V2/V3는 아직 시작하지 않았다.

후속 입력·모델 자산의 동일성을 재확인하고 단계별 확인된 조건과 미기재 조건을 분리했다.
Clip 판정·FPS sampling·학습 세부 등은 각 실험 시작 전에 별도 재구현 config로 고정한다.
[F1 이후 실행 조건](2026-09-20_post_f1_execution_contracts_shared.md)에 후속 전체 선행 관계와
원본 동등성의 한계를 정리했다. 이전 정의 대기는 해제됐다.
