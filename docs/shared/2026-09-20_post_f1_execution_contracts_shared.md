# F1 이후 후속 실험 — 재현 범위와 실행 조건

- 문서 ID: `DOC-20260920-post-f1-execution-contracts-R7`
- 기준일: 2026-09-22
- 연구 상태: `paused` — V2 controls·V1/V2 joint 완료, V3 후속 정의 검토 필요

## 최신 연구 진행 — 완료 결과와 남은 검증

[전체 진행](2026-09-22_recovery_execution_shared.md)에서 현재 상태를 구분한다.
[V2 대조 모델](2026-09-22_safer_v2_controls_shared.md)과
[J0/J1 V1/V2](2026-09-22_joint_reconstruction_shared.md)의 전체 평가·독립 검산을 완료했다.
V1/V2 J1의FU nestedF1은93.373/92.683%이며, V2에서는오탐과recall이함께줄었다.
[FU4분류기](2026-09-22_fu_classifier_reconstruction_shared.md)도nested검산을완료했지만V3reference는없다.
[V3pilot](2026-09-22_safer_v3_reconstruction_shared.md)은원본좌표0프레임620개로
별도정의한퇴화입력검사를미통과했다. full생성·V3후속학습은중단했고평가정의추가검토가필요하다.
원본과동일한수치복원이나후반연구전체완료를뜻하지않는다. 아래이전진행은당시이력이다.

## 이전 진행 — R6 당시 상태

확인된 역사적 조건과 미기재 세부를 구분하고, 새 세부는 각 실행 전에 고정하는 재구현 범위를
후속 전체 단계에 적용한다. 이미 실행한 단계의 새 정의와 검증 결과는 각 실험 문서에 고정했다.

현재 ZS0/ZS1·D0/D1/D2·P0/P1의 전체 비교와 검산은 완료됐다.
[P0](2026-09-20_primitive_reconstruction_shared.md)는 결합 F1이 소폭 높지만 낙상 TP가 줄어
사전 채택 기준에 미달했다. P1은 전체 초기 동등성·10epoch 학습·독립 검산을 완료했으며
epoch2의 validation Macro-F1은75.313%다. test/OOD 일반화는 미검증이다.
아래 표의 P0/P1 미기재 항목은 별도 재구현 정의로 해결했다.
[V2 전처리](2026-09-20_safer_v2_reconstruction_shared.md)는 전체497개 sequence와
1,007,723windows 생성·전수 검산을 완료했다.
[Matched controls](2026-09-22_safer_v2_controls_shared.md)는 전체 실행 중이며,
[J0/J1 계약](2026-09-22_joint_reconstruction_shared.md)의 미기재 세부도 별도 정의로 고정했다.
공동학습 성능은 아직 없고 V3 label-blind pilot와 후반 연구도 남아 있다.

문서 기반 F1 재구현은 전체 학습·고정 평가·독립 검산을 완료했다.
후속 입력으로 사용할 FU 993 clips, SAFER 497 sequences 및 모델 자산의 동일성을 재확인했다.
이는 후속 실험의 성능 검증이나 완료를 의미하지 않는다.

1. SAFER→FU 무학습 전이 ZS0/ZS1과 DSTE/adapter/concat probe D0/D1/D2.
2. Body-relative primitive P0와 frozen F1에 대한 P1 correction.
3. SAFER corrected V2 및 V3 overlap-add 전처리와 matched controls.
4. J0/J1 development, locked SAFER, nested FU와 Final J1.
5. RGB/global-motion 통합 및 데이터셋별 외부·상태/회복 평가.

## 확인된 조건과 남은 정의

| 실험 | 유지할 확인된 조건 | 실행 전 고정이 필요한 항목 |
| --- | --- | --- |
| ZS0/ZS1 | whole-clip64 또는 contiguous64/stride8, native30/aligned25, padding output 제외, overlap raw-logit 평균 | clip-level score·판정, FPS sampling 세부, 끝부분 window 정책 |
| D0/D1/D2 | 같은 시간 처리·subject-disjoint probe, 원본 표현/adapter hidden/concat 비교 | clip feature pooling, 학습·선택 세부 |
| P0/P1 | body-relative12 signals/156D, zero-init correction, frozen backbone·temporal층 | 정확한 signal·aggregation 수식, 학습 부분집합·optimizer |
| V2/V3 | native243, 명시적 H36M17→NTU25, V3 stride121/triangular floor0.05, label-blind geometry 검증 | 입력 변환·normalization 전체 정의, 유실된 감사 metric 구현 |
| J0/J1 | dataset-specific heads, zero-init shared residual, ADL 보존, outer/inner 분리 | 일부 optimizer·loss·선택 규칙, 새 nested 결과와 final epoch 정책 |
| RGB·외부 평가 | 모델 고정 후 데이터셋별 별도 평가 | 입력 자산·적격성·평가 단위·decoder의 전체 계약 |

끝부분 window 정책만으로도 FU 입력 집계가 달라진다. 기존 stride 위치만 사용하면4,430개,
끝부분을 추가로 덮으면5,186개로756 clips가 영향을 받는다. 이 계산은 정책 차이를 보여주는
진단일 뿐, 둘 중 하나를 역사적 규칙으로 선택하거나 성능을 비교한 결과가 아니다.

## 재현 주장 경계

F1에 승인된 별도 재구현 조건을 후속 실험의 원본 설정으로 간주하지 않는다. 남은 정의는
실행 전에 고정하고 역사적 근거와 새 조건을 분리해야 한다. 학습·평가 결과를 본 뒤 과거 수치에
맞추어 설정을 선택하지 않는다. 이미 재구현한 F1을 사용하는 후속 결과에도 동일성 한계가 이어진다.

최초 기록에 계획으로만 남은 VLM/YAMNet 통합이나 외부 봉인 평가를 완료 실험의 복원으로
표현하지 않는다. 최초 R1 시점에는 후속 실험을 시작하지 않았다. R2에서 정의 대기를 해제했고,
[ZS0/ZS1 전체 평가](2026-09-20_fu_zs_reconstruction_shared.md)는 완료했으며
[D0/D1/D2 비교](2026-09-20_fu_probe_reconstruction_shared.md)도 전체 검산까지 완료했다.
V2/V3 생성과 이후 단계는 아직 완료되지 않았다.

[F1 재현 계약과 결과](2026-09-20_f1_reconstruction_contract_shared.md) ·
[전체 복구 현황](2026-09-19_recovery_restart_shared.md)
