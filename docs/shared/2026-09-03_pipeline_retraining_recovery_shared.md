# 파이프라인 재학습 순서 연구 진행 공유

- 문서 ID: `DOC-20260903-pipeline-retraining-recovery-R26`
- 기준일: 2026-09-22
- 연구 상태: `paused` — V2 controls·V1/V2 joint 완료, V3 평가 정의·후반 데이터 검토 필요

## 최신 연구 진행 — 완료 결과와 남은 검증

[전체 진행](2026-09-22_recovery_execution_shared.md)에서 현재 상태를 구분한다.
[V2 대조 모델](2026-09-22_safer_v2_controls_shared.md)과
[J0/J1 V1/V2](2026-09-22_joint_reconstruction_shared.md)의 전체 평가·독립 검산을 완료했다.
V1/V2 J1의FU nestedF1은93.373/92.683%이며, V2에서는오탐과recall이함께줄었다.
[FU4분류기](2026-09-22_fu_classifier_reconstruction_shared.md)도nested검산을완료했지만V3reference는없다.
[V3pilot](2026-09-22_safer_v3_reconstruction_shared.md)은원본좌표0프레임620개로
별도정의한퇴화입력검사를미통과했다. full생성·V3후속학습은중단했고평가정의추가검토가필요하다.
원본과동일한수치복원이나후반연구전체완료를뜻하지않는다. 아래이전진행은당시이력이다.

## 이전 진행 — R25 당시 상태

[SAFER V2](2026-09-20_safer_v2_reconstruction_shared.md)는 전체497개 sequence와
1,007,723개 입력 window 생성·전수 검산을 완료했다.
[Matched controls](2026-09-22_safer_v2_controls_shared.md)는 전체 실행 중이다.
[J0/J1](2026-09-22_joint_reconstruction_shared.md)의 방법·구현 검증을 진행했으며 학습 성능은 없다.
이후 V3 label-blind pilot와 후반 연구로 이어간다.

[P0 비교](2026-09-20_primitive_reconstruction_shared.md)는 검산을 완료했지만 fall TP 손실로
미채택이다. P1은 전체 학습·독립 검산을 완료했다. epoch2 validation Macro-F1 75.313%,
fall F1 77.419%이며 test/OOD는 미평가다. 이후 V2/V3와 J0/J1로 이어간다.

FU ZS0/ZS1과 D0/D1/D2 전체 별도 재구현·독립 검산을 완료했다. Native30 D0-TS/D1/D2 개발
OOF F1은95.575/91.765/95.575%다. [후속 진단 결과](2026-09-20_fu_probe_reconstruction_shared.md)를
따르며 validation epoch선택이 포함된 개발 결과를 nested 성능으로 해석하지 않는다.

F1은 미기재 조건을 별도로 고정한 문서 기반 재구현 실험과 독립 검산을 완료했다.
선택된 temporal+spatial/sqrt epoch17의 test/OOD Macro-F1은 75.278/56.516%, fall F1은
79.063/54.271%다. [실행 결과](2026-09-20_f1_run_shared.md)를 따르며 후속
V2 matched controls가 진행 중이고 V3 pilot는 남아 있다. F0B와 평가 단위가 달라 직접 비교하지 않는다.

고정 F0B test/OOD 평가와 독립 검산이 완료됐다. Macro-F1은 71.505/53.452%, fall F1은
79.228/52.744%다. 이하 이전 revision의 평가 대기 표현은 이 완료 결과로 대체한다.
[F0B 결과](2026-09-19_f0b_postselection_shared.md) 이후
[F1 재현 계약](2026-09-20_f1_reconstruction_contract_shared.md)을 따른다. 원본과의 동일성 한계를 유지한다.

## 1. 연구 목적

Frozen skeleton action representation을 기준으로 ADL 성능을 보존하면서 낙상 전문 분기와
전역 움직임 분기를 순서대로 재구성한다. 각 단계는 다음 모델의 입력과 비교 기준을 제공하므로
의존성을 건너뛰지 않는다.

## 2. 핵심 원칙

1. 공개된 공식 DSTE encoder를 우선 검증하고 사용할 수 있으면 불필요한 사전학습을 반복하지 않는다.
2. 공식 encoder가 없거나 입력·성능 조건을 충족하지 못할 때만 DSTE를 self-supervised 방식으로
   다시 사전학습한다.
3. 낙상 모델보다 NTU60 ADL linear baseline을 먼저 확립한다.
4. 낙상 학습 중 DSTE encoder와 ADL head를 변경하지 않는다.
5. 단순 linear control과 J0를 먼저 평가한 뒤 J1 residual adapter의 추가 효과를 측정한다.
6. 외부평가 결과를 본 뒤 threshold, checkpoint 또는 decoder를 사후 조정하지 않는다.

## 3. 모델 의존성

```text
DSTE encoder
  ├─ Frozen NTU60 60-class ADL head
  └─ Frozen skeleton feature
       ├─ SAFER/FU linear controls
       └─ J0 dataset-specific heads
            └─ J1 zero-init shared adapter
                 ├─ locked SAFER + nested FU evaluation
                 ├─ Final J1
                 │    └─ G0/G1/G2 global-motion probes
                 └─ optional S0 state/recovery heads

RGB pretrained front-end → proxy skeleton/global motion → Final J1 + G0/G2
                                                     └─ D1 event decoder
```

## 4. 권장 재학습 순서

| 순서 | 연구 단계 | 학습 대상 | 진행 조건 | 완료 기준 |
| ---: | --- | --- | --- | --- |
| 0 | DSTE 기준 확보 | 없음 | 공식 NTU60 3D/xsub encoder 우선 검증 | 구조와 공식 입력 좌표계 검증 완료 |
| 1 | 조건부 DSTE 사전학습 | DSTE encoder | 공식 encoder를 사용할 수 없거나 검증 실패 | NTU60 3D/xsub encoder 확정 |
| 2 | ADL baseline | Frozen DSTE 위 60-class linear head | DSTE 확정 | 완료: ADL 성능과 encoder 불변성 확인 |
| 3 | 단순 낙상 controls | FU binary 및 SAFER legacy F0A/F0B/F1 | ADL baseline 고정 | 역사적 입력·label·boundary 계약 확인 |
| 4 | Corrected SAFER | V2/V3 입력과 SAFER control | Legacy diagnostic 완료 | topology/time 교정과 continuity 확인 |
| 5 | J0 control | Dataset-specific heads | 동일 frozen feature 준비 | J1 비교 기준 확정 |
| 6 | J1 specialist | Zero-init shared residual와 두 heads | J0 확정 | Epoch-0 equality와 낙상/lying trade-off 확인 |
| 7 | Generalization 평가 | 새 학습 없음 | J0/J1 고정 | Locked SAFER와 nested FU 결과 확정 |
| 8 | Final fall model | Final J1 | Epoch 규칙 사전 확정 | Frozen ADL을 보존한 final model 확정 |
| 9 | Global-motion 비교 | G0/G1/G2 linear probes | Final J1 고정 | 상보성 및 domain별 trade-off 확인 |
| 10 | RGB 외부평가 | Front-end는 공식 pretrained model 사용 | Final J1과 global probe 확정 | Geometry/coverage와 외부 event 성능 평가 |
| 11 | Recovery 연구 | S0-A, S0-G0 | Core fall trigger 복원 후 | State/recovery의 별도 일반화 검증 |

## 5. 가장 먼저 학습할 모델

공식 DSTE encoder가 연구 조건을 충족하면 **NTU60 60-class ADL linear head가 첫 학습 대상**이다.
공식 encoder를 사용할 수 없거나 검증에 실패할 때만 **DSTE self-supervised pretraining을 먼저**
수행한다.

따라서 J1, G2 또는 S0부터 시작하지 않는다. J1은 J0 baseline과 frozen DSTE feature에 의존하고,
G2는 Final J1과 global feature에 의존하며, S0 recovery는 dense state supervision과 core fall
trigger가 먼저 필요하다.

## 6. 단계별 연구 검증 기준

### DSTE와 ADL

- NTU60 3D cross-subject protocol을 사용한다.
- 기존 기록 기준 ADL baseline은 Top-1 85.285%, Top-5 97.234%다.
- 낙상 모델 학습 전후 encoder와 ADL logits의 동일성을 검증한다.

### J0와 J1

- 동일한 feature, split과 dataset-specific head를 사용한다.
- J1의 zero-init 시점 출력이 J0와 동일해야 한다.
- Macro-F1뿐 아니라 fall precision/recall과 intentional-lying FPR을 함께 비교한다.
- Subject-disjoint, locked test/OOD와 nested evaluation을 분리한다.

### G0/G1/G2

- G0는 skeleton-only, G1은 global-only, G2는 두 표현의 결합으로 비교한다.
- 동일한 학습·선택 조건을 사용한다.
- 기존 연구에서 G2의 효과가 데이터셋에 따라 달랐으므로 G2를 사전에 최종 모델로 고정하지 않는다.

### RGB 및 recovery

- YOLOv8x, ViTPose-B와 MotionAGFormer-B는 초기 단계에서 재학습하지 않고 공식 pretrained model을
  사용한다.
- RGB front-end의 geometry와 coverage를 label-blind 방식으로 먼저 검증한다.
- D1은 학습 모델이 아니라 고정 score를 causal event로 변환하는 decoder다.
- 기존 외부평가에서 recovery 일반화가 충분하지 않았으므로 core fall model보다 후순위로 둔다.

## 7. 현재 연구 계획

- `completed`: NTU60 3D cross-subject 56,578개 샘플을 train 40,091개와 validation 16,487개로 구성함
- `completed`: ADL linear evaluation의 제한 배치 실행에서 60-class head만 학습되고 DSTE encoder가 변경되지 않음을 확인함. 제한 배치 정확도는 성능 결과로 사용하지 않음
- `not_selected`: 현재 가공 입력의 ADL linear head는 Top-1 79.899%, Top-5 94.899%로 역사적 성능 gate 미통과
- `completed`: 공식 NTU60 raw skeleton 56,880개 archive의 무결성과 S001–S017/A001–A060 범위 검증
- `completed`: 공식 missing-sample 302개 목록의 형식·중복·raw archive 포함 여부 검증
- `completed`: 원본 추출 후 전체 56,880개 및 제외 후 유효 56,578개 계약 검증
- `completed`: 전처리 호출을 이전 UmURL 중심·어깨축 조건과 일치시키고 regression test 통과
- `completed`: official UmURL XSub train/validation 입력과 완료 manifest 생성 및 공식 구현 출력 동등성 검증
- `completed`: 새 공식 입력을 선택하는 downstream profile과 frozen DSTE 제한 실행 검증
- `completed`: 동일 계약의 NTU60 60-class 전체 ADL linear evaluation에서 epoch 150 Top-1 85.285%, Top-5 97.234% 재현
- `completed`: FU-Kinect skeleton 1,006개 감사와 993개 품질 집합·21-subject 5-fold 전처리 재현
- `completed`: Frozen DSTE 기반 FU binary linear control — OOF F1 94.611%, AUPRC 98.620%
- `completed`: SAFER 497-sequence raw/lineage audit — 총 8,091,357 frames 확인
- `completed`: legacy `clean3d_v1` subject split과 window-count gate
- `completed`: 연구 근거 기반 structural reconstruction의 993,307-window materialization과 무결성 검증
- `completed`: reconstructed V1 F0A full — Validation/Test/OOD fall F1 0.000/1.784/1.925%, conditional AUPRC 55.184/69.560/54.271%
- `completed`: explicit recovery configuration의 F0B 50-epoch 학습과 validation selection — temporal+spatial epoch 46
- `in_progress`: 선택된 F0B head의 test/OOD 단일 평가
- `planned`: F0B test/OOD 평가 후 F1 temporal adapter
- `planned`: corrected V2/V3 입력 복구
- `planned`: J0/J1 matched 재학습
- `planned`: Locked/nested evaluation 후 Final J1 확정
- `planned`: G0/G1/G2 matched global-motion 비교
- `planned`: RGB 외부평가 재연결
- `paused`: S0 recovery의 우선 재학습
- `not_selected`: 기존 LaDy, motion-predicted tracking과 단순 camera-residual 확장

## 8. 다음 연구

Core skeleton pipeline과 외부평가가 다시 확립된 뒤 비식별 skeleton과 audio, vibration,
thermal/depth, radar, IMU 등 보조 센서의 late-fusion 연구를 별도 revision으로 시작한다. 이 단계도
기존 Frozen J1을 유지하고 sensor-only와 fusion을 matched comparison으로 평가한다.

원본 NTU60 좌표계 복구, ADL baseline 재검증, FU-Kinect 993개 전처리와 FU binary linear control이
완료됐다. FU control은 OOF Precision 93.491%, Recall 95.758%, F1 94.611%, AUPRC 98.620%를
기록했으며 오탐 11건은 모두 intentional lying이었다. SAFER 원본 감사도 497 sequences·8,091,357
frames로 통과했다. 이어 subject-disjoint split과 64-frame/stride-8 window gate를 통과하고 V1
train/validation/test/OOD 599,986/104,589/216,768/71,964 windows를 materialize했다. reconstructed V1
F0A 전체 평가는 고정 ADL decision의 낮은 F1과 남아 있는 ranking signal을 확인했다. 다음은 새 SAFER
4-class linear boundary를 검증하는 F0B이며, validation으로 고정한 뒤 test/OOD를 평가한다. 이후 F1,
corrected V2/V3와 J0/J1 학습으로 이동한다.

NTU 완료 조건과 이후 데이터 입수 순서는
[NTU 이후 실행·데이터 확보](2026-09-03_next_steps_dataset_download_shared.md)를 따른다.

## 9. 참고문헌

- [FoundSkelModel 공식 저장소와 Model Zoo](https://github.com/wengwanjiang/FoundSkelModel)
- [Foundation Model for Skeleton-Based Human Action Understanding](https://arxiv.org/abs/2508.12586)
- [UmURL 공식 전처리](https://github.com/HuiGuanLab/UmURL/blob/main/data_gen/preprocess.py)
- [NTU RGB+D 공식 데이터셋](https://github.com/shahroudy/NTURGB-D)
