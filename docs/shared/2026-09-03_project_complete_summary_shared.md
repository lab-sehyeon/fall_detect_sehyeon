# FoundSkelModel 낙상·이상행동 연구 공유 통합본

- 문서 ID: `DOC-20260903-project-complete-summary-R34`
- 기준일: 2026-09-30
- 연구 상태: in_progress — S0-E 완료·미채택, S0-F 진행; 문맥 회복·외부 평가 잔여
- 문서 목적: 전체 연구 배경, 방법, 결과, 한계와 현재 진행 상태를 하나의 공유 문서로 제공
- R2 갱신: 본문 인용표시와 사용·검토한 논문·공개자료 46개를 §26–§27에 통합
- R3 갱신: official UmURL NTU60 전처리 진행 상태와 이후 데이터·weight 확보 순서를 반영
- R4 갱신: official UmURL NTU60 입력 생성·검증 완료와 ADL 재평가 시작점을 반영
- R5 갱신: official UmURL downstream profile과 frozen DSTE 제한 실행 통과를 반영
- R6 갱신: official UmURL 전체 ADL 성능 재현과 다음 SAFER/FU control 단계를 반영
- R7 갱신: FU-Kinect 1,006개 원본 감사와 993개·21-subject 5-fold 전처리 재현 완료를 반영
- R8 갱신: F0-FU recovery control 완료, OOF 성능·hard-negative 오류 분포와 다음 SAFER 단계를 반영
- R9 갱신: SAFER 497-sequence raw audit 완료와 legacy `clean3d_v1` 선행 순서를 반영
- R10 갱신: SAFER legacy subject/window-count gate 완료와 mapping 검증 대기를 반영
- R11 갱신: reconstructed V1 전체 입력과 F0A full negative control 완료, F0B 다음 순서를 반영
- R12 갱신: F0B explicit SGD recovery full validation 완료, temporal+spatial epoch-46 선택과
  test/OOD 단일 평가 대기를 반영
- R13 갱신: 고정 F0B test/OOD 전체 평가·독립 검산 완료와 다음 F1 문서 기반 재현 계약을 반영
- R14 갱신: 문서 기반 F1 별도 설정·구현 검증 및 전체 실험 착수, 자동 진행 기록 연결
- R15 갱신: 별도 F1 네 후보 학습·선택·고정 평가·독립 검산 완료와 평가 단위 차이 반영
- R16 갱신: 전체 후속 별도 재구현 범위와 FU ZS/D0/D1/D2 검증 완료·개발 지표의 한계 반영
- R17 갱신: P0 전체 비교·미채택 결과와 P1 실행 진행 반영
- R18 갱신: P1 전체 학습·초기 동등성·독립 검산 완료, validation 소폭 개선과 한계 반영
- R19 갱신: V2 입력 전수 검사·첫 실제 sequence 검산 완료와 전체 생성 진행 반영

## 최신 연구 진행 — R34 S0-E 완료·S0-F 진행

[S0-E 규칙 비교](2026-09-30_s0e_reconstruction_shared.md)의72조건과고정평가를완료했다.
채택가능후보는0개이며진단후보는8개기준중4개만충족했다. Validation Macro-F1은60.813%,
lying_down F1은7.925%로전체상태품질을보존하지못해기존 S0-C를유지한다.
[S0-F 전이보조모델](2026-09-30_s0f_reconstruction_shared.md)은고정posterior35차원입력,
cap20/50,validation최저camera-view우선선택으로진행한다. 원본의누락세부값은별도실험의
사전조건으로명시했고아직최종성능은없다. 이후문맥회복·D1·외부event평가는남아있다.

## 이전 연구 진행 — R33 S0-B/C 완료·결과 분석

[상세 결과 분석](2026-09-29_s0bc_results_analysis_shared.md)에12epoch학습·44조건비교·최종평가를정리했다.
B는epoch7선택,Test Macro-F1 68.783%,낙상재현율82.870%로분류가개선됐지만
validation전환감소4.030%가25%기준에미달해단독미채택이다.
C는6개eligible중A+EMA0.5/연속3프레임/margin0.05선택이다. Test SegmentF1@50
13.508→49.304%,Edit18.778→66.851%,전환100.382→18.482/분으로상태가안정화됐다.
그러나낙상재현율68.616→65.396%(-3.220%p)이며OOD Macro-F1도36.380%다.
따라서validation상태안정화기준통과와낙상경보·회복event의운영적합성을구분한다.
저장예측전수검산·선택재계산·별도decoder재생은통과했으며후속S0-E/F/G·D1·외부event평가는남아있다.
아래과거실험·이전진행기록과R01–R46참고문헌은보존하며현재결과와혼합하지않는다.

## 이전 연구 진행 — R32 S0-B 동기식 병렬 계승

[S0-B 병렬계승](2026-09-29_s0b_acceleration_shared.md)은첫epoch373760windows의학습상태를
이어받아두복제본이각512개를동시에처리한다. Globalweighted loss로gradient를합산한뒤clip과
optimizer업데이트를한번만한다. 실질batch1024/FP32/12epoch/평가batch128/선택기준은불변이다.
업데이트검사와양복제본난수복원·제한검증을통과했으며추가Dropout난수seed1을고정했다.
기존단일계산과bit-exact동등성은주장하지않는다. [S0-C](2026-09-29_s0c_acceleration_shared.md)는
동일44조건으로이결과를기다린다. 정식성능·채택결과와후속회복·외부평가는아직남아있다.

## 이전 연구 진행 — R31 S0-B 마이크로배치512 계승

[S0-B 계승 실행](2026-09-29_s0b_mb512_shared.md)은 첫epoch217088windows까지의학습상태를
유지하며 마이크로배치128×8에서512×2로전환했다. 실질배치1024, 모델·데이터순서·학습률·12epoch,
검증·선택·평가기준은유지한다. 학습상태보존과새구성의제한검증을통과해진행중이다.
Dropout 난수 배정과연산순서차이로기존고정구성과bit-exact동등성을주장하지않는다.
[S0-C 비교](2026-09-29_s0c_mb512_shared.md)는동일44조건으로새학습완료를기다린다.
정식성능·채택결과는아직없으며GT0완료와잔여회복·외부평가범위는변경없다.

## 이전 연구 진행 — R30 S0-B/C 착수·GT0 완료

[S0-B](2026-09-29_s0b_reconstruction_shared.md)는학습전전체validation의857,528프레임출력이
S0-A와정확히일치함을확인한뒤12epoch학습을시작했다.
[S0-C](2026-09-29_s0c_reconstruction_shared.md)는고정44조건의동일후처리비교와전이지연진단을
준비했고선행실험완료후진행한다. 두후속모델의정식성능이나채택결과는아직없다.
[GT0 문맥회복타깃](2026-09-29_s0gt0_reconstruction_shared.md)은train/validation371개sequence에서
독립검산을완료했다. Train낙상2405개중2349개,val520개중512개가회복구간으로연결돼기존표본수와
일치했다. 모든8개view에회복표본이있고subject중복은없다. Test/OOD는이감사에사용하지않았다.
이는타깃준비완료이지회복검출기성능검증이아니며후속회복모델·외부event평가는남아있다.

## 이전 연구 진행 — R29 S0-A 상태 분류 완료 및 한계

[S0-A 결과](2026-09-29_s0a_reconstruction_shared.md): 두 후보 각12epoch를 학습하고 validation으로
sqrt epoch4를 선택한 후 고정 test/OOD 평가와 저장 예측 검산을 마쳤다.
Val/Test/OOD Frame Macro-F1은66.417/64.685/36.440%, test/OOD fall frame F1은74.712/51.370%다.
그러나 test/OOD SegmentF1@50은13.508/6.505%, 상태 전환/분은100.382/154.741이고,
누운 상태 F1도32.644/18.114%로 낮다. 기준 모델은 확보했지만 연속 상태 출력이나 회복 감지의
운영 성능을 확보한 것으로 해석하지 않는다. 원본과 조건 차이가 있는 별도 실험이며 일부 지표 상승과
과거 대비 fall recall 하락을 함께 기록한다. S0-B/C·문맥 회복 모델·외부 event 평가는 후속으로 남아 있다.

## 이전 연구 진행 — R28 S0-A 상태 분류 학습 착수

[S0-A 실험](2026-09-29_s0a_reconstruction_shared.md)의 제한 실행을 검증하고 두 후보의
전체12epoch 학습을 시작했다. 동결 DSTE 특징으로16개 frame 상태를 분류하고 plain/sqrt CE를 비교한다.
Validation 기반 선택 후 test/OOD를 평가하며 기존 ADL·낙상 모델은 변경하지 않는다.
[자동 진행 기록](2026-09-29_s0a_run_shared.md)에서 현재 단계를 확인할 수 있다.
정식 결과는 아직 없고 상태 안정화·회복·외부 event 평가는 후속 과제로 남아 있다.
소실된 세부 설정을 사전에 명시한 별도 실험이며 아래 역사적 S0 수치를 새 결과로 간주하지 않는다.

## 이전 연구 진행 — R27 RGB 통합 기술 검사 완료

[RGB 통합](2026-09-28_rgb_integration_shared.md)에서 고정된 OOPS3개 영상775프레임을 검사했다.
2개 영상44windows는 RGB→관절→3D스켈레톤→J1/G0/G1/G2 경로를 완주하고 저장값·CPU 검산을
통과했다. 1개 영상은 coverage76.821%로 사전 기준80%에 못 미쳐 후속 분류를 중단했다.
G0/G2의 fall-argmax window는 통과한2개 모두0개다. 이는 기술 연결 완료이지 낙상 성능 재현이나
외부 event 평가 완료가 아니다. S0·회복·전체 외부 평가 및 추가 비교는 남아 있다.
기존 모델과 G2 미채택 결론을 유지하며 과거 실험과 수치적 동일성을 주장하지 않는다.

## 이전 연구 진행 — R26 Global 학습·고정 평가 완료

[데이터 정합·G0/G1/G2 비교](2026-09-28_global_motion_training_shared.md)를 완료했다.
SAFER 좌표 정합과 OOPS 시간축을 별도 자료에서 수정했으며 원자료는 보존했다.
SAFER 497개 sequence의 1,007,723개 window를 검증하고 고정 J1 표현 위의 세 분류기를
각 50epoch 학습했다. Validation 선택 후 test/OOD 고정 평가와 독립 검산을 마쳤다.

| 모델 | Val Macro-F1 (%) | Test fall F1 (%) | OOD fall F1 (%) |
| --- | ---: | ---: | ---: |
| G0: skeleton | 75.360 | 79.560 | 53.018 |
| G1: global | 68.893 | 70.755 | 31.657 |
| G2: 결합 | 75.449 | 79.296 | 53.117 |

G2의 validation Macro-F1 향상은 +0.089%p로 사전 기준 +0.5%p에 못 미쳐 **미채택**이다.
OOD fall recall도 G0 75.000%에서 G2 70.576%로 감소했다. 기존 모델을 일괄 대체하거나
모든 도메인에서 개선됐다고 해석하지 않는다. 소실된 세부 설정을 사전에 명시한 별도 재구현이며
과거 결과와 동일하다는 주장은 하지 않는다. OOPS는 학습·선택에 사용하지 않았다.
RGB end-to-end, S0·회복 episode와 외부 모델 평가는 아직 남아 있다.
이하 이전 진행과 과거 결과·참고문헌 46개는 당시 근거로 보존한다.

## 이전 연구 진행 — R25 데이터 정합·Global 학습

[좌표·시간축 정합과 G0/G1/G2](2026-09-28_global_motion_training_shared.md)를 별도재구현으로 진행한다.
공식1920×1080리사이즈에따른SAFER30개공간정합과OOPS818개실제시간축검증을완료했다.
OOPS라벨643행의초과끝부분9.415초를제한했지만낙상914행과전체영상은유지했다.
이는문제확인후별도자료에적용한수정이며원본을동일하게복원했다는의미는아니다.

SAFER497개원자료와train/validation715,863개131D특징의source재계산·순서·인과성검사를통과했다.
고정J1/DSTE/ADL위에G0/G1/G2새분류기각50epoch학습을시작했다. 아직새확정성능은없다.
test/OOD는선택후고정평가하며OOPS는학습에사용하지않는다. 이하이전진행은당시이력이다.

## 이전 연구 진행 — R24 RGB 자료·입력 검사

[OOPS818·SAFER OOD30 원영상](2026-09-23_rgb_data_acquisition_shared.md)의 확보·검증을 완료했다.
[후속 입력 검사](2026-09-28_rgb_data_preparation_shared.md)에서 OOPS276,539프레임 전체 읽기와
라벨 정규화를 완료했고, 과거와 같은 배경 겹침2건24.033초를 제거했다. 낙상914행은 유지됐다.
36개는 비디오 트랙과 파일 전체 길이의 차이가 있어 입력 시간축 처리 규칙 확인이 필요하다.

SAFER 전체497개 pose 자료에는 전역 움직임에 필요한 bbox·좌표·confidence가 존재한다.
따라서 normal RGB 전체 확보 없이 global 입력 준비를 이어갈 수 있지만131D 생성·학습은 남았다.
OOD30개는 pose와 프레임 수가 일치하고25개는 이미지 해상도가 달라RGB결합 전 좌표 검증이 필요하다.
모델 외부평가·전체 연구 완료를 뜻하지 않는다. 이하 이전 진행은 당시 이력이다.

## 이전 연구 진행 — R23 V3 완료·후반 데이터 준비

별도사후평가정의를명시한 [V3 R3](2026-09-22_v3_valid_support_shared.md)의 전체497개 전처리,
[동일조건기준모델](2026-09-22_safer_v3_controls_shared.md),
[공동학습](2026-09-22_joint_v3_reconstruction_shared.md),
[FU고정분류기비교](2026-09-22_fu_classifier_reconstruction_shared.md)가완료됐다.
V3controls의test/OOD fallF1은80.551/56.109%,J1의FU nestedF1은93.333%다.
J1이모든기준분류기를능가한것은아니다. 원본동등성과전체후반연구완료도주장하지않는다.
[OOPS-Fall818·SAFER OOD30 원영상](2026-09-23_rgb_data_acquisition_shared.md)을 준비 중이며,
SAFER normal467개는아직미확보다. 이하이전진행은당시이력이다.

## 이전 연구 진행 — R22 별도 V3 재검산

[V3 R3 유효 근거 평가](2026-09-22_v3_valid_support_shared.md)를 시작했다.
이전 V3 미채택 결과를 보존한 별도 post-hoc 실험이다. 원본 프레임과 모델은 유지하고,
정의 가능한 2D 근거에서만 재투영·seam을 계산한다. 전체 시간축의 3D 검사는 유지한다.
검증을 통과한 경우에만 전체 전처리·기준 모델 비교·공동학습을 진행한다.
R3 pilot의 고정 기준은 통과했으며 현재 전체497개 전처리를 진행 중이다. 아직 R3 분류 성능은
없으며 [전체 진행](2026-09-22_recovery_execution_shared.md)에서 완료 범위를 구분한다.

## 이전 진행 — R21 완료 결과와 남은 검증

[전체 진행](2026-09-22_recovery_execution_shared.md)에서 현재 상태를 구분한다.
[V2 대조 모델](2026-09-22_safer_v2_controls_shared.md)과
[J0/J1 V1/V2](2026-09-22_joint_reconstruction_shared.md)의 전체 평가·독립 검산을 완료했다.
V1/V2 J1의FU nestedF1은93.373/92.683%이며, V2에서는오탐과recall이함께줄었다.
[FU4분류기](2026-09-22_fu_classifier_reconstruction_shared.md)도nested검산을완료했지만V3reference는없다.
[V3pilot](2026-09-22_safer_v3_reconstruction_shared.md)은원본좌표0프레임620개로
별도정의한퇴화입력검사를미통과했다. full생성·V3후속학습은중단했고평가정의추가검토가필요하다.
원본과동일한수치복원이나후반연구전체완료를뜻하지않는다. 아래이전진행은당시이력이다.

## 이전 진행 — R20 당시 상태

[V2 방법·검증](2026-09-20_safer_v2_reconstruction_shared.md)의 전체497sequences/
8,091,357frames와1,007,723windows 생성·전수 검산을 완료했다.
[V2 기준 모델 비교](2026-09-22_safer_v2_controls_shared.md)는 전체 실행 중이다.
[J0/J1](2026-09-22_joint_reconstruction_shared.md)은 방법·구현 검증을 진행했으며 아직 학습 성능은 없다.
이는 별도 문서 기반 재구현으로 원본 동등성·성능 개선을 뜻하지 않는다.
V3와 후반 연구는 남아 있으며 [전체 진행](2026-09-22_recovery_execution_shared.md)에서 구분한다.

## 1. 먼저 읽어야 할 결론

이 연구는 skeleton 기반 일상행동 인식 모델을 보존하면서 낙상 전문 분기를 추가하고, RGB 영상에서
얻은 전역 움직임과 시간 상태를 결합해 intentional lying과 낙상을 구분하는 것을 목표로 한다.

기존 기록에서 확인된 핵심 결론은 다음과 같다.

1. NTU60 native skeleton에서 DSTE Cross-Subject Top-1 85.285%, Top-5 97.234%를 기록했다.
2. 낙상 분기 학습 전후 16,487개 validation sample의 ADL logits와 frozen parameter가 완전히
   동일했다.
3. FU nested 평가에서 J1-V3는 Precision 94.268%, Recall 89.697%, F1 91.925%,
   intentional-lying FPR 5.357%를 기록했다.
4. J1은 recall 일부를 희생해 precision과 lying 오탐을 개선하는 보수적 specialist다.
5. SAFER 전처리의 topology·시간축 결함을 V2/V3로 교정했으며, V3는 seam artifact를 97.471%
   줄였다.
6. RGB global-motion 결합 G2는 SAFER에서는 Macro-F1을 높였지만 CAUCA와 Le2i에서는 G0가 더
   높아 domain-dependent하다.
7. OOPS의 C2 front-end는 coverage를 높였고, D1은 fall alert와 recovery memory가 결합된 latch
   결함을 분리했다.
8. LaDy 물리 특징과 다수 recovery 확장은 채택 기준을 통과하지 못했다.
9. VLM은 낙상 판정기가 아니라 detector가 발생시킨 사건을 설명하는 보조 interpreter로 설계한다.
10. 비식별 요구를 위해 RGB와 audio는 edge에서 처리하고 서버에는 축약된 event 중심 정보를
    전달하는 방향을 우선한다.

위 성능 수치는 과거 실험 산출물을 대조해 고정한 **기존 기록 기준 역사적 결과**다. 현재
재현 결과로 간주하지 않으며, 새 실행 결과는 reproduced result로 분리한다.

## 2. 연구 질문

이 프로젝트는 다음 질문을 단계적으로 검증한다.

1. 기존 NTU60 60-class ADL 성능을 변경하지 않고 낙상 전문 모델을 추가할 수 있는가?
2. 낙상과 intentional lying, bending, sitting, squatting 같은 hard negative를 구분할 수 있는가?
3. 다른 사람, 센서, 카메라, 해상도와 공간에서도 성능이 유지되는가?
4. body-relative skeleton에서 사라지는 영상 내 전역 이동을 RGB 특징으로 보완할 수 있는가?
5. frame/window 예측을 안정된 상태와 fall/recovery event로 변환할 수 있는가?
6. VLM과 audio model을 안전성과 비식별성을 해치지 않는 보조 계층으로 사용할 수 있는가?

## 3. 전체 시스템

~~~text
RGB video
  ├─ person detection / tracking [R03]
  ├─ COCO17 2D pose [R04, R06]
  ├─ H36M17 3D pose lifting [R05, R07]
  │    └─ V3 overlap-add: 243-frame / stride 121 / triangular
  ├─ H36M17 → proxy NTU25
  ├─ Frozen DSTE 2048-D [R01, R02]
  │    ├─ Frozen 60-class ADL head
  │    ├─ J1 skeleton fall specialist
  │    └─ G0 skeleton-only control
  ├─ image-relative Global Motion 131-D
  │    └─ G2: J1 + global-motion fusion
  └─ S0 state/recovery + D1 event decoder

확장 계층
  tracking → bounded TrackMemory
           → fall / wandering / inactivity event
           → VLM interpretation
           → deterministic output composer
~~~

### 권한 원칙

- Frozen DSTE와 ADL head는 낙상 학습에서 변경하지 않는다.
- Global Motion은 G2에만 입력하며 ADL head에는 연결하지 않는다.
- G0는 skeleton-only 기준선이자 필수 fallback이다.
- VLM과 YAMNet은 detector의 alert를 자동 취소할 권한이 없다.
- 외부 데이터 결과를 확인한 뒤 같은 데이터에 맞춰 threshold나 모델을 사후 조정하지 않는다.

## 4. Upstream과 프로젝트 기여 구분

| 구분 | 구성요소 |
| --- | --- |
| Upstream | FoundSkelModel DSTE [R01, R02], YOLOv8 [R03], ViTPose [R04], MotionAGFormer [R05] |
| 프로젝트 구현 | V3 overlap-add, H36M17→proxy NTU25 |
| 프로젝트 구현 | J0/J1 공동학습과 dataset-specific fall heads |
| 프로젝트 구현 | Global Motion 131-D, G0/G1/G2 비교 |
| 프로젝트 구현 | S0 상태·회복 모델과 D1 event decoder |
| 확장 설계 | detector-authoritative VLM, TrackMemory와 edge audio |

프로젝트 구성요소를 upstream 논문의 원래 구조로 표현하지 않으며, 논문 보고값과 프로젝트 재실행값도
구분한다.

PYSKL [R09]은 skeleton 자료 형식과 benchmark 도구의 참고 출처이며, 프로젝트 DSTE나 J1의
원 모델로 사용한 것은 아니다.

## 5. 데이터와 평가 역할

| 데이터 | 연구 역할 | 주요 평가 단위 |
| --- | --- | --- |
| NTU RGB+D 60 [R08] | native 3D ADL baseline | XSub Top-1/Top-5, logit invariance |
| FU-Kinect-Fall [R10] | fall과 intentional lying hard negative | subject-disjoint nested binary 평가 |
| SAFER Activities [R11] | dense fall/state/recovery 학습 | Macro-F1, class F1, locked test/OOD |
| CAUCAFall [R12] | RGB-lifted domain 진단 | sequence F1, AUROC/AUPRC |
| OOPS/OmniFall [R14, R15] | front-end와 event failure audit | event precision/recall/F1 |
| Le2i [R13] | 저해상도 engineering external | end-to-end 실행성과 sequence/event 성능 |
| AI-Hub/HQFSD/MCFD [R42–R44] | 향후 봉인·개발·cross-view 평가 | 사전 등록된 외부평가 |

CAUCA, OOPS와 Le2i는 반복 진단 또는 engineering 역할이므로 pristine external proof로 표현하지
않는다.

## 6. 현재 재현 진행 상태

| 단계 | 현재 상태 |
| --- | --- |
| 공식 NTU60 XSub DSTE 백본 | 구조 호환성과 최소 순전파 검증 완료 |
| NTU60 입력 | 기존 가공 bundle은 negative control로 보존, official UmURL 56,578개 생성·검증 완료 |
| ADL linear evaluation | official 입력 전체 실행에서 DSTE 동결과 60-class head 학습 확인 |
| ADL 전체 성능 재현 | epoch 150 Top-1 85.285%, Top-5 97.234%; 전체 구간 best 85.298/97.234% |
| FU-Kinect 입력 | 1,006개 원본 감사 후 993개 품질 집합과 subject-disjoint 5 folds 생성·검증 완료 |
| FU binary linear control | completed — OOF F1 94.611%, AUPRC 98.620% |
| SAFER 원본 감사 | completed — 497 sequences, 8,091,357 frames |
| SAFER 입력·linear/temporal control | V1/F1·V2 completed; 초기 V3 pilot not_selected 보존, [V3 R3 controls](2026-09-22_safer_v3_controls_shared.md) completed |
| SAFER J0/J1 재학습 | V1/V2 및 [V3 R3 개발·고정/nested·최종 학습](2026-09-22_joint_v3_reconstruction_shared.md) completed |
| Final J1와 Global G0/G1/G2 | V3 final completed; [Global 각50epoch·고정 평가](2026-09-28_global_motion_training_shared.md) completed, G2 not_selected |
| RGB 외부평가 재연결 | [고정3개 기술 연결 검사](2026-09-28_rgb_integration_shared.md) completed; 전체 외부 성능 평가는 미완료 |
| S0-A/B/C 상태 모델 | [별도 재구현·최종 분석](2026-09-29_s0bc_results_analysis_shared.md) completed; B 단독미채택, C 상태안정화 기준통과·낙상재현율 한계 |
| S0-E 규칙 비교 | [72조건·고정평가 완료](2026-09-30_s0e_reconstruction_shared.md), 채택가능후보0으로 not_selected |
| S0-F 전이보조모델 | [실험 진행](2026-09-30_s0f_reconstruction_shared.md), 최종성능미확정 |
| GT0와 Recovery 후속 | GT0 타깃검산 completed; S0-G0/G1/G2·D1·외부event 재검증 잔여 |
| VLM/YAMNet/TrackMemory 통합 | core detector 재현 후 planned |

제한 배치 정확도는 성능 결과로 사용하지 않았다. 위 NTU60 ADL 현재 재현값은 전체 학습과 16,487개 validation
평가 및 encoder 불변성 검사를 완료한 결과다.

## 7. E01 — NTU60 ADL baseline과 exact 보존 [R01, R02, R08]

### 방법

- NTU RGB+D 60 3D Cross-Subject
- 25 joints, XYZ, 최대 2명, 64-frame DSTE 입력
- DSTE와 60-class ADL head를 낙상 학습 중 frozen 처리
- 학습 전후 logits, sample order와 parameter 불변성 비교

### 기존 기록 기준 결과

| 항목 | 결과 |
| --- | ---: |
| XSub Top-1 | 85.285% |
| Top-5 | 97.234% |
| 비교 validation samples | 16,487 |
| ADL logits 최대 차이 | 0.0 |

NTU A043 낙상 sanity에서는 기존 ADL logit F1 98.188%, 별도 binary linear head F1 96.416%였다.
이는 native skeleton 표현에 낙상 정보가 존재함을 보이지만 intentional lying 안전성을 검증하지는
않는다.

## 8. E02 — FU-Kinect fall/intentional-lying [R10]

품질 기준을 통과한 993 clips와 21 subjects를 사용했다. Falling만 positive이며 walking, bending,
sitting, squatting과 lying은 negative다. 최종 평가는 subject-disjoint nested protocol이다.

현재 복구한 frozen-DSTE binary linear control의 5-fold OOF 결과는 다음과 같다.

| Precision | Recall | F1 | AUPRC | Accuracy | Lying FPR |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 93.491% | 95.758% | 94.611% | 98.620% | 98.187% | 6.548% |

OOF confusion은 TP 158, FP 11, FN 7, TN 817이다. 993개 표본은 OOF에 정확히 한 번 포함됐고
subject-disjoint split, frozen encoder와 linear-head-only 조건을 통과했다. FP 11건은 모두 lying에서
발생했고 walking, bending, sitting과 squatting의 FP는 0건이었다. 과거 trainer 원문과 전체 학습 설정이
남아 있지 않으므로 이 수치를 byte-identical 역사 재현 또는 성능 개선으로 주장하지 않으며, 구조와
split을 복구한 별도 control로 사용한다.

기존 기록의 J0/J1 결과는 다음과 같다.

| 모델 | Precision | Recall | F1 | AUPRC | Lying FPR |
| --- | ---: | ---: | ---: | ---: | ---: |
| J0 | 89.412% | 92.121% | 90.746% | 97.429% | 10.714% |
| J1 | **94.268%** | 89.697% | **91.925%** | **97.581%** | **5.357%** |

J1은 fall true positive 4개를 잃는 대신 lying false positive를 18개에서 9개로 줄였다. RBF-SVM의
F1 92.169%가 J1보다 0.243%p 높으므로 J1을 모든 지표에서 최고라고 주장하지 않는다.

## 9. E03 — SAFER 전처리와 temporal baseline [R05, R07, R11]

SAFER 497 sequences와 8,091,357 frames에서 other, fall, lie_down, lying_down 4-class를
구성했다.

현재 원본 감사에서도 in-lab 467개와 non-lab OOD 30개, 총 497 sequences·8,091,357 frames가
확인됐다. 공식 2D 좌표는 모두 finite였고 legacy 3D의 non-finite 값도 기존 기록과 같은 193
sequences·93,256 frames였다.

legacy 입력의 subject split과 window-count gate도 통과했다. 64-frame/stride-8에서 비유한 3D frame을
포함한 window를 제외한 train/validation/test/OOD 수는 599,986/104,589/216,768/71,964로 기존
기록과 일치한다. 유실된 mapper를 성능에 사후 적합하지 않고 보존 기록과 공식 전처리에 가장 가까운
구조적 호환안으로 고정했으며, 전체 993,307-window materialization 무결성도 통과했다.

이 reconstructed V1 입력의 F0A 전체 평가 결과는 다음과 같다. DSTE와 epoch-150 ADL head는 고정했고
학습이나 threshold 조정 없이 A043 argmax만 fall로 판정했다.

| Split | Fall F1 | All-action AUPRC | Fall-vs-lie AUPRC |
| --- | ---: | ---: | ---: |
| Validation | 0.000% | 9.433% | 55.184% |
| Test | 1.784% | 14.334% | 69.560% |
| OOD | 1.925% | 2.872% | 54.271% |

고정 ADL decision은 SAFER fall을 거의 검출하지 못했으나 일부 순위 신호는 남았다. 이후 Frozen DSTE
표현 위에 새 SAFER 4-class linear boundary만 학습하는 F0B를 실행했고, validation으로 후보를 고정한
뒤 test/OOD를 평가했다. 다음은 F1 temporal adapter와 corrected V2/V3다.

F0B는 유실된 역사적 optimizer를 동일하다고 가정하지 않고 별도 explicit SGD recovery control로
실행했다. 전체 validation에서 temporal-only epoch 35는 Macro-F1 65.456%, fall F1 70.920%였고,
temporal+spatial epoch 46은 69.913%, 74.345%였다. 사전 고정한 primary metric에 따라
temporal+spatial epoch 46을 선택했으며, 이 선택 전에는 F0B 선택에 test/OOD를 사용하지 않았다.
고정 head의 후속 전체 평가와 독립 검산을 완료했다.

| Split | Windows | Macro-F1 | Fall F1 | Fall-vs-lie AUPRC |
| --- | ---: | ---: | ---: | ---: |
| Test | 216,768 | 71.505% | 79.228% | 95.573% |
| OOD | 71,964 | 53.452% | 52.744% | 85.659% |

DSTE·ADL·head 불변성과 전체 평가 범위를 확인했다. 원 V1 mapper와 optimizer가 유실돼 아래
역사적 V1/V2/V3 수치와 동일한 실험 또는 인과적 개선으로 비교하지 않는다.
[후속 평가 상세](2026-09-19_f0b_postselection_shared.md)와
[다음 F1 계약](2026-09-20_f1_reconstruction_contract_shared.md)에 방법·한계를 분리했다.

F1은 문서에 확인되는 네 표현/loss 후보·dense 시간 표현·validation 선택 순서를 유지하면서
미기재 optimizer·시간층 세부를 별도 재구현 조건으로 확정했다. 네 후보를 각각 20 epochs 학습하고
validation에서 temporal+spatial/sqrt epoch17을 선택한 뒤 test/OOD 평가와 독립 검산을 완료했다.

| F1 평가 | Covered frames | Macro-F1 | Fall F1 | Fall-vs-unstable AUPRC |
| --- | ---: | ---: | ---: | ---: |
| Test | 1,743,832 | 75.278% | 79.063% | 92.575% |
| OOD | 577,392 | 56.516% | 54.271% | 94.671% |

미커버 frame은 test 21,056개·OOD 126개이며 지표에서 제외했다. 상세는
[F1 실행 결과](2026-09-20_f1_run_shared.md)와 [재현 계약](2026-09-20_f1_reconstruction_contract_shared.md)을
따른다. F0B는 window-level, F1은 unique-frame timeline 평가이므로 수치 차이를 직접적인 개선량으로
해석하지 않는다. 역사적 F1의 완전 복원이 아니다.

후속 전체에는 확인된 역사적 조건과 미기재 새 정의를 분리해 실행 전에 고정하는 별도 재구현을
적용한다. FU ZS0/native30/aligned25의 전체993clips F1은23.705/20.000/20.449%, AUPRC는
13.759/12.850/12.878%였다. 세 조건 모두 다수 오탐을 보였으며 역사적 all-non-fall 결과와는 다르다.

동일 window pipeline의 D0/D1/D2를 두 시간 처리×네 표현×5fold, 각각50epochs로 비교했다.

| 표현 | Native30 개발 OOF F1 / AUPRC | Aligned25 개발 OOF F1 / AUPRC |
| --- | ---: | ---: |
| D0 temporal | 95.522 / 98.543% | 96.341 / 99.161% |
| D0 temporal+spatial | 95.575 / 98.802% | 96.970 / 99.051% |
| D1 adapter hidden | 91.765 / 96.722% | 93.051 / 96.913% |
| D2 concat | 95.575 / 98.739% | 95.522 / 98.554% |

모델 불변성·저장 prediction·epoch 선택·OOF coverage의 독립 검산을 통과했다. Validation에서
epoch를 선택했으므로 최종 nested 또는 외부 일반화 수치가 아니다. 두 profile 모두 hidden-only가
DSTE-TS보다 낮았고 concat의 추가 이득은 확인되지 않았다. 원본과 정확히 같은 실험이나 통계적
동등성 검정으로 주장하지 않는다. [ZS 결과](2026-09-20_fu_zs_reconstruction_shared.md)와
[probe 결과](2026-09-20_fu_probe_reconstruction_shared.md)에 세부를 분리했다.

후속 [P0/P1 별도 재구현](2026-09-20_primitive_reconstruction_shared.md)에서 P0 비교와 독립 검산을
완료했다. P1도 전체 초기 동등성·10epoch 학습·독립 검산을 완료했다.

| P0 표현 | 개발 OOF F1% | AP% | fall TP/165 | lying FP/168 |
| --- | ---: | ---: | ---: | ---: |
| D0-TS | 95.575 | 98.802 | 162 | 12 |
| body primitive | 88.462 | 95.643 | 161 | 34 |
| D0-TS+primitive | 95.808 | 98.406 | 160 | 9 |

결합 mean-fold F1은0.304%p 높지만 fall TP를2개 잃어 사전 point gate를 통과하지 못했다.
Subject bootstrap95%구간[−1.425,+2.008]%p도0을 포함한다. P0는미채택이며
과거 “TP손실 없는 개선”을 이번 결과로 재사용하지 않는다.
P1은 epoch2가 선택됐으며 validation Macro-F1 75.033→75.313%, fall F1 76.789→77.419%다.
초기 출력과 원본 F1이 정확히 같고 F1 동결이 유지됨을 확인했다. 과거 best_epoch0과 다른
새 재구현 결과다. classifier도 학습했으므로 primitive만의 효과로 분리할 수 없고 test/OOD는
미평가다. 작은 validation 개선을 외부 일반화로 주장하지 않는다.
V2/V3·J0/J1·RGB/외부 평가는 아직 남아 있다.

### 확인된 V1 문제

- H36M17 출력을 COCO17 joint 순서로 해석한 topology mismatch
- 243-frame resampling 결과를 원래 timeline에 잘못 배치한 frame alignment 문제
- 일부 legacy 3D 구간의 non-finite 값

### V2/V3 교정

V2는 원본 2D pose에서 native timeline으로 다시 lifting하고 H36M17→NTU25 mapping을 명시했다.
V3는 243-frame, stride 121, triangular weighting, edge floor 0.05를 사용했다.

| 평가 | V1 | V2/V3 | 해석 |
| --- | ---: | ---: | --- |
| F0A Test F1 | 14.640% | 47.436% | topology/time 교정 후 신호 회복 |
| F0A OOD F1 | 4.834% | 22.378% | domain gap 지속 |
| F0B Test Macro-F1 | 55.584% | 55.306% | 일방적인 성능 개선은 아님 |
| F0B OOD fall F1 | 21.775% | 25.578% | +3.803%p |

V3는 seam candidate rate를 79.490%에서 2.011%로 줄였다. V2와 분류 성능은 실질적으로
동등하므로 V3는 정확도 상승 기법이 아니라 temporal geometry 교정으로 해석한다.

## 10. E04 — J0/J1 공동학습과 Final J1-V3

J0는 dataset-specific heads만 학습하고, J1은 frozen 2048-D 표현에 zero-initialized shared
residual adapter를 추가한다.

| Split | Metric | J0 | J1 | 변화 |
| --- | --- | ---: | ---: | ---: |
| SAFER Test | Macro-F1 | 70.477% | 73.105% | +2.628%p |
| SAFER Test | Fall F1 | 71.750% | 75.692% | +3.942%p |
| SAFER OOD | Macro-F1 | 53.487% | 54.512% | +1.025%p |
| SAFER OOD | Fall F1 | 46.275% | 50.297% | +4.021%p |

J1은 test/OOD fall F1을 높였지만 fall recall은 각각 약 2.1/2.2%p 낮췄다. Final J1-V3는 이후
skeleton reference로 채택됐으며 ADL exact 보존이 필수 조건이다.

## 11. E05 — LaDy·물리 특징 ablation [R26]

관절 속도, 가속도, 관절각과 회전 특징의 상보성을 평가했다. 관절 순서, sequence FPS, SO(3),
scale normalization과 결측 mask를 교정한 뒤에도 다음 결과를 얻었다.

| 데이터 | J1 F1 | LaDy F1 | 결합 F1 | 결합−J1 |
| --- | ---: | ---: | ---: | ---: |
| SAFER | 87.840% | 69.658% | 87.949% | +0.109%p |
| FU | 93.671% | 61.111% | 88.608% | -5.063%p |

FU 결합은 recall 5.128%p 감소와 lying FPR 5%p 증가를 보였다. 안전한 상보성을 확인하지 못해
not_selected로 유지한다.

## 12. E06 — RGB와 Global Motion 131-D

12개 global channel의 level·velocity·vertical-acceleration 통계와 7개 quality 특징으로 131-D를
구성했다.

| 평가 | G0 | G1 | G2 | G2−G0 |
| --- | ---: | ---: | ---: | ---: |
| SAFER Validation Macro-F1 | 67.012% | 50.607% | 69.242% | +2.230%p |
| SAFER Test Macro-F1 | 65.315% | — | 68.312% | +2.997%p |
| SAFER OOD Macro-F1 | 47.439% | — | 50.398% | +2.959%p |

G1 global-only는 skeleton을 대체하지 못했다. G2는 Macro-F1을 개선했지만 OOD fall recall은
3.909%p 감소했다. G0를 필수 fallback으로 유지하고 G2는 domain별 locked 비교 후 선택한다.

카메라 움직임 보상 후보 비교에는 Shi–Tomasi corner, Lucas–Kanade optical flow와 RANSAC의
대표 원문 [R17–R19]을 구현 근거로 사용했다. 다만 Global Motion 131-D와 최종 G2 결합은 해당
논문들의 모델이 아니라 본 프로젝트 설계다.

## 13. E07 — CAUCAFall 외부 진단 [R12]

| 평가 | 결과 | 해석 |
| --- | --- | --- |
| Native ADL head on lifted skeleton | Top-1 26.667% | RGB-lifted ADL domain gap |
| 초기 E0 ranking | AUROC 53.880%, AUPRC 50.610% | 분리력 부족 |
| corrected E1 ranking | AUROC 84.480%, AUPRC 85.917% | ranking 회복, hard F1 0% |

현재 RGB 경로의 100-sequence 진단은 다음과 같다.

| 모델 | Precision | Recall | F1 | AUROC | AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: |
| G0 | 92.453% | 98.000% | 95.146% | 97.840% | 98.482% |
| G2 | 95.833% | 92.000% | 93.878% | 96.600% | 97.251% |

CAUCA는 반복 개발 진단에 사용됐으므로 untouched external test가 아니다.

## 14. E08 — OmniFall/OOPS front-end와 failure audit [R14, R15]

818 videos, 914 fall events와 299 recovery events에서 front-end, decoder와 representation 실패를
분리했다.

| 경로 | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| 초기 B0+D0 | 76.820% | 43.873% | 55.850% |
| C2+D0 | 75.311% | 53.063% | 62.259% |
| C2+D1 | 54.893% | 67.505% | 60.550% |

C2는 quality-pass 영상을 592/818에서 685/818로 높였다. D1은 latch 귀책 miss를 146개에서
16개로 줄였지만 precision 감소로 F1은 1.710%p 낮아졌다. D1은 최고 F1 선택이 아니라 fall alert와
recovery memory를 분리한 구조 결함 수정이다.

잔여 representation failure 137건의 최대 fall posterior는 모두 0.4912 이하, 중앙값은
0.1586이었다. 다음 개선 대상은 threshold보다 supervision과 표현이다.

## 15. E09 — 상태·회복과 D1 decoder

현재 별도 재구현은 [S0-B/C 완료 분석](2026-09-29_s0bc_results_analysis_shared.md)을 따른다.
아래는 **기존 기록 기준**이며, 과거EMA0.70 결과를 이번선택인EMA0.5/연속확인 결과로 간주하지 않는다.

| 실험 | 결과 | 결정 |
| --- | --- | --- |
| S0-A state linear | Val/Test/OOD Macro-F1 62.652/59.637/34.314% | 기준선 |
| S0-C EMA 0.70 | segment-F1 +19.467%p, edit +26.970%p, switches -68.73% | 상태 안정화 채택 |
| S0-E hard rule | lying-down F1 30.990→7.094% | not_selected |
| S0-F transition auxiliary | Precision 2.535% | not_selected |

Recovery validation에서 S0-G0는 Precision 32.943%, Recall 82.422%, F1 47.072%였다. 후속
variant는 외부 recall 안전 조건을 충족하지 못했거나 untouched 평가가 없어 운영 채택하지 않았다.
Segment F1@k와 Edit score는 MS-TCN의 평가 관례 [R16]를 사용했으며, S0와 D1의 구조를 가져온
것은 아니다.

## 16. E10 — Le2i 저해상도 engineering external [R13]

평가 가능한 127개 영상은 fall 96개와 negative 31개로 구성됐다. quality pass는 109/127이었다.

| 모델 | Precision | Recall | F1 | 지위 |
| --- | ---: | ---: | ---: | --- |
| G2 | 84.946% | 82.292% | 83.598% | locked primary |
| G0 | 85.106% | 83.333% | 84.211% | post-hoc control |

G0가 0.613%p 높았지만 사후 비교이므로 primary를 대체하지 않는다. 작은 표본과 negative 분포
편향 때문에 최종 일반화 성능으로 확대 해석하지 않는다.

## 17. E11 — 외부 데이터 적격성과 봉인 평가 [R42–R46]

외부평가는 추론 전에 다음을 고정한다.

1. 공식 source, version과 license
2. fall/recovery/negative annotation 의미
3. container, frame, FPS와 annotation 시간축
4. production decoder 전수 완독
5. 입력 window 구성 가능 여부
6. GT·예측 event 평가 가능 여부와 matching tolerance

Window capable과 event scorable은 별도로 판정한다. 결과를 확인한 뒤 모델이나 threshold를
수정하면 해당 데이터는 development로 재분류한다.

## 18. E12 — VLM, TrackMemory, YAMNet

### Detector-authoritative VLM

~~~text
Detector event
  ├─ Pass A: detector 정보 없는 blind visual observation
  ├─ Pass B: observation과 detector context reconciliation
  └─ deterministic composer
       └─ VLM 실패 시 detector-only fallback
~~~

기존 LLAVIDAL [R27] standalone 진단에서는 160개 중 159개를 FALL로 예측했고 specificity가 0%였다.
따라서 LLAVIDAL을 단독 낙상 판정기로 사용하지 않는다. Detector→VLM bridge, Pass A/B와
end-to-end 통합 평가는 아직 planned 상태다.

### TrackMemory

- Short-term: 최근 pose, tracking continuity와 event evidence
- Medium-term: 최근 낙상·회복·비활동 episode
- Long-term: 명시적으로 허용된 축약 통계만 제한 보존

낙상, 배회와 장시간 비활동은 독립 detector 또는 state machine이 계산하고, VLM에는 필요한
event와 축약 context만 제공한다.

### YAMNet

YAMNet [R28]은 충격음·환경음의 보조 evidence로 사용할 수 있다. 단독 낙상 판정이나 alert 취소 권한은
부여하지 않는다. 성능 평가는 skeleton-only, audio-only와 late fusion을 동일 event split에서
비교하고, false alarm과 결측 audio 상황을 함께 측정한다.

YAMNet에는 별도 동료평가 논문이 확인되지 않아 공식 구현을 직접 인용한다. 기반 학술자료는
AudioSet [R29], large-scale audio CNN [R30]과 MobileNet [R31]이며, 실제 낙상 audio 적용의
타당성은 SAFE [R35] 등 별도 낙상 자료로 검증해야 한다.

## 19. 비식별·개인정보 보호 설계

Edge에서 skeleton만 생성한다고 해서 자동으로 완전한 익명성이 확보되지는 않는다. Skeleton에는
체형, 보행, 동작 습관과 시간 패턴이 남을 수 있기 때문이다.

이 판단은 skeleton에서 민감 속성을 억제하려는 익명화 연구 [R32]와 skeleton 기반 보행인식이
가능함을 보인 연구 [R33]에 근거한다. 저해상도 depth [R34], audio·floor vibration [R35, R36],
wearable·multimodal sensor [R37–R39]와 mmWave radar [R40, R41]는 향후 대체 또는 보조 센서
설계의 참고자료이며, 현재 통합 완료된 분기는 아니다.

### 권장 원칙

1. RGB와 audio 원본은 edge에서 처리하고 기본적으로 저장하거나 서버로 전송하지 않는다.
2. 서버 payload는 event type, model confidence, quality, coarse time과 축약 표현으로 제한한다.
3. Skeleton 좌표는 사람 중심 이동, scale·orientation 정규화와 필요 최소 joint/precision 축소를
   적용한다.
4. 영구 person ID 대신 session-scoped random ID를 사용한다.
5. Raw skeleton과 embedding에도 짧은 TTL, 전송·저장 암호화와 최소 접근권한을 적용한다.
6. Debug clip 저장은 명시적 승인, 목적 제한과 자동 삭제 정책 아래 별도 운영한다.
7. Audio는 음성 내용 보존을 피하고 edge event score 중심으로 축약한다.
8. VLM이 RGB를 필요로 하면 edge VLM을 우선하며 서버 VLM 사용은 별도 개인정보 영향평가 후 결정한다.

### 서버 권장 payload 예시

| 범주 | 전송 가능 정보 |
| --- | --- |
| Event | fall candidate, recovery candidate, inactivity |
| Timing | coarse start/end, duration |
| Model | confidence, quality, detector version |
| Context | session-scoped track ID, zone category |
| Audio | event class score와 quality |
| 제외 | raw RGB, raw audio, 얼굴, 영구 식별자 |

## 20. E13 — 비교 모델과 provenance

### FU classifier controls

Logistic regression, SVM과 random forest는 표준 통제모델 [R20–R22]이며 특정 외부 낙상 시스템을
재현한 것이 아니다.

| 모델 | F1 | AUPRC | Lying FPR |
| --- | ---: | ---: | ---: |
| Logistic regression | 88.347% | 97.246% | 18.452% |
| Linear SVM | 89.011% | 97.101% | 17.857% |
| RBF-SVM | **92.169%** | **97.727%** | 8.333% |
| Random forest | 84.039% | 94.037% | 7.738% |
| J1-V3 | 91.925% | 97.581% | **5.357%** |

### NTU60 benchmark 실행

| 모델 | 실행 상태 | XSub 결과 | 비교 경계 |
| --- | --- | --- | --- |
| PCM3 [R23] | 공식 설정 교정 후 실행 완료 | Top-1 83.933%, Top-5 96.664% | project rerun |
| MAMP [R24] | preprocessing·forward 확인 | 성능 없음 | 공식 checkpoint 미확보 |
| UmURL [R25] | 공개 multi-modal 설정 실행 | Top-1 80.979%, Top-5 96.282% | 게시 joint-only 조건과 다름 |
| Project DSTE | 공식 DSTE 기반 역사적 실행 | Top-1 85.285%, Top-5 97.234% | native NTU60 XSub |

Modality, preprocessing, checkpoint와 split이 다른 수치를 하나의 순위로 해석하지 않는다.

## 21. 채택·미채택 정리

### 채택 또는 유지

- 공식 DSTE 기반과 frozen ADL head
- V3 overlap-add temporal preprocessing
- J1 shared residual fall specialist
- G0 skeleton-only fallback
- G2 domain-dependent 후보 branch
- D1 fall alert/recovery memory 분리
- S0-C 상태 안정화
- detector-authoritative VLM 원칙

### 미채택 또는 보류

- LaDy/physics fusion
- motion-predicted bbox와 단순 camera residual 확장
- S0-E hard rule과 S0-F transition auxiliary
- recovery head의 운영 배포
- raw DSTE/J1 feature의 LLM token 직접 주입
- strict realtime, 완성된 multi-person, wandering/inactivity 성능 주장

## 22. 주장 가능한 범위와 한계

### 주장 가능

- Native NTU60 skeleton에서 역사적 ADL baseline을 재현했고 낙상 학습 중 ADL logits를 보존했다.
- FU/SAFER 내부 skeleton domain에서 J1의 precision·F1 및 lying FPR trade-off를 평가했다.
- V3가 MotionAGFormer window seam을 크게 줄였다.
- G2의 효과가 domain-dependent임을 matched 평가로 확인했다.
- C2와 D1의 front-end/decoder 영향과 잔여 representation failure를 분리했다.

### 아직 주장 불가

- 현재 weight가 소실된 과거 weight와 bit-exact하게 동일하다는 주장
- 모든 RGB 카메라 환경에 대한 일반화
- Recovery, multi-person, strict realtime과 장시간 운용의 검증 완료
- VLM 또는 YAMNet이 낙상 판정 성능을 개선했다는 주장
- Skeleton 전송만으로 완전한 익명성이 보장된다는 주장
- CAUCA, OOPS 또는 Le2i 결과를 pristine final external proof로 사용하는 주장

## 23. 다음 실행 순서

1. FU/SAFER 원본과 split provenance 확정
2. Linear controls와 J0 matched baseline
3. J1 공동학습과 locked SAFER/nested FU 평가
4. Final J1 확정 후 G0/G1/G2 비교
5. RGB front-end와 D1 event 경로 재연결
6. Development external 오류 유형 고정 뒤 sealed dataset 단일 확증 평가
7. Edge 비식별 정책 아래 VLM/YAMNet/TrackMemory 연구

세부 입수 순서와 완료 gate는 [NTU 이후 실행·데이터 확보](2026-09-03_next_steps_dataset_download_shared.md)에
고정한다.

## 24. 공유 시 인용 지침

- 모든 과거 수치에는 기존 기록 기준이라는 표현을 유지한다.
- 현재 완료, 진행 중, 계획과 미채택 상태를 섞지 않는다.
- J1을 최고 F1 모델이라고 표현하지 않는다.
- G2가 모든 domain에서 G0보다 우월하다고 표현하지 않는다.
- VLM과 YAMNet을 detector 또는 alert 취소기로 표현하지 않는다.
- Diagnostic external 결과를 final generalization proof로 표현하지 않는다.

## 25. 세부 공유 문서

- [공유 문서 인덱스](README.md)
- [실험별 전체 카탈로그](2026-09-03_experiment_catalog_shared.md)
- [파이프라인 재학습 순서](2026-09-03_pipeline_retraining_recovery_shared.md)
- [DSTE 백본과 ADL 진행](2026-09-03_dste_checkpoint_recovery_shared.md)
- [NTU 이후 실행·데이터 확보](2026-09-03_next_steps_dataset_download_shared.md)
- [VLM·TrackMemory·비식별](2026-09-03_12_vlm_trackmemory_shared.md)
- [외부 데이터 평가 프로토콜](2026-09-03_11_external_protocol_shared.md)
- [비교 모델과 provenance](2026-09-03_13_benchmarks_provenance_shared.md)

## 26. 논문·공개자료 사용 범위

현재 J1–G0 구조의 입력·변환·DSTE layer·adapter 출처는
[2026-10-03 Related Work](2026-10-03_j1_g0_related_work_shared.md)에 별도로 정리했다.
직접 사용·원 모델의 계보·관련 연구·프로젝트 정의를 구분하며 새 BibTeX를 제공한다.
아래 R01–R46은 과거 확장까지 포함하는 목록으로 보존하며 새 문서의 R번호와 혼용하지 않는다.

아래 참고문헌은 모두 프로젝트 문서에서 실제로 사용하거나 검토한 자료다. 다만 **인용했다는 것이
해당 모델을 최종 시스템에 적용했다는 뜻은 아니다.** 공동 연구 시 다음 구분을 유지한다.

| 범위 | 참고문헌 | 프로젝트에서의 의미 |
| --- | --- | --- |
| 직접 사용 | R01–R19 | backbone·pose·데이터셋·평가 관례 또는 구현한 표준 vision 연산 |
| 통제·비교·ablation | R20–R26 | baseline 재실행 또는 비교 후 미채택까지 포함 |
| 계획·설계 근거 | R27–R31 | VLM/YAMNet 확장 검토; end-to-end 효과는 아직 미검증 |
| 비식별·대체 센서 근거 | R32–R41 | 개인정보 위험 판단과 향후 sensor pilot 참고; 현재 모델에 미통합 |
| 외부평가 후보 | R42–R46 | 데이터 적격성·봉인 계획 참고; 현재 성능 결과로 사용하지 않음 |

프로젝트 자체 설계인 V3 overlap-add, H36M17→proxy NTU25, J0/J1, Global Motion 131-D,
G0/G2, S0, D1, TrackMemory와 detector-authoritative composer에는 동일 구조의 원 논문이 없다.
이를 R01–R46 논문의 기여로 귀속하지 않는다.

## 27. 전체 참고문헌

### 27.1 직접 사용한 backbone·pose·표현

- **R01.** Hongsong Wang, Wanjiang Weng, Junbo Wang, Fang Zhao, Guo-Sen Xie, Xin Geng,
  Liang Wang. [*Foundation Model for Skeleton-Based Human Action Understanding*](https://arxiv.org/abs/2508.12586).
  IEEE TPAMI, 2025 early access·2026 issue. DSTE와 pretrained skeleton representation의 주 출처.
- **R02.** Wanjiang Weng, Hongsong Wang, Junbo Wang, Lei He, Guo-Sen Xie.
  [*USDRL: Unified Skeleton-Based Dense Representation Learning with Multi-Grained Feature Decorrelation*](https://doi.org/10.1609/aaai.v39i8.32899).
  AAAI, 2025. DSTE의 선행 연구. [공식 코드](https://github.com/wengwanjiang/USDRL).
- **R03.** Ultralytics. [*YOLOv8 공식 모델 문서*](https://docs.ultralytics.com/models/yolov8/).
  Person detector 출처. 독립 동료평가 YOLOv8 논문이 아니라 공식 software documentation이다.
- **R04.** Yufei Xu et al. [*ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation*](https://proceedings.neurips.cc/paper_files/paper/2022/hash/fbb10d319d44f8c3b4720873e4177c65-Abstract-Conference.html).
  NeurIPS, 2022. COCO17 2D pose model 출처.
- **R05.** Soroush Mehraban, Vida Adeli, Babak Taati.
  [*MotionAGFormer: Enhancing 3D Human Pose Estimation with a Transformer-GCNFormer Network*](https://openaccess.thecvf.com/content/WACV2024/html/Mehraban_MotionAGFormer_Enhancing_3D_Human_Pose_Estimation_With_a_Transformer-GCNFormer_Network_WACV_2024_paper.html).
  WACV, 2024. H36M17 3D lifting 출처이며 V3 overlap-add는 프로젝트 보정이다.
- **R06.** Tsung-Yi Lin et al. [*Microsoft COCO: Common Objects in Context*](https://doi.org/10.1007/978-3-319-10602-1_48).
  ECCV, 2014. COCO keypoint 체계의 배경.
- **R07.** Catalin Ionescu, Dragos Papava, Vlad Olaru, Cristian Sminchisescu.
  [*Human3.6M: Large Scale Datasets and Predictive Methods for 3D Human Sensing in Natural Environments*](https://doi.org/10.1109/TPAMI.2013.248).
  IEEE TPAMI, 2014. H36M17 pose 체계의 배경.
- **R08.** Amir Shahroudy, Jun Liu, Tian-Tsong Ng, Gang Wang.
  [*NTU RGB+D: A Large Scale Dataset for 3D Human Activity Analysis*](https://openaccess.thecvf.com/content_cvpr_2016/html/Shahroudy_NTU_RGBD_A_CVPR_2016_paper.html).
  CVPR, 2016. NTU25, 60-class ADL과 XSub 평가의 출처.
- **R09.** Haodong Duan, Jiaqi Wang, Kai Chen, Dahua Lin.
  [*PYSKL: Towards Good Practices for Skeleton Action Recognition*](https://doi.org/10.1145/3503161.3548546).
  ACM Multimedia, 2022. Skeleton 자료 형식과 benchmark 도구의 참고 출처.

### 27.2 직접 사용한 낙상·행동 데이터와 평가 관례

- **R10.** Muzaffer Aslan, Yakup Akbulut, Abdulkadir Şengür, M. Cevdet İnce.
  [*Skeleton Based Efficient Fall Detection*](https://doi.org/10.17341/gazimmfd.369347).
  Journal of the Faculty of Engineering and Architecture of Gazi University, 2017.
  FU-Kinect-Fall 출처. [공개 데이터](https://github.com/MuzafferAslan23/Fall-Detection-Dataset).
- **R11.** Diwas Lamsal et al. [*SAFER-Activities: A Dataset for Smart Assessment of Fall Events and Routine Activities*](https://safer-activities.github.io/).
  ECCV, 2026. Frame-level fall/state 학습과 test/OOD split 출처.
  [공식 코드·데이터 설명](https://github.com/safer-activities/SAFER-Activities).
- **R12.** José Camilo Eraso Guerrero, Elena Muñoz España, Mariela Muñoz-Añasco,
  Jesús Emilio Pinto Lopera. [*Dataset for Human Fall Recognition in an Uncontrolled Environment*](https://doi.org/10.1016/j.dib.2022.108610).
  Data in Brief, 2022. CAUCAFall 출처.
- **R13.** Imen Charfi, Johel Mitéran, Julien Dubois, Mohamed Atri, Rached Tourki.
  [*Optimized Spatio-Temporal Descriptors for Real-Time Fall Detection: Comparison of Support Vector Machine and Adaboost-Based Classification*](https://doi.org/10.1117/1.JEI.22.4.041106).
  Journal of Electronic Imaging, 2013. Le2i FDD 출처.
- **R14.** Dave Epstein, Boyuan Chen, Carl Vondrick.
  [*Oops! Predicting Unintentional Action in Video*](https://openaccess.thecvf.com/content_CVPR_2020/html/Epstein_Oops_Predicting_Unintentional_Action_in_Video_CVPR_2020_paper.html).
  CVPR, 2020. OOPS 원 데이터와 intentionality annotation의 출처.
- **R15.** David Schneider et al. [*OmniFall: From Staged Through Synthetic to Wild, A Unified Multi-Domain Dataset for Robust Fall Detection*](https://arxiv.org/abs/2505.19889).
  arXiv:2505.19889, 2025. OmniFall/OOPS-Fall 계보와 annotation 출처.
  [공개 dataset card](https://huggingface.co/datasets/simplexsigil2/omnifall).
- **R16.** Yazan Abu Farha, Jürgen Gall.
  [*MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation*](https://openaccess.thecvf.com/content_CVPR_2019/html/Abu_Farha_MS-TCN_Multi-Stage_Temporal_Convolutional_Network_for_Action_Segmentation_CVPR_2019_paper.html).
  CVPR, 2019. Segment F1@k와 Edit score 관례만 사용했다.
- **R17.** Jianbo Shi, Carlo Tomasi. [*Good Features to Track*](https://ieeexplore.ieee.org/document/323794/).
  CVPR, 1994. 카메라 보상 후보의 corner selection 근거.
- **R18.** Bruce D. Lucas, Takeo Kanade.
  [*An Iterative Image Registration Technique with an Application to Stereo Vision*](https://mlanthology.org/ijcai/1981/lucas1981ijcai-iterative/).
  IJCAI, 1981. Optical-flow 기반 보상 후보의 근거.
- **R19.** Martin A. Fischler, Robert C. Bolles.
  [*Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography*](https://doi.org/10.1145/358669.358692).
  Communications of the ACM, 1981. Robust motion fitting 후보의 근거.

### 27.3 통제모델·benchmark·미채택 ablation

- **R20.** D. R. Cox. [*The Regression Analysis of Binary Sequences*](https://doi.org/10.1111/j.2517-6161.1958.tb00292.x).
  Journal of the Royal Statistical Society Series B, 1958. Logistic control의 대표 원문.
- **R21.** Corinna Cortes, Vladimir Vapnik. [*Support-Vector Networks*](https://doi.org/10.1007/BF00994018).
  Machine Learning, 1995. Linear/RBF-SVM control의 대표 원문.
- **R22.** Leo Breiman. [*Random Forests*](https://doi.org/10.1023/A:1010933404324).
  Machine Learning, 2001. Random-forest control의 대표 원문.
- **R23.** Jiahang Zhang, Lilang Lin, Jiaying Liu.
  [*Prompted Contrast with Masked Motion Modeling: Towards Versatile 3D Action Representation Learning*](https://arxiv.org/abs/2308.03975).
  ACM Multimedia, 2023. PCM3 재실행 출처.
- **R24.** Yunyao Mao, Jiajun Deng, Wengang Zhou, Yao Fang, Wanli Ouyang, Houqiang Li.
  [*Masked Motion Predictors are Strong 3D Action Representation Learners*](https://openaccess.thecvf.com/content/ICCV2023/html/Mao_Masked_Motion_Predictors_are_Strong_3D_Action_Representation_Learners_ICCV_2023_paper.html).
  ICCV, 2023. MAMP 호환성 비교 출처.
- **R25.** Shengkai Sun et al. [*Unified Multi-Modal Unsupervised Representation Learning for Skeleton-Based Action Understanding*](https://doi.org/10.1145/3581783.3612449).
  ACM Multimedia, 2023. UmURL 공개 설정 재실행과 NTU 전처리 비교 출처.
- **R26.** Haoyu Ji, Xueting Liu, Yu Gao, Wenze Huang, Zhihao Yang, Weihong Ren,
  Zhiyong Wang, Honghai Liu. [*LaDy: Lagrangian-Dynamic Informed Network for Skeleton-Based Action Segmentation via Spatial-Temporal Modulation*](https://openaccess.thecvf.com/content/CVPR2026/html/Ji_LaDy_Lagrangian-Dynamic_Informed_Network_for_Skeleton-based_Action_Segmentation_via_Spatial-Temporal_CVPR_2026_paper.html).
  CVPR, 2026. Physics ablation의 출처이며 최종 파이프라인에는 미채택됐다.

### 27.4 VLM·audio 확장 자료

- **R27.** Dominick Reilly et al. [*LLAVIDAL: A Large LAnguage VIsion Model for Daily Activities of Living*](https://openaccess.thecvf.com/content/CVPR2025/html/Reilly_LLAVIDAL_A_Large_LAnguage_VIsion_Model_for_Daily_Activities_of_CVPR_2025_paper.html).
  CVPR, 2025. 기존 standalone 진단과 향후 interpreter 설계의 출처이며 detector로 채택하지 않았다.
- **R28.** TensorFlow Model Garden. [*YAMNet 공식 구현과 모델 설명*](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet).
  521개 AudioSet class를 예측하는 MobileNet_v1 기반 공식 model documentation. 현재 통합은 planned다.
- **R29.** Jort F. Gemmeke et al. [*Audio Set: An Ontology and Human-Labeled Dataset for Audio Events*](https://research.google/pubs/audio-set-an-ontology-and-human-labeled-dataset-for-audio-events/).
  ICASSP, 2017. YAMNet 학습 ontology·corpus의 배경.
- **R30.** Shawn Hershey et al. [*CNN Architectures for Large-Scale Audio Classification*](https://arxiv.org/abs/1609.09430).
  ICASSP, 2017. AudioSet 기반 대규모 audio classifier의 배경.
- **R31.** Andrew G. Howard et al. [*MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications*](https://arxiv.org/abs/1704.04861).
  arXiv:1704.04861, 2017. YAMNet backbone 구조의 배경.

### 27.5 비식별성·audio·대체 센서 참고연구

- **R32.** Saemi Moon, Myeonghyeon Kim, Zhenyue Qin, Yang Liu, Dongwoo Kim.
  [*Anonymization for Skeleton Action Recognition*](https://doi.org/10.1609/aaai.v37i12.26754).
  AAAI, 2023. Skeleton에서도 민감 속성 억제가 별도 문제임을 뒷받침한다.
- **R33.** Chao Fan, Jingzhe Ma, Dongyang Jin, Chuanfu Shen, Shiqi Yu.
  [*SkeletonGait: Gait Recognition Using Skeleton Maps*](https://doi.org/10.1609/aaai.v38i2.27933).
  AAAI, 2024. Skeleton 기반 재식별 가능성의 근거다.
- **R34.** Edward Chou, Matthew Tan, Cherry Zou, Michelle Guo, Albert Haque, Arnold Milstein,
  Li Fei-Fei. [*Privacy-Preserving Action Recognition for Smart Hospitals Using Low-Resolution Depth Images*](https://arxiv.org/abs/1811.09950).
  ML4H Workshop at NeurIPS, 2018. 저해상도 depth의 utility/privacy trade-off 참고자료.
- **R35.** Antony García, Xinming Huang.
  [*SAFE: Sound Analysis for Fall Event Detection Using Machine Learning*](https://doi.org/10.1016/j.smhl.2024.100539).
  Smart Health, 2025. Audio-only fall 연구와 SAFE dataset의 출처.
- **R36.** Yaniv Zigel, Dima Litvak, Israel Gannot.
  [*A Method for Automatic Fall Detection of Elderly People Using Floor Vibrations and Sound—Proof of Concept on Human Mimicking Doll Falls*](https://doi.org/10.1109/TBME.2009.2030171).
  IEEE Transactions on Biomedical Engineering, 2009. Airborne audio 대안으로 floor vibration을 검토한 근거.
- **R37.** Lourdes Martínez-Villaseñor, Hiram Pönce, Jorge Brieva, Ernesto Moya-Albor,
  José Núñez-Martínez, Carlos J. Penafort-Asturiano.
  [*UP-Fall Detection Dataset: A Multimodal Approach*](https://pmc.ncbi.nlm.nih.gov/articles/PMC6539235/).
  Sensors, 2019. Wearable·ambient·vision 결합 dataset 참고자료.
- **R38.** Angela Sucerquia, José David López, J. F. Vargas-Bonilla.
  [*SisFall: A Fall and Movement Dataset*](https://pmc.ncbi.nlm.nih.gov/articles/PMC5298771/).
  Sensors, 2017. Wearable inertial fall dataset 참고자료.
- **R39.** Carla Taramasco, Miguel Piñeiro, Pablo Ormeño-Arriagada, Diego Robles Cruz,
  David Araya. [*Multimodal Dataset for Sensor Fusion in Fall Detection*](https://pmc.ncbi.nlm.nih.gov/articles/PMC11970414/).
  PeerJ, 2025. Accelerometer·thermal·LiDAR·radar fusion 후보의 근거.
- **R40.** Ann-Christine Fröhlich et al.
  [*A Millimeter-Wave MIMO Radar Network for Human Activity Recognition and Fall Detection*](https://doi.org/10.1109/RADARCONF2458775.2024.10548702).
  IEEE Radar Conference, 2024. Multistatic mmWave 후보의 근거.
- **R41.** Dylan Jayabahu, Parthipan Siva.
  [*Dataset for Real-World Human Action Detection Using FMCW mmWave Radar*](https://arxiv.org/abs/2412.17517).
  arXiv:2412.17517, 2024. 실제 주거환경 radar generalization 참고자료.

### 27.6 향후 외부평가 후보 자료

- **R42.** AI-Hub. [*낙상사고 위험동작 영상-센서 쌍 데이터(데이터셋 71641)*](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&dataSetSn=71641&topMenu=100).
  2023 구축 공식 데이터 페이지. 논문이 아니며 sealed final 후보로만 계획됐다.
- **R43.** Greet Baldewijns, Glen Debard, Gert Mertes, Bart Vanrumste, Tom Croonenborghs.
  [*Bridging the Gap between Real-Life Data and Simulated Data by Providing a Highly Realistic Fall Dataset for Evaluating Camera-Based Fall Detection Algorithms*](https://doi.org/10.1049/htl.2015.0047).
  Healthcare Technology Letters, 2016. HQFSD 출처.
- **R44.** Edouard Auvinet, Caroline Rougier, Jean Meunier, Alain St-Arnaud, Jacqueline Rousseau.
  [*Multiple Cameras Fall Data Set*](https://www.iro.umontreal.ca/~labimage/Dataset/).
  DIRO–Université de Montréal Technical Report 1350, 2010. MCFD 출처이며 동료평가 논문으로 표현하지 않는다.
- **R45.** Bogdan Kwolek, Michal Kępski.
  [*Human Fall Detection on Embedded Platform Using Depth Maps and Wireless Accelerometer*](https://doi.org/10.1016/j.cmpb.2014.09.005).
  Computer Methods and Programs in Biomedicine, 2014. URFD 출처.
- **R46.** Zhong Zhang, Christopher Conly, Vassilis Athitsos.
  [*Evaluating Depth-Based Computer Vision Methods for Fall Detection under Occlusions*](https://doi.org/10.1007/978-3-319-14364-4_19).
  ISVC, 2014. EDF/OCCU 후보와 occlusion 평가의 출처.
