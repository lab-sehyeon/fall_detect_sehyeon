# J1–G0 Related Work: 입력부터 분류기까지의 논문 계보

- 문서 ID: DOC-20261003-j1-g0-related-work-R1
- 기준일: 2026-10-03
- 문헌 정리 상태: completed
- 범위: 현재 **Frozen DSTE → J1 → G0**와 입력 생성 과정. 기존 확장 실험은 별도 분류한다.
- 부속 자료: [BibTeX](2026-10-03_j1_g0_related_work_references.bib), [구성요소–출처 CSV](2026-10-03_j1_g0_component_reference_matrix.csv).
- 연결 문서: [현행 구조](2026-10-02_j1_g0_only_shared.md), [방법론 상세](2026-09-29_paper_methodology_shared.md), [기존 전체 연구·46개 참고자료](2026-09-03_project_complete_summary_shared.md).

## 1. 먼저 구분해야 할 것

**DSTE는 기존 논문의 모델이고, J1과 G0는 그 표현 위에 구성한 프로젝트 모듈이다.**
논문과 구조가 비슷하다는 사실만으로 “그 논문을 보고 구현했다”는 역사적 사실을 확정하지 않는다.

| 관계 | 뜻 | 대표 예 |
| --- | --- | --- |
| 직접 사용 | 해당 모델·데이터·소프트웨어를 실제 사용 | ViTPose, MotionAGFormer, DSTE, SAFER |
| 원 모델의 계보·기본 연산 | 원 논문이 참고한 연구 또는 포함된 표준 연산 | DSA의 dense temporal modeling, attention, LayerNorm |
| 관련 연구 | 비교·설명에 적합하지만 동일 구현이나 직접 영향은 입증되지 않음 | Houlsby adapter, AdaptFormer, ST-GCN |
| 프로젝트 정의 | 구성·변환·학습 계약을 프로젝트에서 정함 | proxy NTU25, 중첩 구간 결합, J1, G0, 사건 판정 |
| 과거·미사용 | 이전 비교, 확장, 미채택 또는 후보 | G1/G2, S0, YAMNet, VLM, LaDy |

이 문서에는 **40개 인용 항목**을 정리했다. YOLOv8과 COCO keypoint 명세는 논문이 아닌
공식 자료다. 모두 Related Work 본문에 나열할 필요는 없으며, 기본 연산은 Methods,
데이터 출처는 Datasets 절에 배치하는 편이 명확하다. 기존 46개 자료와 이 문서의 R번호는
**서로 다른 번호 체계**다.

## 2. 현재 파이프라인과 직접 인용 대응

### 2.1 입력 계보가 하나가 아니라는 점

~~~text
SAFER 학습: 저자 배포 ViTPose-H COCO17
                                   ┐
외부 RGB: YOLOv8x → ViTPose-B COCO17 ├→ H36M17 입력 변환
                                   ┘
 → MotionAGFormer-B → H36M17 3D → proxy NTU25·정규화
 → 64프레임 / stride 8 → Frozen DSTE → max-pool·concat 2048D → J1 → G0

FU 학습: Kinect V1의 native 20관절 → 프로젝트 proxy NTU25·정렬
 → Frozen DSTE → clip 내 window 특징 평균 → J1 → FU binary head
~~~

FU는 RGB pose/lifting을 거치지 않는다. SAFER·FU의 두 head는 **J1 공동학습 설명**에 필요하지만,
현재 추론 경로에는 FU head 및 공동학습 당시 SAFER head를 넣지 않고 별도 G0만 사용한다.
SAFER의 공식 pose 생성 계보는 [저자 저장소](https://github.com/safer-activities/SAFER-Activities)에도 명시되어 있다.

### 2.2 단계별 출처표

| 단계 | 현재 적용 | 인용할 자료 | 직접 가져온 것과 자체 정의의 경계 |
| --- | --- | --- | --- |
| RGB 시간축 | actual PTS에 따른 past-only 25 fps 표본 선택 | 프로젝트 방법 | 특정 논문의 알고리즘을 재현했다고 주장하지 않음 |
| 사람 검출 | YOLOv8x person detection | [R03](#r03) | 검출 모델은 직접 사용; confidence fallback·선택 규칙은 프로젝트 정의 |
| 주 인물 연결 | IoU와 confidence의 가중 점수, 이전 유효 track 유지 | 프로젝트 방법 | SORT·DeepSORT·ByteTrack 구현으로 부르면 안 됨 |
| RGB→2D 관절 | 외부 RGB는 ViTPose-B, SAFER 배포 pose는 ViTPose-H | [R04](#r04), [R28](#r28) | 같은 논문 계열이지만 모델 크기·입력 생성 계보가 다름 |
| 2D 좌표 처리 | top-down affine·heatmap decoding, UDP, flip test | [R05](#r05), [R04](#r04) | ViTPose 공식 설정에 포함된 UDP 계열 처리 |
| 17관절 형식 | COCO17 입력과 H36M17 lifting 형식 | [R07](#r07), [R08](#r08), [R39](#r39) | 데이터셋 출처와 구체적인 좌표 매핑 규칙을 구분 |
| 2D→3D | H36M 가중치의 MotionAGFormer-B | [R06](#r06), [R08](#r08) | transformer/GCNFormer 결합 lifting 직접 사용 |
| 긴 시퀀스 결합 | 243프레임, stride 121, triangular overlap-add, floor 0.05 | R06 + 프로젝트 방법 | 모델과 별개로 이 결합 규칙 전체를 원 논문 제안으로 쓰지 않음 |
| H36M17→NTU25 | 관절 재배치·가상 관절·말단 복제 | [R08](#r08), [R09](#r09) + 프로젝트 방법 | 실제 Kinect25 측정이 아니라 **proxy** |
| 공간 정규화 | torso median→0.5, SpineMid 중심화, 첫 유효 어깨 방향 정렬 | [R10](#r10) + 프로젝트 방법 | UmURL 계보와 현행 SAFER/RGB scale·yaw 계약을 구분 |
| FU native 입력 | Kinect20→proxy25, 중심·어깨 정렬, native 30 fps | [R29](#r29), [R10](#r10) | FU에는 SAFER/RGB의 torso 0.5 scaling 미적용 |
| skeleton encoding | temporal·spatial 두 stream의 DSTE | [R01](#r01), [R02](#r02) | 공식 사전학습 백본 직접 사용·동결 |
| 표현 집약 | 각 stream max-pool 후 concat 2048D | [R01](#r01), [R02](#r02) | CLS token이나 두 stream 평균이 아님 |
| J1 | pooled 특징 뒤 zero-up residual bottleneck | [R21](#r21)–[R23](#r23) + 프로젝트 방법 | 관련 adapter 연구이지 동일 구현·직접 영향 이력의 입증은 아님 |
| 공동학습 | 공유 J1 + SAFER 4-class/FU binary head | [R28](#r28), [R29](#r29), 배경 [R23](#r23) | supervision 단위를 보존한 프로젝트 학습 구성 |
| G0 | 고정 J1 특징→4-class linear | R01/R02의 linear evaluation 배경 + 프로젝트 방법 | G0라는 독립 제안 논문은 없음; 별도로 학습한 classifier |
| 낙상 사건 | fall 시작점·허용 시간범위 기반 matching | 프로젝트 평가 방법 | OmniFall annotation 사용과 공식 지표·decoder 재현은 별개 |

## 3. 입력과 변환의 논문 근거

### 3.1 검출과 2D pose

YOLOv8은 사람 bounding box를 제공하고, ViTPose는 해당 사람의 2D 관절을 추정한다.
두 모델의 역할을 분리해서 적는다. YOLOv8-Pose로 관절까지 추정하는 구조가 아니다.
핵심 인용은 ViTPose이고, ViT 자체는 배경을 확장할 때 인용한다.
[R03](#r03), [R04](#r04), [R14](#r14)

UDP는 좌표 변환과 heatmap 해석에서 발생하는 편향을 줄이는 처리의 출처다.
현재 공식 설정의 use_udp=True와 flip test를 설명할 때 인용한다.
이것을 **COCO17→H36M17 관절 이름 변환의 논문**으로 설명하면 안 된다.
[공식 ViTPose 설정](https://github.com/ViTAE-Transformer/ViTPose/blob/main/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py), [R05](#r05)

### 3.2 COCO→H36M→proxy NTU

- **COCO17→H36M17 입력:** 관절을 재배치하고, 양 hip 평균으로 pelvis,
  양 shoulder 평균으로 shoulder midpoint 등을 만든다. 화면 정규화는
  \((2x/W-1,\;2y/W-H/W)\)다.
- **현재 입력의 제한:** XY는 H36M 순서로 바뀌지만 confidence는 COCO 순서로 남는다.
  모든 채널을 관절 의미까지 일치시켰다고 쓰면 안 된다. 이 문헌 정리가 입력을 교정한 것도 아니다.
- **3D lifting:** MotionAGFormer는 전역 관계의 transformer와 국소 관계의 GCNFormer를 결합한다.
  이는 **3D pose 추정기**이며 낙상 분류용 DSTE와 별개다. [R06](#r06)
- **H36M17→proxy NTU25:** 입력에 없는 hand-tip/thumb/foot 말단을 wrist/ankle 등으로 복제한다.
  관절 수가 늘어도 새로운 3D 관측 정보가 생기는 것은 아니다.
- **정규화:** scale·중심화·어깨 정렬은 모델 입력을 맞추기 위한 처리다.
  H36M/NTU 논문은 관절 체계 출처이며 프로젝트 변환 전체를 제안한 논문이 아니다.
  [R08](#r08), [R09](#r09)

정확한 관절·confidence 대응과 normalization 조건은
[방법론 상세 §12](2026-09-29_paper_methodology_shared.md#reproducibility)에 연결한다.
Proxy 관절을 원 센서의 생체역학적 측정과 동일하다고 주장하지 않는다.

### 3.3 시간 처리와 인과성

243프레임 lifting과 64프레임 분류창은 서로 다른 단계다. 분류창은 stride 8로 이동한다.
SAFER 학습 label은 창 중앙 offset 32에 대응한다. 마지막 입력 offset 63은 중앙보다
31프레임, 25 fps에서 1.24초 뒤다. 긴 lifting 문맥과 영상 전체 품질 검사도 포함된다.
따라서 프레임 선택이 past-only라는 이유만으로 **전체 모델을 causal 실시간 모델이라고
부를 수 없다**. Label 시각, 출력 가능 시점, event matching 단위를 별도로 명시해야 한다.

## 4. DSTE 레이어의 논문 계보

### 4.1 가장 직접적인 두 논문

1. **USDRL, AAAI 2025 [R02](#r02):** DSTE와 내부 DSA·CA, multi-grained feature
   decorrelation을 제안한 직접 출처.
2. **Foundation Model, 2025 arXiv·TPAMI accepted [R01](#r01):** USDRL의 확장 연구.
   Multi-perspective consistency training과 더 넓은 downstream 과제를 다룬다.

두 논문을 함께 인용하되 DSTE를 이 프로젝트의 신규 백본이라고 쓰지 않는다.
원 모델의 자기지도학습을 소개하는 것과 **현재 프로젝트가 그 사전학습을 재실행했다는 주장**도
구별한다. 낙상 학습에서는 공개 사전학습 DSTE를 고정했다.

### 4.2 레이어별 대응

| 구성 | 현행 연산·차원 | 직접 출처 | 더 이전 계보와 주의 |
| --- | --- | --- | --- |
| 입력 재배열 | \(B×3×64×25×2\)→temporal \(B×64×150\), spatial \(B×50×192\) | R01/R02 | 단일 인물 입력의 두 번째 사람 slot은 0; spatial token은 50 |
| stream embedding | 각 stream Linear→LayerNorm→ReLU→Linear, hidden 1024 | R01/R02 | LayerNorm 기본 연산 R15 |
| positional encoding | temporal sinusoidal, spatial learned embedding | R01/R02 | 기본 배경 R13; 둘 모두 sinusoidal은 아님 |
| 두 stream | 시간 변화와 공간 구성을 별도 처리 | R01/R02 | Foundation §3이 Wang & Wang 2017 R11을 명시적으로 인용 |
| Dense Shift | token 축 두 선형층·ReLU·residual, gap 4 위치는 원 입력 유지 | R02 | USDRL DSA 절이 Xing et al. 2023 R12의 dense temporal modeling 영향을 명시 |
| DSA attention | 원 입력/shift 입력에 공유 attention·FFN 적용 후 평균 | R01/R02 | 기본 self-attention R13; 별도 sparse-window attention 구현 주장 금지 |
| CA | Conv1d→ReLU 보정 후 self-attention·FFN | R01/R02 | 현행 kernel 1, groups 기본값 1. **Depthwise나 넓은 temporal kernel이 아님** |
| DST layer 결합 | \(0.5\,CA(x)+0.5\,DSA(x)\) | R01/R02 | 각 stream 두 layer, attention head 수 1 |
| FFN·정규화 | GELU MLP, LayerNorm, residual, DropPath | R01/R02 | 기본 연산 배경 R15–R18; 전체 구성 출처는 DSTE |
| pooling | \(H_T:64×1024,\;H_S:50×1024\) 각각 token-wise max | R01/R02 | concat 2048D; 평균 pooling·CLS 방식과 구별 |
| 사전학습 목적 | MG-FD와 temporal/spatial/instance projectors | R01/R02 | Barlow Twins R40·VICReg R20를 원 논문이 명시; 현재 추론 graph 및 J1 loss에는 없음 |

**계보를 읽는 법:** R11의 recurrent network나 R12의 GgHM 전체를 프로젝트가 사용한 것은 아니다.
DSTE 원 저자가 선행연구로 명시한 관계다. 가장 직접적인 인용은 R01/R02로 두고,
원천 계보까지 설명하는 문단에서 R11/R12를 추가한다.

### 4.3 논문 수식과 구현의 경계

현행 DSA는 gap 위치를 원 입력으로 복원하고 두 branch를 평균한다. 논문의 mask 설명이나
단순 합산 수식을 그대로 현행 구현 수식으로 옮기지 않는다. CA의 마지막 FFN residual
처리도 흔한 Transformer 도식과 세부적으로 같지 않다.

특히 kernel 1의 CA convolution 자체는 여러 인접 frame을 직접 혼합하지 않는다.
시간·관절 token 간 결합은 뒤의 attention과 DSA token-mixing에서 이루어진다.
“CA가 긴 시계열 convolution으로 local motion을 추출한다”는 설명은 현행 설정에 맞지 않는다.

## 5. J1 어댑터의 실제 구조와 관련 논문

### 5.1 실제 J1

DSTE의 집약 특징 \(z∈\mathbb{R}^{2048}\)에 다음 보정을 적용한다.

\[
z' = z + W_{\mathrm{up}}\,
\mathrm{Dropout}_{0.1}\!\left(
\mathrm{GELU}\!\left(W_{\mathrm{down}}\mathrm{LN}(z)+b_{\mathrm{down}}\right)
\right)+b_{\mathrm{up}}.
\]

- \(W_{\mathrm{down}}:2048→256,\;W_{\mathrm{up}}:256→2048\).
- Up weight와 up bias를 0으로 초기화하므로 초기에는 \(z'=z\).
- LayerNorm을 포함한 adapter 학습 파라미터는 **1,054,976개**.
- DSTE 각 layer **안이 아니라 최종 pooled 특징 뒤에 한 번** 적용한다.
- DSTE는 고정하고 J1 공동학습에서 adapter와 데이터별 head를 학습한다.
- LoRA, prompt tuning, temporal convolution adapter가 아니다.

### 5.2 어댑터 선행연구 비교

| 연구 | 원 연구 핵심 | J1과 공통점 | 차이·인용 경계 |
| --- | --- | --- | --- |
| Houlsby et al., ICML 2019 [R21](#r21) | Transformer 내부 serial bottleneck adapter, backbone 고정 | 차원 축소/복원·residual·효율적 전이 | 내부 여러 위치, near-identity 초기화. J1의 외부 single adapter·정확한 zero-up과 동일하지 않음 |
| AdaptFormer, NeurIPS 2022 [R22](#r22) | ViT FFN의 병렬 bottleneck, up projection zero-init | frozen backbone, zero-up, residual 병목 | 내부 AdaptMLP·ReLU·scale factor. J1은 pooled 특징 뒤 LN/GELU/dropout |
| VMT-Adapter, AAAI 2024 [R23](#r23) | 공유·작업별 지식을 구분한 다중 작업 전이 | 여러 supervision으로 공유 표현 조정 | VMT의 task-specific knowledge extraction이나 Lite 구성 재현 근거 없음 |
| ResNet, CVPR 2016 [R17](#r17) | identity 경로와 residual 학습 | \(z+\Delta z\) | adapter 전용 논문이 아니며 ResNet backbone도 미사용 |

**직접 영향에 대해 말할 수 있는 범위:** 기존 방법 문서는 VMT-Adapter를 관련 선행연구로
인용했다. 보존 설계 원장은 “DSTE 정보를 보존하면서 공유 residual과 데이터별 head를 학습”하는
목적과 zero-init을 명시한다. 그러나 특정 adapter 논문의 코드나 구조를 그대로 재현했다는
근거는 확인되지 않는다. Houlsby/AdaptFormer는 이번 정리에서 구조 비교와 학술적 위치를
설명하는 선행연구로 활용하며, 과거 구현자의 착안 경위를 새로 확정하지 않는다.

권장 표현:

> 동결된 skeleton encoder의 pooled representation을 낙상 도메인에 적응시키기 위해
> zero-initialized residual bottleneck adapter를 구성하였다. 이는 parameter-efficient
> adapter 연구와 관련되지만, Transformer 내부가 아니라 최종 표현에 적용한다.

### 5.3 공동학습과 현재 추론의 구별

J0는 고정 DSTE 특징 위 데이터별 head 기준선이며 J1은 이를 바탕으로 공유 adapter를 학습한다.
SAFER는 window center의 4-class label, FU는 clip binary label을 사용한다.
FU에서는 window 특징을 먼저 clip별 평균한 뒤 J1에 전달한다.
FU clip label을 모든 frame의 정답처럼 복제했다고 설명하지 않는다.

J0/J1 optimizer는 AdamW [R24](#r24)다. 여러 데이터셋을 사용했다는 이유만으로
domain-adversarial learning, contrastive alignment나 별도 domain alignment loss를
사용했다고 주장할 수 없다. 두 데이터의 label 정의·frame rate·정규화도 같지 않다.

## 6. G0와 사건 평가의 출처

G0는 \(z'\)에 적용하는 **2048→4 선형층**, bias 포함 **8,196개 파라미터**다.
Frozen representation의 linear evaluation은 R01/R02에서도 사용하는 계열이지만,
현재 4-class 구성과 G0라는 이름은 프로젝트 정의다.

이미 학습된 J1을 고정하고 SAFER로 새 G0를 학습했다. 50 epoch 중 validation 선택
epoch 5를 사용하며 이 단계 optimizer는 SGD다. **J1의 AdamW 설정을 G0로 복사하지 않는다.**

원고에서는 다음을 분리한다.

- DSTE 사전학습: upstream 연구.
- SAFER/FU J1 학습: 프로젝트의 공동 supervision 적응.
- 고정 J1 위 G0 학습: 최종 분류 경계 학습.
- Event 생성·matching: 프로젝트 평가 계약.
- OmniFall/Le2i 등 데이터·annotation: 데이터셋 출처.

현재 출력은 other/fall/lie_down/lying_down 네 ID다. SAFER의 두 lying 이름을 일반적인
“동작/정지 상태”로 임의 번역하여 교체하지 않는다. 명칭·의미의 한계는 방법론 상세를 따른다.
평가 지표 F1과 **과거 F1 temporal adapter 실험명**도 구별한다. 그 temporal adapter는
현행 J1–G0 구조가 아니다.

## 7. Related Work에서 비교할 연구군

### 7.1 Skeleton 표현 학습

| 연구 | 원고에서 쓰는 이유 | 현재 방법과의 차이 |
| --- | --- | --- |
| ST-GCN [R25](#r25) | 관절·시간 graph 인식의 대표 배경 | DSTE는 ST-GCN 백본이 아님 |
| PoseConv3D [R26](#r26) | 좌표 graph 대신 heatmap volume을 쓰는 표현 | 현재는 lifted 3D 좌표 입력; PoseConv3D는 별도 비교 계열 |
| PYSKL [R27](#r27) | benchmark·자료 형식·비교 구현 | DSTE/J1 구조의 원 논문이 아님 |
| UmURL [R10](#r10) | 자기지도 표현·공식 전처리 계보 | joint-only DSTE와 multimodal UmURL 점수의 설정을 구분 |
| USDRL/FoundSkel [R02](#r02), [R01](#r01) | dense 표현과 전이의 직접 기반 | 본 연구는 백본 발명이 아니라 낙상 적응·외부평가 |

비교 모델 언급은 이 문헌 조사에서 성능 우열을 검증했다는 뜻이 아니다.
결과는 대응 평가 문서의 완료 상태와 조건을 따른다.

### 7.2 직접 관련된 낙상 연구

| 논문 | 입력·방법 | 연결 논점 | 직접 수치 비교의 제한 |
| --- | --- | --- | --- |
| Kim et al. 2024 [R33](#r33) | pose/ST-GCN·시간 단위 decision fusion | 시간 정보를 활용한 낙상 판단 | HAR-UP 학습·평가 조건과 집약 단위가 다름 |
| Raza et al. 2025 [R34](#r34) | pose 추정기와 ML/RNN/GCN/Transformer 비교 | front-end와 classifier 조합 | URFD·UP-Fall·Le2i 70/30 및 10-fold CV; target 미학습 외부평가와 다름. Subject/video grouping 별도 확인 필요 |
| Zobi et al. 2025 [R35](#r35) | 3D skeleton, GCN, 가림/회전 증강·추론 집약 | lifting·시점·가림·매핑 | 원 영상 분할 후 증강하되 Le2i로 학습; 일부 설정은 NTU 포함 |
| BenAbdennour et al. 2026 [R36](#r36) | YOLOv11-Pose + temporal Transformer | target 미세조정 없는 전이 | Le2i 109영상에서 835 clip 선별, F1 91.65%; boundary/post-fall 제외 규칙이 있어 전체 연속영상 event F1과 다름 |
| SAFER-Activities [R28](#r28) | 연속 행동·낙상 데이터와 pose/RGB 기준선 | 일상행동·낙상·lying 구별 | 데이터 논문의 기준선과 DSTE/J1/G0는 다른 모델 |
| OmniFall v3 [R32](#r32) | 통합 도메인·16-class annotation | staged→wild 일반화·평가 조건 | v1 10-class와 v3를 혼합하지 않음; annotation 사용과 공식 benchmark 재현은 별개 |

따라서 기존 논문의 높은 F1과 비교할 때 video/subject split, target 학습 포함 여부,
window와 stride, 제외 규칙, clip/frame/event 단위, 품질 거부 처리를 함께 대조해야 한다.
[기존 비교 검토](2026-10-02_paper_evidence_review_shared.md)에도 이 경계를 유지한다.

### 7.3 데이터 인용의 위치

- **Pretraining/Input:** NTU [R09](#r09), H36M [R08](#r08), COCO [R07](#r07)/명세 [R39](#r39).
  이 프로젝트에서 해당 원 데이터 전부로 모델을 재학습했다는 뜻이 아니다.
- **J1/G0 학습:** SAFER [R28](#r28), FU [R29](#r29). G0는 SAFER 학습이다.
- **외부평가:** Le2i [R30](#r30), URFD [R31](#r31), CAUCAFall [R37](#r37).
  통합 annotation을 쓴 경우 OmniFall [R32](#r32)도 인용한다.
- **이전 RGB/OOPS 실험:** OOPS [R38](#r38)와 OmniFall 재주석의 계보를 구분한다.
  취득·진단 실행을 학습 사용이나 일반화 성능 검증 완료로 바꾸지 않는다.

## 8. 원고에 옮겨 쓸 수 있는 Related Work 초안

### 8.1 Pose 기반 낙상 인식

관절 표현은 영상의 외형과 행동 분류를 분리하는 수단으로 사용되어 왔다. Kim 등은
skeleton과 시간 단위 의사결정 결합을 연구했고 [R33](#r33), Raza 등은 pose 추정기와
분류기의 여러 조합을 비교하였다 [R34](#r34). 최근 연구는 3D skeleton의 가림·시점 변화에
대한 증강과 추론 집약 [R35](#r35), 또는 2D pose와 경량 Transformer의 결합 [R36](#r36)으로
확장되었다. 그러나 학습 데이터와 clip 구성, 사건 평가 여부가 달라 보고된 F1을 직접
순위화하기 어렵다. 본 연구는 입력 생성 과정과 평가 단위를 명시하면서 사전학습 skeleton
표현의 낙상 적응 및 대상 데이터셋 미학습 조건의 전이를 검토한다.

### 8.2 RGB로부터의 skeleton 구성

본 연구는 사람 검출과 pose 추정, 3D lifting을 순차적으로 수행한다. 사람 검출에는
Ultralytics YOLOv8, 2D 관절 추정에는 ViTPose를 사용한다 [R03](#r03), [R04](#r04).
Pose 추정에는 UDP 계열 좌표 처리가 포함된다 [R05](#r05). MotionAGFormer는 transformer와
GCNFormer를 결합하여 2D pose에서 3D 구조를 추정하며 [R06](#r06), 본 연구는 그 H36M 출력을
skeleton backbone의 입력 체계로 변환한다. 이때 H36M17→NTU25 변환은 관측되지 않은 말단을
복제하는 proxy 표현이므로 원 센서의 25관절 측정과 동일한 정보로 해석하지 않는다.

### 8.3 사전학습 skeleton 표현

Skeleton 인식은 ST-GCN의 관절 graph [R25](#r25), PoseConv3D의 heatmap 표현 [R26](#r26),
자기지도 표현 학습 [R10](#r10) 등으로 발전해 왔다. USDRL은 공간과 시간의 dense
representation을 생성하는 DSTE를 제안했으며 DSA와 CA를 결합한다 [R02](#r02).
Foundation Model 연구는 이를 다양한 action understanding 과제로 확장한다 [R01](#r01).
본 연구는 DSTE를 새로 제안하는 대신 공개 사전학습 encoder를 고정하고 집약 표현 위에서
낙상 관련 supervision을 학습한다.

### 8.4 파라미터 효율적 적응

동결 backbone에 작은 모듈을 추가하는 adapter는 전체 fine-tuning과 구별되는 전이 방식이다.
Houlsby 등은 bottleneck adapter를, AdaptFormer는 시각 Transformer의 병렬 adapter를
연구하였다 [R21](#r21), [R22](#r22). VMT-Adapter는 여러 작업의 공유·작업별 정보를 활용하는
방향을 제시한다 [R23](#r23). 본 연구의 J1은 이러한 계열과 관련된 zero-initialized residual
bottleneck이지만 encoder 각 layer가 아니라 pooled skeleton 특징에 한 번 적용한다.
SAFER/FU의 supervision 단위를 유지하며 공유 adapter를 학습하고, 최종적으로 고정 J1 표현
위에 별도의 G0 선형 분류기를 학습한다.

### 8.5 데이터셋 간 평가

SAFER와 FU는 본 연구의 낙상 적응 데이터 출처다 [R28](#r28), [R29](#r29).
Le2i, URFD와 CAUCAFall은 외부평가 조건을 설명할 때 인용한다 [R30](#r30), [R31](#r31),
[R37](#r37). OmniFall은 여러 수집 조건을 통합하고 staged·실제 환경 사이의 일반화를
검토하는 맥락을 제공한다 [R32](#r32). 본 연구는 데이터셋 이름만으로 동일 평가라고 보지 않고,
학습 포함 여부, 영상 분할, 표본 구성과 사건 판정 규칙을 함께 보고한다.

## 9. 기존 46개 자료의 보존과 현행 구조의 관계

아래 “이전 R번호”는 [기존 통합본 §27](2026-09-03_project_complete_summary_shared.md#27-전체-참고문헌)의
번호다. 전체 46개의 제목·원문 링크는 그 문서에 보존하며 다음처럼 재분류한다.

| 이전 R번호 | 기존 자료 | 현재 원고에서의 위치 |
| --- | --- | --- |
| 01–09 | FoundSkel, USDRL, YOLOv8, ViTPose, MotionAGFormer, COCO, H36M, NTU, PYSKL | 현행 입력·백본·benchmark 출처 |
| 10–15 | FU, SAFER, CAUCA, Le2i, OOPS, OmniFall | 학습·평가·이전 입력 실험 역할 구분 |
| 16 | MS-TCN | 과거 segment F1/Edit 관례; J1/G0 구조 아님 |
| 17–19 | Shi–Tomasi, Lucas–Kanade, RANSAC | 과거 global/camera-motion; 활성 G0 입력에 없음 |
| 20–22 | Logistic regression, SVM, Random Forest | 기존 FU classifier controls를 보고할 때 |
| 23–25 | PCM3, MAMP, UmURL | 비교 학습 계보; UmURL은 전처리 출처도 유지 |
| 26 | LaDy | 과거 physics ablation, 현재 미채택 |
| 27–31 | LLAVIDAL, YAMNet, AudioSet, audio CNN, MobileNets | VLM/audio 확장; 현행 구성 아님 |
| 32–34 | Skeleton anonymization, SkeletonGait, low-resolution depth privacy | privacy 논의 시 별도 인용; skeleton 익명성 보장 주장 금지 |
| 35–41 | SAFE, floor vibration/sound, UP-Fall, SisFall, multimodal sensors, radar 연구 2개 | 대체 센서·확장 배경; 현재 classifier 학습 방법 아님 |
| 42–44 | AI-Hub, HQFSD, MCFD | 후보/기존 검토; 완료 범위는 평가 문서에 따름 |
| 45 | URFD | 현행 외부평가 데이터 출처 |
| 46 | EDF/OCCU | 가림 평가 후보; 현행 백본 출처 아님 |

기존 Bib의 FU 저자명 Yakup Akbulut은 출판사 원문에 따라 **Yaman Akbulut**으로 바로잡았다.
새 Bib은 OmniFall을 v3(2026-07-01)로 고정한다. Foundation과 SAFER는 확인한 arXiv 버전을
인용하며 미확인 출판 권호나 proceedings 정보를 임의 추가하지 않았다.

## 10. 제출 전 주장 체크

- DSTE/DSA/CA의 원 기여는 R01/R02로 귀속한다.
- J1을 “Houlsby/AdaptFormer/VMT 그대로 구현”이라고 쓰지 않는다.
- J1을 backbone 내부 adapter라고 하거나 DSTE 전체를 낙상 데이터로 fine-tuning했다고 쓰지 않는다.
- MotionAGFormer·DSTE·J1의 역할과 학습 범위를 구분한다.
- Proxy NTU25, confidence 대응, FU와 SAFER 전처리 차이를 Methods에 남긴다.
- Long-context·whole-video 처리에서 causal 실시간/edge 성능을 입증했다고 쓰지 않는다.
- 기본 연산 인용을 독립적인 신규 모델 기여로 바꾸지 않는다.
- Target 학습 포함 여부와 clip/frame/event가 다른 논문과 직접 우월성을 비교하지 않는다.
- 새 모델·데이터 변경이나 성능 실험은 이 문헌 정리의 결과에 포함하지 않는다.

## 11. 전체 인용 목록

아래는 이 문서의 R번호이며 BibTeX key와 사용 위치를 함께 둔다.

<a id="r01"></a>

### R01. Foundation Model for Skeleton-Based Human Action Understanding

Wang, Hongsong; Weng, Wanjiang; Wang, Junbo; Zhao, Fang; Xie, Guo-Sen; Geng, Xin; Wang, Liang. 2025.
arXiv preprint arXiv:2508.12586. [원 출처](https://arxiv.org/abs/2508.12586).

- 인용 key: `wang2025foundation`.
- 이 연구와의 관계: 직접 백본·공식 사전학습 출처.

<a id="r02"></a>

### R02. USDRL: Unified Skeleton-Based Dense Representation Learning with Multi-Grained Feature Decorrelation

Weng, Wanjiang; Wang, Hongsong; Wang, Junbo; He, Lei; Xie, Guo-Sen. 2025.
Proceedings of the AAAI Conference on Artificial Intelligence; DOI: 10.1609/aaai.v39i8.32899. [원 출처](https://ojs.aaai.org/index.php/AAAI/article/view/32899).

- 인용 key: `weng2025usdrl`.
- 이 연구와의 관계: DSTE·DSA·CA의 직접 원 연구.

<a id="r03"></a>

### R03. Ultralytics YOLOv8 Documentation

Ultralytics. 발행연도 미지정.
공식 소프트웨어·데이터 문서. [원 출처](https://docs.ultralytics.com/models/yolov8/).

- 인용 key: `ultralyticsyolov8`.
- 이 연구와의 관계: 사람 검출 소프트웨어; 독립 논문 아님.

<a id="r04"></a>

### R04. ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation

Xu, Yufei; Zhang, Jing; Zhang, Qiming; Tao, Dacheng. 2022.
Advances in Neural Information Processing Systems. [원 출처](https://proceedings.neurips.cc/paper_files/paper/2022/hash/fbb10d319d44f8c3b4720873e4177c65-Abstract-Conference.html).

- 인용 key: `xu2022vitpose`.
- 이 연구와의 관계: 2D pose 직접 사용.

<a id="r05"></a>

### R05. The Devil Is in the Details: Delving Into Unbiased Data Processing for Human Pose Estimation

Huang, Junjie; Zhu, Zheng; Guo, Feng; Huang, Guan. 2020.
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. [원 출처](https://openaccess.thecvf.com/content_CVPR_2020/html/Huang_The_Devil_Is_in_the_Details_Delving_Into_Unbiased_Data_CVPR_2020_paper.html).

- 인용 key: `huang2020udp`.
- 이 연구와의 관계: ViTPose 설정에 포함된 UDP.

<a id="r06"></a>

### R06. MotionAGFormer: Enhancing 3D Human Pose Estimation With a Transformer-GCNFormer Network

Mehraban, Soroush; Adeli, Vida; Taati, Babak. 2024.
Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. [원 출처](https://openaccess.thecvf.com/content/WACV2024/html/Mehraban_MotionAGFormer_Enhancing_3D_Human_Pose_Estimation_With_a_Transformer-GCNFormer_Network_WACV_2024_paper.html).

- 인용 key: `mehraban2024motionagformer`.
- 이 연구와의 관계: 2D→3D lifting 직접 사용.

<a id="r07"></a>

### R07. Microsoft COCO: Common Objects in Context

Lin, Tsung-Yi; Maire, Michael; Belongie, Serge; Hays, James; Perona, Pietro; Ramanan, Deva; Dollár, Piotr; Zitnick, C. Lawrence. 2014.
Computer Vision -- ECCV 2014; DOI: 10.1007/978-3-319-10602-1_48. [원 출처](https://doi.org/10.1007/978-3-319-10602-1_48).

- 인용 key: `lin2014coco`.
- 이 연구와의 관계: COCO 데이터 계보.

<a id="r08"></a>

### R08. Human3.6M: Large Scale Datasets and Predictive Methods for 3D Human Sensing in Natural Environments

Ionescu, Catalin; Papava, Dragos; Olaru, Vlad; Sminchisescu, Cristian. 2014.
IEEE Transactions on Pattern Analysis and Machine Intelligence; DOI: 10.1109/TPAMI.2013.248. [원 출처](https://vision.imar.ro/human3.6m/description.php).

- 인용 key: `ionescu2014human36m`.
- 이 연구와의 관계: lifting 사전학습 데이터·관절 체계 배경.

<a id="r09"></a>

### R09. NTU RGB+D: A Large Scale Dataset for 3D Human Activity Analysis

Shahroudy, Amir; Liu, Jun; Ng, Tian-Tsong; Wang, Gang. 2016.
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. [원 출처](https://openaccess.thecvf.com/content_cvpr_2016/html/Shahroudy_NTU_RGBD_A_CVPR_2016_paper.html).

- 인용 key: `shahroudy2016ntu`.
- 이 연구와의 관계: DSTE 사전학습 데이터·NTU25.

<a id="r10"></a>

### R10. Unified Multi-Modal Unsupervised Representation Learning for Skeleton-Based Action Understanding

Sun, Shengkai; Liu, Daizong; Dong, Jianfeng; Qu, Xiaoye; Gao, Junyu; Yang, Xun; Wang, Xun; Wang, Meng. 2023.
Proceedings of the 31st ACM International Conference on Multimedia; DOI: 10.1145/3581783.3612449. [원 출처](https://github.com/HuiGuanLab/UmURL).

- 인용 key: `sun2023umurl`.
- 이 연구와의 관계: 공식 코드·NTU 전처리 계보; 활성 백본과 구별.

<a id="r11"></a>

### R11. Modeling Temporal Dynamics and Spatial Configurations of Actions Using Two-Stream Recurrent Neural Networks

Wang, Hongsong; Wang, Liang. 2017.
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. [원 출처](https://openaccess.thecvf.com/content_cvpr_2017/html/Wang_Modeling_Temporal_Dynamics_CVPR_2017_paper.html).

- 인용 key: `wang2017twostream`.
- 이 연구와의 관계: Foundation 논문이 명시한 두 stream 선행연구.

<a id="r12"></a>

### R12. Boosting Few-Shot Action Recognition with Graph-Guided Hybrid Matching

Xing, Jiazheng; Wang, Mengmeng; Ruan, Yudi; Chen, Bofan; Guo, Yaowei; Mu, Boyu; Dai, Guang; Wang, Jingdong; Liu, Yong. 2023.
Proceedings of the IEEE/CVF International Conference on Computer Vision. [원 출처](https://openaccess.thecvf.com/content/ICCV2023/html/Xing_Boosting_Few-shot_Action_Recognition_with_Graph-guided_Hybrid_Matching_ICCV_2023_paper.html).

- 인용 key: `xing2023gghm`.
- 이 연구와의 관계: USDRL이 DSA 영향으로 명시한 dense temporal modeling.

<a id="r13"></a>

### R13. Attention Is All You Need

Vaswani, Ashish; Shazeer, Noam; Parmar, Niki; Uszkoreit, Jakob; Jones, Llion; Gomez, Aidan N.; Kaiser, Lukasz; Polosukhin, Illia. 2017.
Advances in Neural Information Processing Systems. [원 출처](https://arxiv.org/abs/1706.03762).

- 인용 key: `vaswani2017attention`.
- 이 연구와의 관계: self-attention·위치 부호화의 기본 계열.

<a id="r14"></a>

### R14. An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale

Dosovitskiy, Alexey; Beyer, Lucas; Kolesnikov, Alexander; Weissenborn, Dirk; Zhai, Xiaohua; Unterthiner, Thomas; Dehghani, Mostafa; Minderer, Matthias; Heigold, Georg; Gelly, Sylvain; Uszkoreit, Jakob; Houlsby, Neil. 2021.
International Conference on Learning Representations. [원 출처](https://arxiv.org/abs/2010.11929).

- 인용 key: `dosovitskiy2021vit`.
- 이 연구와의 관계: ViTPose의 ViT 배경; 독립 분류기로 미사용.

<a id="r15"></a>

### R15. Layer Normalization

Ba, Jimmy Lei; Kiros, Jamie Ryan; Hinton, Geoffrey E.. 2016.
arXiv preprint arXiv:1607.06450. [원 출처](https://arxiv.org/abs/1607.06450).

- 인용 key: `ba2016layernorm`.
- 이 연구와의 관계: DSTE·J1 기본 연산.

<a id="r16"></a>

### R16. Gaussian Error Linear Units (GELUs)

Hendrycks, Dan; Gimpel, Kevin. 2016.
arXiv preprint arXiv:1606.08415. [원 출처](https://arxiv.org/abs/1606.08415).

- 인용 key: `hendrycks2016gelu`.
- 이 연구와의 관계: DSTE FFN·J1 비선형 연산.

<a id="r17"></a>

### R17. Deep Residual Learning for Image Recognition

He, Kaiming; Zhang, Xiangyu; Ren, Shaoqing; Sun, Jian. 2016.
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. [원 출처](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html).

- 인용 key: `he2016residual`.
- 이 연구와의 관계: residual 연결의 개념적 배경; ResNet backbone 미사용.

<a id="r18"></a>

### R18. Deep Networks with Stochastic Depth

Huang, Gao; Sun, Yu; Liu, Zhuang; Sedra, Daniel; Weinberger, Kilian Q.. 2016.
Computer Vision -- ECCV 2016. [원 출처](https://arxiv.org/abs/1603.09382).

- 인용 key: `huang2016stochastic`.
- 이 연구와의 관계: DSTE DropPath 계열; 평가 때 비활성.

<a id="r19"></a>

### R19. Dropout: A Simple Way to Prevent Neural Networks from Overfitting

Srivastava, Nitish; Hinton, Geoffrey; Krizhevsky, Alex; Sutskever, Ilya; Salakhutdinov, Ruslan. 2014.
Journal of Machine Learning Research. [원 출처](https://www.jmlr.org/papers/v15/srivastava14a.html).

- 인용 key: `srivastava2014dropout`.
- 이 연구와의 관계: J1 dropout 기본 연산.

<a id="r20"></a>

### R20. VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning

Bardes, Adrien; Ponce, Jean; LeCun, Yann. 2022.
International Conference on Learning Representations. [원 출처](https://arxiv.org/abs/2105.04906).

- 인용 key: `bardes2022vicreg`.
- 이 연구와의 관계: USDRL MG-FD 선행 목적함수; 현재 J1 loss 아님.

<a id="r21"></a>

### R21. Parameter-Efficient Transfer Learning for NLP

Houlsby, Neil; Giurgiu, Andrei; Jastrzebski, Stanislaw; Morrone, Bruna; De Laroussilhe, Quentin; Gesmundo, Andrea; Attariyan, Mona; Gelly, Sylvain. 2019.
Proceedings of the 36th International Conference on Machine Learning. [원 출처](https://proceedings.mlr.press/v97/houlsby19a.html).

- 인용 key: `houlsby2019adapters`.
- 이 연구와의 관계: 관련 bottleneck adapter 연구; 직접 설계 영향은 미확인.

<a id="r22"></a>

### R22. AdaptFormer: Adapting Vision Transformers for Scalable Visual Recognition

Chen, Shoufa; Ge, Chongjian; Tong, Zhan; Wang, Jiangliu; Song, Yibing; Wang, Jue; Luo, Ping. 2022.
Advances in Neural Information Processing Systems. [원 출처](https://proceedings.neurips.cc/paper_files/paper/2022/hash/69e2f49ab0837b71b0e0cb7c555990f8-Abstract-Conference.html).

- 인용 key: `chen2022adaptformer`.
- 이 연구와의 관계: zero-up 병목 adapter 비교; J1 직접 재현 아님.

<a id="r23"></a>

### R23. VMT-Adapter: Parameter-Efficient Transfer Learning for Multi-Task Dense Scene Understanding

Xin, Yi; Du, Junlong; Wang, Qiang; Lin, Zhiwen; Yan, Ke. 2024.
Proceedings of the AAAI Conference on Artificial Intelligence; DOI: 10.1609/aaai.v38i14.29541. [원 출처](https://ojs.aaai.org/index.php/AAAI/article/view/29541).

- 인용 key: `xin2024vmt`.
- 이 연구와의 관계: 기존 문서에 인용된 다중 작업 adapter 배경; 동일 구현 아님.

<a id="r24"></a>

### R24. Decoupled Weight Decay Regularization

Loshchilov, Ilya; Hutter, Frank. 2019.
International Conference on Learning Representations. [원 출처](https://arxiv.org/abs/1711.05101).

- 인용 key: `loshchilov2019adamw`.
- 이 연구와의 관계: J0/J1 optimizer; G0 SGD와 구별.

<a id="r25"></a>

### R25. Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition

Yan, Sijie; Xiong, Yuanjun; Lin, Dahua. 2018.
Proceedings of the AAAI Conference on Artificial Intelligence; DOI: 10.1609/aaai.v32i1.12328. [원 출처](https://ojs.aaai.org/index.php/AAAI/article/view/12328).

- 인용 key: `yan2018stgcn`.
- 이 연구와의 관계: 좌표 기반 skeleton 인식 비교 계열; 활성 백본 아님.

<a id="r26"></a>

### R26. Revisiting Skeleton-Based Action Recognition

Duan, Haodong; Zhao, Yue; Chen, Kai; Lin, Dahua; Dai, Bo. 2022.
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. [원 출처](https://openaccess.thecvf.com/content/CVPR2022/html/Duan_Revisiting_Skeleton-Based_Action_Recognition_CVPR_2022_paper.html).

- 인용 key: `duan2022posec3d`.
- 이 연구와의 관계: PoseConv3D heatmap 표현 비교; DSTE 내부 모듈 아님.

<a id="r27"></a>

### R27. PYSKL: Towards Good Practices for Skeleton Action Recognition

Duan, Haodong; Wang, Jiaqi; Chen, Kai; Lin, Dahua. 2022.
Proceedings of the 30th ACM International Conference on Multimedia; DOI: 10.1145/3503161.3548546. [원 출처](https://arxiv.org/abs/2205.09443).

- 인용 key: `duan2022pyskl`.
- 이 연구와의 관계: benchmark 도구·자료 형식; J1 출처 아님.

<a id="r28"></a>

### R28. SAFER-Activities: A Dataset for Smart Assessment of Fall Events and Routine Activities

Lamsal, Diwas; Wickramatilake, Pramod; Moonrinta, Jednipat; Ekpanyapong, Mongkol; Dailey, Matthew N.. 2026.
arXiv preprint arXiv:2609.08038. [원 출처](https://arxiv.org/abs/2609.08038v2).

- 인용 key: `lamsal2026safer`.
- 이 연구와의 관계: SAFER 학습 데이터와 배포 pose.

<a id="r29"></a>

### R29. Skeleton Based Efficient Fall Detection

Aslan, Muzaffer; Akbulut, Yaman; Şengür, Abdulkadir; İnce, Melih Cevdet. 2017.
Journal of the Faculty of Engineering and Architecture of Gazi University; DOI: 10.17341/gazimmfd.369347. [원 출처](https://dergipark.org.tr/tr/pub/gazimmfd/article/369347).

- 인용 key: `aslan2017fall`.
- 이 연구와의 관계: FU-Kinect 학습 데이터.

<a id="r30"></a>

### R30. Optimized Spatio-Temporal Descriptors for Real-Time Fall Detection: Comparison of Support Vector Machine and Adaboost-Based Classification

Charfi, Imen; Mitéran, Johel; Dubois, Julien; Atri, Mohamed; Tourki, Rached. 2013.
Journal of Electronic Imaging; DOI: 10.1117/1.JEI.22.4.041106. [원 출처](https://doi.org/10.1117/1.JEI.22.4.041106).

- 인용 key: `charfi2013le2i`.
- 이 연구와의 관계: Le2i 데이터 출처; SVM/AdaBoost 방법은 미사용.

<a id="r31"></a>

### R31. Human Fall Detection on Embedded Platform Using Depth Maps and Wireless Accelerometer

Kwolek, Bogdan; Kępski, Michal. 2014.
Computer Methods and Programs in Biomedicine. [원 출처](https://fenix.ur.edu.pl/~mkepski/ds/uf.html).

- 인용 key: `kwolek2014urfd`.
- 이 연구와의 관계: URFD 외부평가 데이터 출처.

<a id="r32"></a>

### R32. OmniFall: From Staged Through Synthetic to Wild, A Unified Multi-Domain Dataset for Robust Fall Detection

Schneider, David; Marinov, Zdravko; Mistol, Moritz; Zhong, Zeyun; Jaus, Alexander; Düger, Rodi; Baur, Rafael; Sarfraz, M. Saquib; Stiefelhagen, Rainer. 2026.
arXiv preprint arXiv:2505.19889v3. [원 출처](https://arxiv.org/abs/2505.19889v3).

- 인용 key: `schneider2026omnifallv3`.
- 이 연구와의 관계: 통합 annotation·분할; 자체 평가 decoder와 구별.

<a id="r33"></a>

### R33. Fall Recognition Based on Time-Level Decision Fusion Classification

Kim, Juyoung; Kim, Beomseong; Lee, Heesung. 2024.
Applied Sciences; DOI: 10.3390/app14020709. [원 출처](https://doi.org/10.3390/app14020709).

- 인용 key: `kim2024fusion`.
- 이 연구와의 관계: 관련 낙상 연구: pose/ST-GCN/시간단위 융합.

<a id="r34"></a>

### R34. Human Fall Detection Using Pose Estimation: From Traditional Machine Learning to Vision Transformers

Raza, Ali; Yousaf, Muhammad Haroon; Ahmad, Waqar; Velastin, Sergio A.; Viriri, Serestina. 2025.
Engineering Applications of Artificial Intelligence; DOI: 10.1016/j.engappai.2024.109809. [원 출처](https://e-archivo.uc3m.es/entities/publication/4836f6cb-ddd6-4038-8fed-468fb5c3c67b).

- 인용 key: `raza2025pose`.
- 이 연구와의 관계: 관련 낙상 연구: pose·classifier 비교.

<a id="r35"></a>

### R35. Robust 3D Skeletal Joint Fall Detection in Occluded and Rotated Views Using Data Augmentation and Inference-Time Aggregation

Zobi, Maryem; Bolzani, Lorenzo; Tabii, Youness; Oulad Haj Thami, Rachid. 2025.
Sensors; DOI: 10.3390/s25216783. [원 출처](https://www.mdpi.com/1424-8220/25/21/6783).

- 인용 key: `zobi2025vira`.
- 이 연구와의 관계: 관련 낙상 연구: 3D·GCN·증강.

<a id="r36"></a>

### R36. Real-Time Human Fall Detection From Video Using YOLOv11 With Pose Estimation: A Paradigm Shift Toward Efficient Transformer-Based Architectures

BenAbdennour, Adel; Sameh, Mahmoud; Khawaja, Bilal A.; Vallappil, Arshad Karimbu; Alenezi, Abdulmajeed M.; Abbasi, Qammer H.; Qazi, Sameer. 2026.
IEEE Access; DOI: 10.1109/ACCESS.2026.3674843. [원 출처](https://eprints.gla.ac.uk/384046/).

- 인용 key: `benabdennour2026fall`.
- 이 연구와의 관계: 관련 낙상 연구: 2D pose·Transformer·외부 Le2i.

<a id="r37"></a>

### R37. Dataset for Human Fall Recognition in an Uncontrolled Environment

Eraso Guerrero, José Camilo; Muñoz España, Elena; Muñoz-Añasco, Mariela; Pinto Lopera, Jesús Emilio. 2022.
Data in Brief; DOI: 10.1016/j.dib.2022.108610. [원 출처](https://doi.org/10.1016/j.dib.2022.108610).

- 인용 key: `eraso2022caucafall`.
- 이 연구와의 관계: CAUCA 외부평가 원 데이터 출처.

<a id="r38"></a>

### R38. Oops! Predicting Unintentional Action in Video

Epstein, Dave; Chen, Boyuan; Vondrick, Carl. 2020.
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. [원 출처](https://openaccess.thecvf.com/content_CVPR_2020/html/Epstein_Oops_Predicting_Unintentional_Action_in_Video_CVPR_2020_paper.html).

- 인용 key: `epstein2020oops`.
- 이 연구와의 관계: 기존 OOPS 입력 실험 계보; 현재 학습 데이터로 주장하지 않음.

<a id="r39"></a>

### R39. COCO Keypoint Detection Task and Data Format

COCO Consortium. 발행연도 미지정.
공식 소프트웨어·데이터 문서. [원 출처](https://cocodataset.org/#keypoints-2020).

- 인용 key: `cocokeypoints`.
- 이 연구와의 관계: COCO17 형식 소프트웨어·데이터 명세.

<a id="r40"></a>

### R40. Barlow Twins: Self-Supervised Learning via Redundancy Reduction

Zbontar, Jure; Jing, Li; Misra, Ishan; LeCun, Yann; Deny, Stephane. 2021.
Proceedings of the 38th International Conference on Machine Learning. [원 출처](https://proceedings.mlr.press/v139/zbontar21a.html).

- 인용 key: `zbontar2021barlow`.
- 이 연구와의 관계: USDRL이 명시한 decorrelation 배경; 현재 J1 loss 아님.
