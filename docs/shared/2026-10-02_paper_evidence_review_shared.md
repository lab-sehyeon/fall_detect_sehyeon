# 논문 근거 확충과 비판적 검토

- 문서 ID: `DOC-20261002-paper-evidence-review-R1`
- 기준일: 2026-10-02
- 상태: 요청 범위 검토·URFD 평가 `completed`; MCFD 평가 `paused`

이 문서는 실제 실행한 평가와 원문으로 확인한 문헌을 중심으로 연구 근거를 정리한다.
성능 수치의 재현 가능성과 특정 설계의 우수성은 구분한다.

## 진행 상태

| 항목 | 상태 | 확인 범위 |
| --- | --- | --- |
| URFD 외부 평가 | `completed` | 공식 camera0 전체70개, 처리 실패 포함, 독립 검산 완료 |
| MCFD 외부 평가 | `paused` | 공식 영상 확보, 정답 출처 충돌로 평가 보류 |
| 실제 TP/TN/FP/FN 사례 | `completed` | SAFER OOD supplied-pose 경로의 window 사례 |
| 관련 연구·직접 참고문헌 | `completed` | 관련 연구6편, 핵심 인용21개와 적용 범위 대조 |
| FU 결과 독립 검산 | `completed` | 993개 예측, subject 분리, 선택 기록 재검산 |

## 검산에서 확인한 사실

FU에서 체크포인트로 다시 계산한 전체 예측의 class는 저장된 예측과 일치했다.
train·validation·outer test의 피험자 분리와 각 outer 표본의 단일 평가를 확인했다.
입력 배열의 완전 일치 중복은 발견되지 않았다. 이는 모든 형태의 데이터 누출이 없다는 증명은 아니다.
별도로 결과를 보지 않고 선택한 5개 전처리 skeleton의 DSTE 특징도 재계산해 일치 여부를 확인했다.

| FU 모델 | TP | FP | FN | TN | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| J0 | 154 | 13 | 11 | 815 | 92.771% |
| J1 | 154 | 11 | 11 | 817 | 93.333% |

J1−J0의 F1 차이는 +0.562%p이다. 동일 OOF 예측에 대한 사후 피험자 단위 paired bootstrap
2,000회의 95% percentile 구간은 −1.013~+2.189%p로 0을 포함한다.
이 진단은 학습 seed 변동이나 adapter만의 인과효과를 추정하지 않는다.
따라서 현재 결과는 수치의 재현을 뒷받침하지만 확실한 성능 향상 주장의 근거로는 제한적이다.

## 실제 사례

실제 SAFER OOD window 사례 (공개 사본 미포함)

원본 RGB 프레임, supplied-pose 경로의 3D 입력, GT와 fall 확률 및 4class argmax를 함께 표시했다.
source명·시작프레임 순서로 첫 사례를 선택했으며 확률이나 미관으로 순위를 매기지 않았다.
이 그림은 window 단위 사후 설명 자료이다. FN window의 주변에 fall 예측이 있는 경우도 있으므로
이를 사건 전체의 미탐지로 해석할 수 없다. 새 RGB frontend의 성능을 나타내는 그림도 아니다.

## 외부 평가 원칙

URFD에서는 문서 기반 재구현으로 확보한 현재 가중치를 사용하며 과거 원본 실험의 동일 재현으로 주장하지 않는다.
고정 YOLOv8x·ViTPose-B·MotionAGFormer·DSTE·J1 adapter와 별도 G0 head를 사용한다.
G0는 기존 source 평가의 기준 head이며 G2의 미채택 결정을 유지한다.
target 데이터로 학습하거나 임계값을 맞추지 않는다. 전체70 sequences를 분모로 유지하고 처리 coverage를 보고한다.
64프레임 complete window의4class argmax에 fall이 한 번이라도 있으면 sequence 양성으로 집계한다.
짧은 입력이나 품질 기준 미통과는 처리 실패·무경보로 기록한다.
이는 offline sequence 분류이며 event matching, 실시간성, 회복 판단을 평가하지 않는다.

[URFD 공식 설명](https://fenix.ur.edu.pl/~mkepski/ds/uf.html)은 fall30·ADL40 및 frame timestamp를 제공한다.
adl-37에는 RGB350장 중 마지막20장의 timestamp가 없어 처리 실패·무경보로 남기고 전체70개에 포함한다.
또한12개 시퀀스(낙상11·ADL1)가25fps 기준64프레임보다 짧다. 이 경우도 무경보로 집계하므로
이번 고정 파이프라인의 recall 상한은 추론 전부터19/30이다. 이러한 입력 조건의 한계를 수치와 함께 보고한다.
MCFD는 [공식 안내](https://www-labs.iro.umontreal.ca/~labimage/Dataset/)와 technical report 간
scenario23 정답 충돌로 이번 성능평가를 보류했다.

## URFD 추가 평가 결과

**전체70개에서 F1=58.824%, recall=50.000%이다.** 고정 모델을 target 학습이나 임계값 조정 없이
평가했고, 저장 예측과 별도 계산한 confusion matrix·지표가 일치함을 확인했다.

| 평가 범위 | TP | FP | FN | TN | Precision | Recall | F1 | Specificity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 전체70개, primary | 15 | 6 | 15 | 34 | 71.429% | 50.000% | 58.824% | 85.000% |
| 분류 실행45개, 조건부 진단 | 15 | 6 | 4 | 20 | 71.429% | 78.947% | 75.000% | 76.923% |

전체 accuracy는70.000%, AP는71.115%이다. AP는 각 영상의 최대 window fall 확률을 사용하며,
처리 실패에는 사전 고정한 무경보 score0을 부여했다. 실패 영상에서 모델 confidence를 관측한 것은 아니다.
조건부45개 결과로 전체70개 성능을 대체하지 않는다.

| 처리 상태 | 전체 | 낙상 | ADL |
| --- | ---: | ---: | ---: |
| 분류 실행 | 45 | 19 | 26 |
| 공식 timestamp 누락 | 1 | 0 | 1 |
| 64frame보다 짧은 입력 | 12 | 11 | 1 |
| 그 외 pose 품질 기준 미통과 | 12 | 0 | 12 |
| 합계 | 70 | 30 | 40 |

미탐지15개 중11개는 짧은 입력으로 분류기를 실행하지 못했고,4개는 분류기까지 실행했으나 무경보였다.
TN34개에는 정상 분류20개와 처리 실패 후 무경보14개가 포함된다. 후자의14개를 모델의 성공적인
ADL 인식으로 설명하지 않는다. 이는 accuracy나 specificity만 보고 입력 실패를 놓칠 수 있음을 보여준다.

이 결과는 **native3D FU에서의 높은 성능이 현재 RGB 외부 파이프라인에 그대로 이전되지 않음**을 보여준다.
원인은 입력 길이, 모달리티·추정 pose 품질, 장면·행동 차이 등이 함께 바뀐 조건에서 해석해야 한다.
11개 short FN은 고정 처리 규칙에서 직접 확인되지만, 나머지 오류의 원인을 특정 모듈 하나로 귀속하는
비교 실험은 하지 않았다. 낮은 결과를 숨기거나 확인 후 padding·head·threshold를 바꾸지 않았다.

실제 URFD 사례와 입력 실패 (공개 사본 미포함)

그림의TP/TN/FP/FN은 **sequence 판정**이다. 분류 가능한 영상 중 각 범주의 사전순 첫 사례를 선택했다.
낙상 예측은 첫 alarm window, 무경보는 최대 fall 확률 window의 중앙을 표시한다.
짧은 입력과 pose 품질 거부도 별도로 보인다. RGB 및2D skeleton은 실제 입력과 출력이며,
event 시간 정답은 평가하지 않았으므로 임의의 낙상 구간을 표시하지 않는다.

## 비슷한 연구와 비교할 때의 기준

문헌의 높은 숫자를 그대로 한 순위표에 넣으면 입력·정답·평가 단위가 다른 결과가 섞인다.
아래 논문은 관련 연구로 인용할 수 있지만, 동일 데이터 release·분할·모달리티·학습 데이터·집계 단위를
맞춘 재실행이 없는 행은 직접적인 우열 비교 근거로 쓰지 않는다.

| 연구 | 방법과 관련성 | 확인한 평가 조건 | 우리 연구와 비교 가능한 내용 |
| --- | --- | --- | --- |
| Kim, Kim, Lee (2024), [Time-Level Decision Fusion](https://www.mdpi.com/2076-3417/14/2/709) | YOLO detector → AlphaPose/ViTPose → ST-GCN → 시간별 판단 결합. 현재 RGB 모듈 연결 구조와 가까움 | HAR-UP에서 voting/max/mean/probabilistic fusion 비교. 논문이 보고한 평균 accuracy 개선은0.84%p | pose 추출과 행동 분류를 연결하는 선행 사례. 그 개선값은 우리 adapter의 효과 근거가 아님 |
| Raza et al. (2025), [Pose Estimation to Vision Transformers](https://doi.org/10.1016/j.engappai.2024.109809) | 여러 pose estimator와 ML/RNN/LSTM/GCN/Transformer 분류기를 비교 | UR-Fall·UP-Fall·Le2i를 사용하며 Methods에70% train/30% test와10-fold CV를 기술. 검토한 설명만으로 subject/video grouping 전체를 확정하지 못함 | pose 표현과 분류기 선택의 선행 연구. target 학습이 포함된 결과와 무학습 외부평가를 구분 |
| Zobi et al. (2025), [VIRA-GCN](https://pmc.ncbi.nlm.nih.gov/articles/PMC12609388/) | 3D skeleton GCN, 회전·가림 증강, inference-time aggregation, Kinect→MMPose mapping | Le2i 원본 영상을 먼저 train/test로 분리한 뒤 증강한다고 명시. viewpoint·occlusion 변화를 평가 | 다른 skeleton 체계의 연결과 강건성 분석 참고. target 학습·합성 변형을 우리 고정 외부평가와 같은 실험으로 취급하지 않음 |
| BenAbdennour et al. (2026), [YOLOv11-Pose + Transformer](https://eprints.gla.ac.uk/384046/1/384046.pdf) | pose 추출 뒤 소형 시간 Transformer, 구조·입력·학습 조건 ablation | Le2i fine-tuning 없이109 videos의835개 zone-selected clips, stride15. Table13 F1=91.65% | 외부 일반화 평가의 유사 사례. boundary/post-fall 제외 및 whole-fall containment 정답으로, 우리의127편 event F1과 직접 비교 불가 |
| Schneider et al., [OmniFall v1](https://arxiv.org/html/2505.19889v1) 및 [갱신본](https://arxiv.org/abs/2505.19889) | 여러 데이터셋의 행동 라벨·split 통합, staged-to-wild 평가. 갱신본은 synthetic과 지속 fallen 상태 포함 | v1은10class·고정 I3D/VideoMAE 기반 실험, 갱신본은16class 및 확장된 실험. 버전별 방법·수치를 구별해야 함 | 외부 평가와 fall/fallen/lying 구분에서 가장 가까운 연구 축. OmniFall 학습에 포함된 데이터셋을 그 모델의 미학습 외부 데이터로 부르지 않음 |
| Lamsal et al. (2026), [SAFER-Activities](https://arxiv.org/abs/2609.08038) | frame-level fall·일상행동 데이터 및 skeleton/RGB/fusion baseline | in-lab·OOD·cross-dataset 분리. frozen RGB와 skeleton fusion의 효과가 평가 환경에 따라 달라짐 | 실제 학습·감독 단위와 OOD 평가의 직접 출처. 이 논문의 baseline 성능이 본 모델의 결과를 대신하지 않음 |

특히 BenAbdennour 논문의 Le2i 평가에서는 fall 시작30프레임 전까지 끝나는 clip을 ADL로,
낙상 구간 전체를 포함하는 clip을 fall로 정하고, 경계 및 낙상 이후 구간을 제외한다.
따라서 그 F1과 전체 영상의 낙상 사건을 매칭한 F1은 서로 다른 질문에 대한 답이다.
단순히 숫자가 높거나 낮다는 이유로 모델의 우열을 판단하지 않는다.

### Related Work에 사용할 수 있는 서술

> 영상 기반 낙상 인식에서는 사람 탐지, pose 추정, skeleton 행동 분류를 연결하고 시간적 판단을
> 결합하는 접근이 사용되어 왔다(Kim et al., 2024). Pose estimator 및 시간 분류기의 비교
> (Raza et al., 2025), 3D skeleton의 회전·가림에 대한 강건성 개선(Zobi et al., 2025),
> pose 기반 Transformer의 외부 데이터 평가(BenAbdennour et al., 2026) 역시 보고되었다.
> 한편 OmniFall과 SAFER-Activities는 데이터셋 및 환경 차이에 따른 평가 조건의 중요성을 다룬다.
> 본 연구는 사전학습 DSTE를 고정한 상태에서 데이터셋별 감독 단위와 분류기를 유지하면서
> 공유 adapter를 학습한 절차를 기술하고, 기존 일상행동 경로와 낙상 분기를 구분하여 평가한다.

이 문단은 수행한 구조의 위치를 설명한다. adapter 공유의 우수성이나 새로운 회복 알고리즘을
증명했다고 주장하는 문장으로 확장하지 않는다.

## 실제 참고문헌을 Methods에 연결하는 방법

### 직접 사용한 모델·데이터

| 원고 위치 | 인용할 출처 | 가져온 내용 | 본 연구에서 별도로 기술할 내용 |
| --- | --- | --- | --- |
| 사전학습 skeleton encoder | [Wang et al., Foundation Model](https://arxiv.org/abs/2508.12586), 선행 [USDRL](https://doi.org/10.1609/aaai.v39i8.32899) | DSTE와 사전학습 표현 | 고정 여부, 추출 feature, 데이터셋별 pooling. 두 논문을 별개 사용 backbone처럼 세지 않음 |
| RGB 사람 탐지 | [Ultralytics YOLOv8 공식 문서](https://docs.ultralytics.com/models/yolov8/) | YOLOv8x 소프트웨어·가중치 | 사용 설정과 실패 처리. 존재하지 않는 독립 YOLOv8 학술논문을 만들지 않음 |
| 2D pose | [Xu et al., ViTPose, NeurIPS2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/fbb10d319d44f8c3b4720873e4177c65-Abstract-Conference.html) | COCO17 pose 추정기 | SAFER 제공 pose와 외부 RGB의 ViTPose-B 차이 |
| 3D lifting | [Mehraban et al., MotionAGFormer, WACV2024](https://openaccess.thecvf.com/content/WACV2024/html/Mehraban_MotionAGFormer_Enhancing_3D_Human_Pose_Estimation_With_a_Transformer-GCNFormer_Network_WACV_2024_paper.html) | H36M17 lifting 모델 | overlap fusion·좌표 정규화·NTU25 proxy는 프로젝트 적용 절차로 설명 |
| ADL 평가 | [Shahroudy et al., NTU RGB+D, CVPR2016](https://openaccess.thecvf.com/content_cvpr_2016/html/Shahroudy_NTU_RGBD_A_CVPR_2016_paper.html) | 데이터·관절 체계·XSub 분할 | 사용 checkpoint/모달리티/Top1·Top5·기존 경로 유지 |
| FU 낙상 학습·평가 | [Aslan et al., Skeleton Based Efficient Fall Detection](https://doi.org/10.17341/gazimmfd.369347) 및 [공개 데이터](https://github.com/MuzafferAslan23/Fall-Detection-Dataset) | FU-Kinect-Fall | 현재 사용993 clips·21subjects·fold·clip feature mean·낙상 정의 |
| SAFER 감독·OOD | [SAFER-Activities](https://arxiv.org/abs/2609.08038) | 데이터·frame 라벨·split | 64frame center 라벨,4class 매핑, 추가 adapter/head 학습 |
| Le2i 과거 평가 | [Charfi et al., JEI2013](https://doi.org/10.1117/1.JEI.22.4.041106) | Le2i 데이터 출처 | 당시127편·event matching·품질 실패와 현재 재검증 불가 범위 |
| URFD 추가 평가 | [Kwolek & Kępski (2014), 공식 데이터 안내](https://fenix.ur.edu.pl/~mkepski/ds/uf.html) | fall30·ADL40, RGB sequence와 timestamp | camera0 선택,25fps mapping,짧은 입력·처리 실패,sequence 집계 |
| MCFD 추가 평가 준비 | [Auvinet et al. (2010), 공식 데이터·기술보고서](https://www-labs.iro.umontreal.ca/~labimage/Dataset/) | 24 scenarios·8 cameras | annotation 충돌과 평가 보류. 준비 완료를 성능 검증 완료로 쓰지 않음 |

COCO/Human3.6M은 각각 pose 체계 및 upstream 학습 데이터의 배경으로 인용한다.
UmURL·PYSKL은 실제 사용한 전처리나 baseline 절차를 설명할 때 연결한다.
OOPS·OmniFall은 해당 외부 영상과 재주석 계보를 설명할 때 연결하며,
미채택 LaDy·상태 안정화·VLM 관련 문헌을 현재 핵심 J1 Methods에 필수 모듈처럼 넣지 않는다.
기존의 전체46개 출처와 사용·검토 범위는 [전체 참고문헌](2026-09-03_project_complete_summary_shared.md#27-전체-참고문헌)에 보존되어 있다.

핵심 인용은 [BibTeX21개](2026-10-02_paper_core_references.bib), 관련연구의 비교 조건은
[CSV6개 연구](2026-10-02_related_work_matrix.csv)로도 제공한다.

### 연산의 출처와 설계 효과를 구분

| 항목 | 참고문헌 | 현재 원고에 적절한 설명 |
| --- | --- | --- |
| residual 연결 | [He et al., ResNet, CVPR2016](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) | 기존 feature에 학습된 보정을 더하는 구조적 배경. 여기의 작은 adapter가 ResNet을 그대로 재현한 것은 아님 |
| adapter 개념 | [Houlsby et al., ICML2019](https://proceedings.mlr.press/v97/houlsby19a.html) | backbone 고정과 소규모 모듈 학습의 선행 맥락. 본 연구의 위치·차원·head 공유 구조는 실제 구현대로 별도 기술 |
| GELU | [Hendrycks & Gimpel](https://arxiv.org/abs/1606.08415) | 실제 사용한 비선형 연산. 본 과제에서 ReLU보다 우수함을 확인했다는 뜻이 아님 |
| AdamW | [Loshchilov & Hutter](https://arxiv.org/abs/1711.05101) | 사용한 optimizer와 decoupled weight decay의 출처. 다른 optimizer보다 낫다는 본 연구 결과가 아님 |

현재 adapter의 구현식은

\[
z' = z + W_{up}\,\mathrm{Dropout}\bigl(\mathrm{GELU}(W_{down}\mathrm{LN}(z)+b_{down})\bigr)+b_{up}
\]

이며 feature2048차원, bottleneck256차원이다. Up-projection의 weight와 bias를0으로 초기화하므로
초기에는 \(z'=z\)이다. 이 등식은 코드에서 따르는 초기 상태의 성질이다.
실제 학습에서의 우수성, 보편적인 망각 방지 효과, GELU·AdamW의 최적성은 별도의 경험적 주장이다.
실험 당시 해당 논문에서 직접 구조를 가져왔다는 기록이 없으면, 지금 추가한 배경 인용을 역사적 설계 출처로 소급하지 않는다.

## 높은 성능을 설명할 수 있는 근거와 아직 없는 근거

### 지금 확인되는 설명

1. **FU는 native3D skeleton을 사용하는 clip 이진 분류이다.** RGB 사람 탐지·pose·lifting의 실패를
   포함하는 외부 전체 경로 평가보다 입력 조건이 다르다. 이것만으로 높은 값이 잘못됐다는 뜻은 아니지만
   실세계 RGB 낙상 검출 성능으로 그대로 옮길 수는 없다.
2. **J0부터 F1이92.771%이다.** 고정 DSTE feature와 선형 head의 조합만으로 높은 값이 나온다.
   따라서 J1의93.333% 전체를 adapter의 기여로 설명할 수 없다. 표현의 유용성과 어댑터의 추가 이득은 구분해야 한다.
3. **J1의 개선은 집계상 FP13→11에 대응한다.** TP154·FN11은 두 모델의 집계가 같다.
   이는 관찰된 결과에 대한 설명이며, 동일 TP 표본을 맞혔다는 뜻이나 원인을 증명하는 설명은 아니다.
4. **FU 분류기 비교에서 RBF-SVM F1은93.617%, logistic regression은93.491%이다.**
   J1은 모든 비교군 중 최고 F1이 아니다. 또한 두 통제모델은 FU만, J0/J1은 SAFER+FU를 사용하여
   동일 학습 데이터 조건의 adapter ablation으로 해석할 수 없다.

관련 현재 수치와 출처는 [벤치마크 검토](2026-09-03_13_benchmarks_provenance_shared.md)에 정리되어 있다.
NTU60의Top1, FU의clip F1, Le2i의event F1은 서로 다른 과제이므로 하나의 종합 성능처럼 평균하지 않는다.

### 현재 자료만으로 확정할 수 없는 설명

- residual 또는 GELU를 사용해서 다른 선택보다 성능이 높아졌다는 주장.
- 두 데이터셋의 상보성이나 adapter 공유만으로 개선이 발생했다는 주장.
- 기존 문헌의 서로 다른 평가 수치보다 높으므로 SOTA라는 주장.
- selected checkpoint의 기록이 존재하므로 과거 holdout을 한 번도 보지 않았다는 주장.
- 사건 F1이 높으므로 낙상 이후 회복 과정을 정확히 판단한다는 주장.

이러한 주장을 하지 않는다면 모든 활성화 함수·optimizer 조합의 비교를 Methods 작성의 선행 조건으로
둘 필요는 없다. 수행한 구성을 정확히 설명하고 현재 검증된 결과의 범위에서 결론을 쓰면 된다.

## 결과가 허위 또는 과장인지에 대한 비판적 검토

| 점검 질문 | 이번에 확인한 것 | 남는 한계·원고 처리 |
| --- | --- | --- |
| 숫자가 실제 예측에서 나오는가? | FU 전체993 예측을 체크포인트에서 재계산, class 전부 일치 | 수치 재현 근거. 연구자의 의도나 모든 데이터 정당성을 증명하지 않음 |
| feature가 실제 encoder에서 나오는가? | 미리 고른5개 입력의 DSTE 특징 재계산 통과 | 전체993 raw 파일의 수집·전처리 전수 검증으로 확대하지 않음 |
| train/test가 섞였는가? | 기록된5fold에서 subject가 분리되고 outer 표본이1회씩 평가됨 | near duplicate·전체 사전학습 데이터와의 overlap·미기록 개발 과정은 별도 문제 |
| 유리한 표본만 골랐는가? | FU 전체분모 확인, URFD 전체70 및 처리 실패 보존 | 품질 통과 표본만의 성능을 주 성능으로 바꾸지 않음 |
| 사후 모델을 primary로 바꿨는가? | Le2i historical primary G2+D1과 사후 G0 결과를 구분 | 현재 새 G0 외부평가를 과거 primary였다고 서술하지 않음 |
| 통계적 차이가 충분한가? | J1−J0의 subject bootstrap 구간이0을 포함 | 개선 관찰은 보고하되 확실한 우월성으로 확대하지 않음 |
| 비교 논문과 같은 문제를 풀었는가? | frame·window·clip·event 및 target 학습 여부를 대조 | 조건 미일치 문헌은 Related Work·맥락 비교로 사용 |
| 안 된 실험을 숨겼는가? | G2 미채택, MCFD 정답 충돌, 짧은 URFD 입력을 명시 | negative result 및 처리 실패를 Results/Limitations에 포함 |
| 과거 Le2i가 현재도 재현되는가? | 영상·주석 재취득 완료, 과거 예측은 없으며 새 모델평가 미실행 | 현재 모델 성능으로 재인증하지 않고 ‘기존 기록 기준’ 표시 |

현재 점검에서 **FU의 보고 수치가 저장 예측 및 체크포인트와 모순된다는 증거는 발견하지 못했다.**
다만 이 결론을 “연구 전체가 부정행위와 무관함을 증명했다”로 확대할 수는 없다.
현재 더 뚜렷한 위험은 서로 다른 평가 조건의 수치를 비교하거나 작은 차이에 큰 설계 효과를 귀속하여
결론을 과장하는 것이다. 검산 범위와 한계를 함께 공개하면 이 위험을 줄일 수 있다.

별도 진행된 [Le2i 데이터 재취득](2026-10-02_le2i_acquisition_shared.md)은 영상·주석130쌍과
유효127개(fall96·nonfall31)를 확인했다. 이는 데이터 확보 결과이며 과거 파일의 byte 동일성,
시간 정합·디코딩 또는 현재 모델의 새 성능을 검증한 결과는 아니다.

## 원고 구성에 반영할 항목

- **Methods:** 실제 encoder·입력·pooling·adapter·head·loss·최적화·선택 절차를 서술하고 직접 출처를 연결한다.
- **Evaluation protocol:** native skeleton/RGB, target 학습 여부,분할,단위,전체 분모,실패 처리,primary 선택을 먼저 명시한다.
- **Results:** NTU/FU/외부 평가를 과제별로 나누고 confusion count와coverage를 함께 제시한다.
- **Qualitative results:** 실제 TP/TN/FP/FN을 GT·시간축과 함께 보이며 window 사례를 event 결과로 확대하지 않는다.
- **Discussion:** FU 결과가 재현되는 근거, 작은 J1 차이의 불확실성, RGB 입력 실패와 역사적 결과의 검증 한계를 다룬다.

후속 ablation은 강화하고 싶은 주장에 맞춰 선택한다. 비교 실험이 아직 없다는 이유로 실제 수행한
Methods의 기술을 미루지 않는다.
