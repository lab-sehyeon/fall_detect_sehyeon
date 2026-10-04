# 추가 비교 모델 후보

- 문서 ID: `DOC-20261004-additional-baseline-candidates-R1`
- 기준일: 2026-10-04
- 상태: 후보 조사 `completed`; 추가 평가 미실행

최신성 우선 요구에 따른 후속 검토는 [최신 비교 모델 후보](2026-10-04_recent_baseline_candidates_shared.md)에 정리했다.

## 1. 조사 목적

현재의 HFD·Privacy X3D 비교에 추가할 후보를 원저자 코드, 학습된 모델의 확보 수준,
학습 데이터, 입력 표현과 평가 목적을 기준으로 조사했다. 평가 대상은 Le2i130개와
CAUCA100개이며, 해당 데이터에 대한 추가 학습이나 판정 기준 조정 없이 비교하는 조건을
유지한다. 새 후보의 외부 성능은 아직 측정하지 않았다.

## 2. 우선 검토 후보

| 후보 | 연구상 역할 | 원본 모델 확인 수준 | 평가 전 필요한 작업 |
| --- | --- | --- | --- |
| **PoseC3D-SAFER** | 2D 관절 기반 행동·낙상 인식과 사용자3D 표현 비교 | 공식 가중치와 기존 부분집합 추론 확인 | 동일 Le2i130·CAUCA100으로 평가 확대 |
| **Modeling Human Skeleton Joint Dynamics for Fall Detection**, DICTA2021 | 학습된 스켈레톤 낙상 이진 분류기 비교 | 저자 가중치 확보·원래 모델 로딩 확인 | NTU25관절 입력·좌표·정규화 및 RGB에서의 관절 획득 조건 확인 |
| **UMDR RGB**, TPAMI2023 | 현대 RGB 행동 표현의 낙상 전이 비교 | 저자 RGB 전용 모델 안내와 공개 파일 목록 확인 | 가중치 취득·로딩, 입력 영역 처리와 A43 낙상 클래스 매핑 확인 |

### PoseC3D

PoseC3D는2D 관절을 heatmap으로 표현하고3D CNN으로 시간 정보를 처리한다.
SAFER 학습 모델은 낙상이 포함된15개 행동을 분류하므로 현재 연구와 직접 비교하기
좋다. NTU60 학습 모델은 A43 낙상 행동을 이용하는 일반 행동인식 전이 기준선으로
구분할 수 있다. [PoseC3D 논문](https://openaccess.thecvf.com/content/CVPR2022/html/Duan_Revisiting_Skeleton-Based_Action_Recognition_CVPR_2022_paper.html),
[SAFER 공식 추론 안내](https://github.com/safer-activities/SAFER-Activities/blob/994ed688ce9e491245ee96c1665e948c6ce6c74d/inference/README.md).

이는 새로운 모델 취득보다 기존 평가의 확대에 해당한다. 기존 완료 범위는 세 데이터셋
합계94개이며, 그중 Le2i38개·CAUCA19개다. 이 부분집합 결과를 전체 데이터셋 점수로
사용하지 않는다. 공통2D 관절 추출기를 사용했던 기존 평가는 원저자의 전체 입력 처리
시스템과 구분하여 보고한다.

### DICTA2021 낙상 모델

이 모델은 SDFA와 다른 논문의 낙상 모델이다. 저자 배포 가중치를 이용할 수 있다는 점이
장점이다. 다만 NTU 형식25개 관절을 기대하므로, 현재 사용자 모델의17개 관절을 그대로
넣을 수 없다. 관절의 의미·좌표·정규화 조건을 맞추는 방법부터 확인해야 한다.
가중치 로딩 성공은 외부 RGB 영상에서의 평가 완료를 의미하지 않는다.
[저자 공식 구현](https://github.com/saniazahan/Modeling-Human-Skeleton-Joint-Dynamics-for-Fall-Detection-),
[논문](https://ieeexplore.ieee.org/abstract/document/9647270).

### UMDR RGB

RGB-only 모델을 RGB-D 융합 모델과 구분하여 선택할 수 있다. NTU60의 낙상 행동과 나머지
행동을 분리하는 보조 비교가 가능하지만, 원래부터 낙상 이진 분류용으로 학습된 모델이라고
소개해서는 안 된다. 또한 저자 전처리는 NTU의 depth mask를 활용한 영상 영역 추출을
포함하므로 외부 RGB에서의 영역 처리 규칙을 고정하고 차이를 명시해야 한다.
[공식 구현과 모델 안내](https://github.com/zhoubenjia/MotionRGBD-PAMI),
[논문](https://ieeexplore.ieee.org/abstract/document/10122710/).

## 3. 우선순위를 낮춘 후보

**CNN + Optical Flow**는 optical flow VGG16 특징과 URFD 낙상 분류기로 구성되어
연구 목적에는 적합하다. 그러나 이번 확인에서는 저자 안내의URFD5개 fold 및 묶음
가중치 링크가 모두 접근 실패였다. 따라서 원본 가중치로 즉시 실행할 수 있는 후보로
분류하지 않았다. FDD/여러 데이터셋 통합 학습 모델을 대체 사용하면 Le2i 학습 미사용
조건을 다시 확인해야 한다. [저자 공식 저장소](https://github.com/AdrianNunez/Fall-Detection-with-CNNs-and-Optical-Flow).

**STGCN-GRU-BiLSTM·DistillH-Mamba·FLASH**는 낙상 중 impact 순간 검출을 중심으로
연구되어, 낙상 유무와 사건 평가로 연결하는 별도 정의가 필요하다. 최근 구조라는 이유만으로
우선하지 않는다. [STGCN-GRU-BiLSTM](https://github.com/Tresor-Koffi/impact-fall-detection-stgcn),
[DistillH-Mamba](https://github.com/Tresor-Koffi/DistillH-Mamba),
[FLASH](https://github.com/Tresor-Koffi/FLASH-Impact-Fall-Detection).

## 4. 권고와 해석 범위

실행 준비 측면에서는 **PoseC3D-SAFER의 전체 평가 확대**를 우선한다. 새 낙상 전용 모델은
**DICTA2021**, 새로운 RGB 계열 보조 비교는 **UMDR RGB**를 검토한다. 모든 후보가 즉시
실행 가능한 상태는 아니며, 외부 성능이나 사용자 모델의 우위는 아직 알 수 없다.

모델 선택은 표현 방식과 학습 출처, 재현 가능성을 기준으로 한다. 기존의 더 높은 비교 모델
결과도 함께 유지한다. 같은 영상·정답·실패 포함 집계로 영상 F1/Precision/Recall을 비교하고,
시간 출력이 제공되어 동일 사건 규칙을 적용할 수 있을 때 사건 F1을 추가한다. 영상 F1과
사건 F1을 한 열에 혼합하지 않는다.

학습 데이터 명세에 Le2i·CAUCA가 없다는 확인과 모든 사전학습 자료의 샘플 중복을
배제했다는 주장은 구별한다. 여러 가중치 중 외부 시험 성능이 가장 높은 것을 고르는 방식은
사용하지 않는다. 이번 조사는 후보와 조건을 정리한 것이며 신규 성능 결과는 없다.
