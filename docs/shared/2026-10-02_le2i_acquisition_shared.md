# Le2i 외부평가 데이터 확보

> 후속 정정: [제외3개의 정답 확인](2026-10-03_le2i_missing_annotation_search_shared.md)에서
> 낙상 시작·종료 번호가 원본 주석 중간에 존재함을 검증했다.130영상/99낙상 정답 목록을 확보했다.
> 아래127개는 당시 첫 두 줄 형식으로 읽을 수 있었던 평가 범위이며, 기존 결과는 보존한다.

문서 ID: DOC-20261002-le2i-acquisition-R1  
갱신일: 2026-10-02  
상태: completed (데이터 확보·파일 검증); 모델 평가 planned

후속 진행: 위 상태와 아래 진행 항목은 데이터 취득 완료 시점의 기록이다.
이후 입력 전체 검증과127개 모델 평가·별도 검산을 완료했으며,
최신 결과는 [현재 모델 Le2i 비교 평가](2026-10-02_le2i_current_evaluation_shared.md)에 정리했다.

## 목적과 데이터 범위

낙상 이벤트의 시간적 탐지 성능을 평가하기 위해 정지 이미지가 아닌 Le2i 영상과 낙상 시작·끝 주석을 확보한다.
기존 기록에 명시된 [Le2i Kaggle 배포본](https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia)의
Coffee_room_01/02와 Home_01/02를 대상으로 하며, 현재 공개 버전 2를 사용한다.

영상 130개와 대응 주석 130개를 모두 확보했다. 주석 헤더가 없는 3개를 제외한
평가 후보는 127개(낙상 96개, 비낙상 31개)다. 현재 자료의 주석을 검사한 결과,
이 수량과 제외 대상 3개가 기존 기록과 일치했다.
Office와 Lecture_room은 이번 부분집합에 포함하지 않는다.

| 장면 | 확보 영상 | 주석 유효 영상 | 낙상 / 비낙상 |
| --- | ---: | ---: | ---: |
| Coffee_room_01 | 48 | 47 | 47 / 0 |
| Coffee_room_02 | 22 | 20 | 12 / 8 |
| Home_01 | 30 | 30 | 30 / 0 |
| Home_02 | 30 | 30 | 7 / 23 |
| 합계 | 130 | 127 | 96 / 31 |

## 진행과 한계

영상·주석 확보와 파일 무결성 검증을 완료했다. 영상·주석 짝의 누락과 파일 검증 불일치는 없었다.
새로운 모델 평가 결과는 없다.
정지 이미지와 객체 위치 주석만 있는 변환본은 시간적 이벤트 평가를 대체하지 못한다.
동일 배포처와 평가 부분집합을 확인하더라도 소실된 과거 자료와 파일 단위 동일성을 증명한 것은 아니다.
전체 프레임 디코딩은 아직 검증하지 않았다. 영상의 시간 정보와 주석 정합성을 검증해야 실제 평가에 사용할 수 있다.
Le2i에는 이 프로젝트의 회복 상태 평가에 필요한 정답이 없으므로 회복 성능 평가용으로 간주하지 않는다.

## 출처 및 사용 제한

공식 [dataUBFC 데이터 기록](https://search-data.ubfc.fr/FR-13002091000019-2024-04-09_Fall-Detection-Dataset.html)
(DOI: 10.25666/DATAUBFC-2024-04-09)은 CC BY-NC-SA로 소개한다.
Kaggle 미러의 라이선스 필드는 Unknown이므로 미러가 별도 이용 권리를 보장한다고 보지 않는다.
연구용 활용을 전제로 하며 원본 영상은 비식별 자료가 아니다.

관련 논문: Charfi et al., “Optimised spatio-temporal descriptors for real-time fall detection:
comparison of SVM and Adaboost based classification,” Journal of Electronic Imaging 22(4), 041106 (2013).
[논문 DOI](https://doi.org/10.1117/1.JEI.22.4.041106).
