# HFD·SDFA 원본 학습 모델 추가 확보

문서 ID: DOC-20261004-original-checkpoint-search-R1  
기준일: 2026-10-04  
상태: 공개 파일 추가 취득·검증 `completed`, HFD·SDFA 낙상 평가 `paused`

HFD의 C3D 가중치를 추가로 확보하고 저자 코드에 정상 로딩됨을 확인했다.
그러나 최종 낙상 판정에 필요한 학습된 SVM은 확보하지 못했다. SDFA도 학습 완료 checkpoint가
아직 확보되지 않아 두 모델의 Le2i 130개·CAUCA 100개 성능 평가는 진행하지 못했다.
기존 RTHFD 비교 결과는 유지한다.

| 항목 | 추가 확인한 내용 | 외부 낙상 평가 상태 |
| --- | --- | --- |
| HFD C3D | 저자가 지정한 사전학습 가중치 확보, 전체 가중치 일치·특징 추출 확인 | 특징 추출기만 준비됨 |
| HFD SVM | 학습 완료 낙상 SVM 미확보 | HFD 전체 평가 미실행 |
| SDFA | 저자 원본 checkpoint 미확보 | 미실행 |
| 같은 저자의 DICTA2021 모델 | 학습 가중치6종 확보, 원래 모델에는 모두 정상 로딩 | 다른 모델이므로 별도 검토 대상 |

HFD의 확보 파일은 Sports1M의487개 행동 분류로 사전학습된 C3D다. 이를 낙상/비낙상 SVM으로
간주할 수 없다. 원본 notebook도 C3D 특징을 추출한 뒤 SVM을 별도로 학습한다.
([HFD 공식 코드](https://github.com/ekramalam/HFD_3DCNN),
[코드가 지정한 C3D 배포처](https://github.com/aslucki/C3D_Sport1M_keras))

SDFA와 같은 저자의 선행 연구에서 발견한 가중치는
*Modeling Human Skeleton Joint Dynamics for Fall Detection*의 모델이다.
NTU60·NTU120·UWA3D에 대응하는6개 파일은 해당 모델에 정상 로딩되지만 SDFA와 파라미터 구조가
다르다. 따라서 SDFA 결과로 대신 보고할 수 없다.
([선행 연구 공식 가중치](https://github.com/saniazahan/Modeling-Human-Skeleton-Joint-Dynamics-for-Fall-Detection-/tree/main/weights),
[SDFA 공식 코드](https://github.com/saniazahan/SDFA))

추가 조사에는 공식 저장소 밖의 연구자료 배포, 저자의 과거 홈페이지 자료, 학위논문 첨부 정보,
모델 허브와 공개 노트북을 포함했다. 이름이 같은 다른 분야의 SDFA 자료는 제외했다.
공개된 학위논문 페이지는 확인했지만 첨부 본문의 접근은 제한되어 그 내용을 검증하지 못했다.
([학위논문 공식 페이지](https://research-repository.uwa.edu.au/en/publications/human-motion-analysis/))

가중치 취득·로딩 성공은 source 성능 재현이나 외부 데이터셋 평가 완료를 의미하지 않는다.
이번 작업에서 HFD·SDFA를 새로 학습하거나 다른 모델의 점수를 두 모델의 결과로 기재하지 않았다.
추가 실행에는 HFD의 학습된 SVM과 SDFA의 원본 checkpoint가 필요하다. 선행 연구 모델을
사용하려면 별도 비교 모델로 구분하고 그 모델의 입력·전처리·평가 조건을 먼저 확정해야 한다.
