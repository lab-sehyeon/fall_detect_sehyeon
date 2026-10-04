# FU 공통 표현의 분류기 비교 재구현

- 문서 ID: `DOC-20260922-fu-classifier-reconstruction-R3`
- 기준일: 2026-09-23
- 상태: `completed` — R3 전체 실행 및 독립 검산 완료

## 최신 완료 확인 — 9월23일

기존분류기를재학습·재선택하지않고V3공동학습과동일993개nested OOF비교를완료했다. F1은Logistic93.491%,LinearSVM89.011%,RBF-SVM93.617%,RandomForest87.261%,V3J0/J1은92.771/93.333%다. J1이기존분류기모두를능가하지는않았다. 모델별선택과평가의독립검산을통과했다. 아래V3대기문장은이전이력이다.

같은 frozen DSTE2048D표현에서 분류기 종류만 바꿨을 때 J0/J1의 비교 위치를 확인한다.
FU993clips, outer5fold/inner다음fold/나머지3fold학습을 공통으로 사용한다.
Balanced logistic, linear SVM, RBF-SVM, random forest를 비교한다.

유실된 hyperparameter grid와 train-only 표준화, seed, inner선택식을 별도 재구현으로
실행 전에 고정했다. 기본 predict 경계를 유지하고 outer threshold calibration을 하지 않는다.
모든 분류기 선택·출력·검산을 고정한 뒤에만 별도 V3 J0/J1 nested 결과를 그대로 비교한다.

네분류기의전체nested평가와독립검산을완료했다. 993clips가outer에각1회포함됐고,
선택모델의독립재학습으로예측·score를정확히재현했다.

| 분류기 | Precision | Recall | F1 | AP | lying FP/168 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic | 91.329% | 95.758% | 93.491% | 98.290% | 15 |
| Linear SVM | 81.407% | 98.182% | 89.011% | 97.888% | 29 |
| RBF-SVM | 93.902% | 93.333% | 93.617% | 98.416% | 10 |
| Random Forest | 91.946% | 83.030% | 87.261% | 96.007% | 12 |

선행V3geometry미채택으로V3J0/J1reference비교는미완료다.
외부 논문 시스템의 재현이 아니라 같은 표현 위의 분류기 통제이며,
원본과 동일한 구현·수치 또는 어느 모델의 보편적 우월성을 주장하지 않는다.

[V3 공동학습](2026-09-22_joint_v3_reconstruction_shared.md) ·
[연구 맥락·참고문헌](2026-09-03_project_complete_summary_shared.md)
