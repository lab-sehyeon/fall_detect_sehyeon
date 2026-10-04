# SAFER V2 기준 모델 비교

- 문서 ID: `DOC-20260922-safer-v2-controls-R3`
- 기준일: 2026-09-22
- 상태: `completed` — 전체 학습·고정 평가·독립 검산 완료

## 질문과 방법

재생성한 V2 skeleton에서 frozen DSTE의 낙상 신호와4-class linear 기준 성능을 평가한다.
F0A는60-class ADL argmax가 A043일 때 낙상으로 판단하며 학습·threshold fitting을 하지 않는다.
F0B는 max-pooled temporal1024D와 temporal+spatial2048D linear head를50epochs 비교한다.
이미 고정한 reconstructed V1의 SGD 학습 조건을 동일하게 사용한다.

Validation macro-F1 우선, conditional fall-vs-lie AUPRC 차순위로 후보·epoch를 고정한다.
그 후 선택된 단일 head만 test/OOD에 적용한다. 학습 과정에서 DSTE와 ADL head는 동결한다.
입력 순서·라벨·모델 불변성·저장 출력과 metric의 독립 검산을 완료해야 확정 결과로 보고한다.

## 결과

전체1,007,723windows의 특징 추출과 두 후보의50epoch 비교를 완료했다.
Validation에서 temporal+spatial epoch22를 선택한 뒤 고정 평가했다.
동결 모델 불변성·전체 epoch 출력·지표·선택 과정의 독립 검산을 통과했다.

| Split | F0B Macro-F1 | F0B fall F1 | Conditional fall-vs-lie AP | F0A fall F1 |
| --- | ---: | ---: | ---: | ---: |
| Validation | 74.168% | 74.356% | 95.356% | 35.108% |
| Test | 76.868% | 79.652% | 96.032% | 45.351% |
| OOD | 56.335% | 54.206% | 89.173% | 22.001% |

## 주장 범위와 다음 단계

소실된 원본과 동일한 수치 복원을 주장하지 않는다.
V2 proxy3D·confidence 순서의 한계가 이어지며, V1/V2는 제외 windows도 달라 전체split 차이를
순수한 mapping 교정 효과로 해석하지 않는다. 후속 공동학습과 V3 geometry 비교의 기준으로 사용한다.

[V2 방법·구조 검증](2026-09-20_safer_v2_reconstruction_shared.md) ·
[전체 진행](2026-09-22_recovery_execution_shared.md) ·
[연구 맥락·참고문헌](2026-09-03_project_complete_summary_shared.md)
