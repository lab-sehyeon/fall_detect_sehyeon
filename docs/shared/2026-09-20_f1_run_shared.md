# SAFER F1 문서 기반 재구현 — 실행 결과

- 문서 ID: `DOC-20260920-f1-run-R26`
- 기준일: 2026-09-20
- 연구 상태: `completed` — 전체 실험·독립 감사 완료

이 문서는 실제 실행 산출물에서 생성되는 진행 기록이다. 원본 코드·checkpoint의 완전 복원이 아니라 사전 고정한 별도 재구현 실험이며, 역사적 수치를 새 결과로 복사하지 않는다.

## 방법

Frozen DSTE/ADL, 64-frame temporal/temporal+spatial 표현과 plain/sqrt CE의 네 후보를 비교한다. Hidden512 residual block, batch128, AdamW lr0.001, 20epochs의 새 조건을 사용한다. Validation timeline macro-F1 → fall-vs-unstable AUPRC → lying F1로 선택하고 이후 고정 모델만 test/OOD에 평가한다. Raw logits를 overlap mean한 뒤 softmax를 적용한다.
| 후보/평가 | Macro-F1 | Fall F1 | Fall-vs-unstable AUPRC | Lying F1 |
| --- | ---: | ---: | ---: | ---: |
| test | 75.278% | 79.063% | 92.575% | 37.959% |
| ood | 56.516% | 54.271% | 94.671% | 21.993% |

선택: temporal_spatial__sqrt, epoch 17. 입력·모델 불변성, 저장 prediction 지표 및 overlap 재구성의 독립 감사를 통과했다.

## 해석 경계

미커버 frame은 평가에서 제외하며 coverage를 별도 보고한다. Event 지표는 사전 정의한 segment IoU≥0.1의 재구현 지표로서 역사적 event F1과 동일하지 않다. 아직 최종 감사 전이면 성능 재현 완료로 해석하지 않는다.

[재현 계약](2026-09-20_f1_reconstruction_contract_shared.md) · [선행 F0B 결과](2026-09-19_f0b_postselection_shared.md)
