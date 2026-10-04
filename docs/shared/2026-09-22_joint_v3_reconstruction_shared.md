# V3 기반 J0/J1 공동학습 재구현

- 문서 ID: `DOC-20260922-joint-v3-reconstruction-R5`
- 기준일: 2026-09-23
- 상태: `completed` — R3 전체 실행 및 독립 검산 완료

## 최신 완료 확인 — 9월23일

R3 개발·고정 평가·nested평가·최종 학습과 독립 검산이 완료됐다. FU nested J0/J1 F1은92.771/93.333%, recall은같은93.333%이며오탐13→11이다. 새nested선택에따른최종학습은13/8epochs다. J1의SAFER고정5foldensemble test/OOD fallF1은79.709/52.915%로,단일final모델성능과구분한다. 모든계보에서향상됐다고주장하지않는다. 아래미실행문장은이전이력이다.

V3 geometry와 matched linear controls가 통과한 입력으로 J0/J1 공동학습을 비교한다.
V1/V2와 같은 frozen2048D표현, dataset별 head, zero-init shared residual,
학습·선택 조건을 유지하고 SAFER 입력 계보만 구분한다.

Development5fold→모델고정후SAFER test/OOD→nested FU→최종 고정횟수 학습 순서다.
최종 epoch는 이번 nested inner 선택값의 중앙값으로 정하며 과거 횟수를 강제하지 않는다.
J1 epoch0는J0와동일해야하며, final 학습데이터 재평가는 일반화 증거로 사용하지 않는다.

현재 학습 결과는 없다. 원본과 동일한 수치의 복원이 아닌 별도 문서 기반 재구현이며,
geometry continuity, 분류 성능, 외부 일반화 주장을 서로 구분한다.

[공동학습 방법](2026-09-22_joint_reconstruction_shared.md) ·
[V3 controls](2026-09-22_safer_v3_controls_shared.md) ·
[연구 맥락·참고문헌](2026-09-03_project_complete_summary_shared.md)

이번 후속은 [R3의 유효 근거 재검산](2026-09-22_v3_valid_support_shared.md)을 선행한다.
이전 V3 미채택 결과는 유지하며 R3 전체 검증이 통과해야 이 비교를 시작한다.
학습·선택 방법은 바꾸지 않으며 현재 R3 J0/J1의 확정 성능은 없다.
