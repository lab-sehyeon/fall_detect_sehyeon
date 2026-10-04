# SAFER V3 동일 조건 기준 모델 비교

- 문서 ID: `DOC-20260922-safer-v3-controls-R5`
- 기준일: 2026-09-23
- 상태: `completed` — R3 전체 실행 및 독립 검산 완료

## 최신 완료 확인 — 9월23일

R3 전체 비교와 독립 검산이 완료됐다. Temporal+spatial epoch22를 validation에서 선택했으며 test/OOD macroF1은76.985/57.442%,fallF1은80.551/56.109%다. 같은 V2 controls의fallF1 79.652/54.206%와 비교할 수 있지만, 의미적3D정확도·외부일반화 검증으로 확대하지 않는다. 아래 미실행 문장은 이전 이력이다.

V3 geometry 기준을 통과한 입력에 대해, V2와 같은 frozen DSTE 및 linear-head 학습 조건을
사용한다. Temporal-only/temporal+spatial 두 후보를50epochs 비교하고 validation에서만 선택한다.
입력 수·순서·라벨은 V2와 같게 유지하고 겹치는 window는 sequence에서 직접 가져온다.

모델 고정 후 test/OOD에 적용하며 ADL argmax-A043 무학습 비교도 함께 보고한다.
모델 불변성, 입력·출력, 지표와 선택 과정의 독립 검산을 통과해야 결과를 확정한다.
V3 geometry 실패 시 이 비교를 진행하지 않는다. 현재 확정 성능은 없다.

별도 문서 기반 재구현으로 원본 구현·수치와의 동일성을 주장하지 않는다.
Geometry continuity와 분류 성능은 별도 판단하며 전자의 개선이 후자의 개선을 보장하지 않는다.

[V3 방법](2026-09-22_safer_v3_reconstruction_shared.md) ·
[V2 비교](2026-09-22_safer_v2_controls_shared.md) ·
[연구 맥락·참고문헌](2026-09-03_project_complete_summary_shared.md)

이번 후속은 [R3의 유효 근거 재검산](2026-09-22_v3_valid_support_shared.md)을 선행한다.
이전 V3 미채택 결과는 유지하며 R3 전체 검증이 통과해야 이 비교를 시작한다.
학습·선택 방법은 바꾸지 않으며 현재 R3 controls의 확정 성능은 없다.
