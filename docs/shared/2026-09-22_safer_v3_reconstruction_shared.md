# SAFER V3 overlap-add 재구현

- 문서 ID: `DOC-20260922-safer-v3-reconstruction-R3`
- 기준일: 2026-09-22
- 상태: `not_selected` — pilot 완료, 퇴화 입력 검사 미통과로 후속 중단

이 문서는 이전 미채택 결과를 보존한다. 이후의 별도 post-hoc 실험은
[R3 유효 근거 재검산](2026-09-22_v3_valid_support_shared.md)에서 구분한다.

## 연구 질문과 방법

243프레임 단위 3D lifting의 경계 불연속을 overlap-add로 완화할 수 있는지 평가한다.
Train에서 label/activity와 무관하게8개 sequence를 고정하고 stride243/121/81 및
uniform/triangular/Hann weighting을 비교한다. 같은 stride의 raw 예측은 공유한다.

저장된 V2 대비 경계 artifact rate75%이상 감소, reprojection p95와bone-length CV p95
각1.05배이하, 정규화속도p99비증가, root exact-zero와 baseline 재추론 일치를 요구한다.
유실된 metric 구현은 local-jump, scale-only reprojection, temporal boneCV와
native-frame velocity 수식으로 별도 정의했다. 과거 수치에 맞춰 사후 조정하지 않는다.

## Pilot 결과

8개sequence/25,928frames를평가했다. V2재추론은저장값과정확히일치했다.
Stride121+triangular의경계artifact는80.180%에서4.950%로줄어93.826%감소했다.
NMEp95/boneCVp95/speedp99비율은각1.001766/1.006171/0.868549였다.

그러나공식2D좌표가모두0인620frames(2.39%)가있어이번별도재구현에추가한
퇴화입력검사를통과하지못했다. 계산불가능한투영오차가포함된ratio만으로채택하지않는다.
현재선택정책은없고전체생성·V3분류실험은시작하지않았다.
유효성마스크를도입하는경우별도사후변경실험으로구분해야하며아직새결과는없다.

## 평가 단계와 한계

Pilot 통과 정책을 고정한 뒤에만497개 전체를 생성한다. 전체 구조 검산과 별도30개
geometry reference 비교를 통과해야 다음 분류 실험의 입력으로 사용한다.
Labels는 geometry 선택에 쓰지 않고 후속 학습 입력의 복사·정렬에만 사용한다.
통과 정책이 없거나 전체 geometry 기준에 미달하면 미채택으로 기록한다.

원본과 동일한 구현·수치의 복원이 아닌 별도 재구현이다. 의미적3D 정확도, 물리세계 이동,
FPS 차이, 낙상 성능 개선은 이 geometry 평가만으로 주장하지 않는다.

[V2 방법](2026-09-20_safer_v2_reconstruction_shared.md) ·
[전체 진행](2026-09-22_recovery_execution_shared.md) ·
[연구 맥락·참고문헌](2026-09-03_project_complete_summary_shared.md)
