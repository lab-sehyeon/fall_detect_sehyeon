# E04 — J0/J1 공동학습과 Final J1-V3

- 문서 ID: `DOC-20260903-exp04-joint-j0-j1-R2`
- 기준일: 2026-09-22
- 연구 상태: V1/V2 재구현 `completed`, V3 `paused`

## 현재 재구현 결과

[V1/V2 전체학습·평가·검산](2026-09-22_joint_reconstruction_shared.md)을완료했다.
새J1nestedF1은V1=93.373%,V2=92.683%다. V2는J0보다오탐과recall이함께줄었다.
V3는[geometry pilot](2026-09-22_safer_v3_reconstruction_shared.md)의퇴화입력검사미통과로
후속학습을시작하지않았다. 아래기존표는역사적결과이며새V1/V2결과와동일하지않다.

## 연구 질문과 구조

Frozen DSTE와 ADL head를 바꾸지 않으면서 SAFER 4-class와 FU binary supervision을 공동 활용해
intentional lying false positive를 줄일 수 있는지 평가했다.

J0는 dataset-specific linear heads이고, J1은 2048-D frozen feature에 zero-initialized residual
adapter를 추가한다. 초기 J1 출력은 J0와 동일하도록 설계했다.

## 기존 기록 기준 locked 결과

| Split | Metric | J0 | J1 | 변화 |
| --- | --- | ---: | ---: | ---: |
| SAFER Test | Macro-F1 | 70.477% | 73.105% | +2.628%p |
| SAFER Test | Fall F1 | 71.750% | 75.692% | +3.942%p |
| SAFER OOD | Macro-F1 | 53.487% | 54.512% | +1.025%p |
| SAFER OOD | Fall F1 | 46.275% | 50.297% | +4.021%p |

FU nested에서는 J0→J1 F1이 90.746→91.925%, lying FPR이 10.714→5.357%로 개선됐다. 다만
SAFER test/OOD fall recall은 각각 약 2.1/2.2%p 낮아졌다.

## 결정과 한계

- J1-V3를 skeleton reference로 채택했다.
- 이득은 precision과 F1 중심이며 recall trade-off를 함께 보고한다.
- 낙상 학습 전후 ADL logits의 exact 보존이 필수다.
- 기존 checkpoint는 재사용할 수 없으므로 동일 split과 고정 epoch 규칙으로 재학습한다.
