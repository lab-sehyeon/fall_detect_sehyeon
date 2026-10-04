# E05 — LaDy·물리 특징 ablation

- 문서 ID: `DOC-20260903-exp05-lady-R1`
- 기준일: 2026-09-03
- 연구 상태: `not_selected`

## 연구 질문

관절 위치 표현에 속도, 가속도, 관절각과 회전 기반 물리 특징을 결합하면 SAFER/FU 낙상 판별을
안전하게 개선할 수 있는지 평가했다.

## 방법

초기 구현의 관절 순서·시간 미분·회전 표현을 감사한 뒤 H36M 공통 17관절, 42-DOF, sequence FPS,
SO(3), scale normalization과 결측 관절 mask를 적용해 다시 비교했다.

## 기존 기록 기준 결과

| Dataset | J1 F1 | LaDy F1 | 결합 F1 | 결합−J1 |
| --- | ---: | ---: | ---: | ---: |
| SAFER | 87.840% | 69.658% | 87.949% | +0.109%p |
| FU | 93.671% | 61.111% | 88.608% | -5.063%p |

FU 결합은 recall이 5.128%p 감소하고 lying FPR이 5%p 증가했다. SAFER의 작은 개선은 FU
hard-negative 안전성 손실을 상쇄하지 못했다.

## 결정

LaDy 특징은 공식 모델에 포함하지 않고 Frozen DSTE+J1-V3를 유지한다. 재검토하려면 SAFER 개선과
함께 FU recall 및 lying FPR 비열화를 동시에 만족해야 한다.
