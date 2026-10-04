# E06 — RGB dual path와 Global Motion 131-D

- 문서 ID: `DOC-20260903-exp06-rgb-global-R3`
- 기준일: 2026-09-28
- 연구 상태: 역사적 `completed`; 별도 재구현도 `completed`, 새 G2는 `not_selected`

## 최신 별도 재구현 결과

[RGB 후속 기술 검사](2026-09-28_rgb_integration_shared.md)도 완료했다. 고정3개775프레임 중
품질 통과2개44windows가 전체 경로를 완주했고1개는 coverage76.821%로 후속 분류를 중단했다.
G0/G2의 fall-argmax는 통과한 두 영상 모두0개였다. 이는 기술 연결 결과이며 외부 event 성능이나
상태·회복 평가 완료를 뜻하지 않는다.

[2026-09-28 정합·학습](2026-09-28_global_motion_training_shared.md)에서 세 모델을 각각
50epoch 학습하고 validation 선택·고정 test/OOD 평가·독립 검산을 완료했다.
G0/G1/G2의 validation Macro-F1은 각각 75.360/68.893/75.449%,
test는 77.168/68.067/77.329%, OOD는 56.436/47.553/56.436%다.
G2의 validation 이득 +0.089%p는 사전 기준 +0.5%p에 못 미쳐 미채택이다.
OOD fall recall도 G0 75.000%에서 G2 70.576%로 감소했다.
소실된 세부 설정을 실행 전에 명시한 별도 재구현이므로 아래 과거 결과와 동일하지 않다.
Global 학습 자체에는 RGB 통합이 포함되지 않았으며 후속 기술 검사는 위 문서에서 구분한다.

## 연구 질문과 방법

body-relative skeleton에서 약해질 수 있는 영상 내 전역 이동 정보를 원본 영상 좌표의 저차원 특징으로
보완했다. 12개 global channel의 level·velocity·vertical-acceleration 통계와 7개 quality 특징으로
131-D를 만들고, skeleton-only G0, global-only G1, 결합 G2를 동일 조건에서 비교했다.

## 기존 기록 기준 결과

| 평가 | G0 | G1 | G2 | G2−G0 |
| --- | ---: | ---: | ---: | ---: |
| SAFER Validation Macro-F1 | 67.012% | 50.607% | 69.242% | +2.230%p |
| SAFER Test Macro-F1 | 65.315% | — | 68.312% | +2.997%p |
| SAFER OOD Macro-F1 | 47.439% | — | 50.398% | +2.959%p |

G1은 skeleton을 대체하지 못했다. G2는 Macro-F1을 개선했지만 OOD fall recall은 3.909%p
감소했다. CAUCA와 Le2i에서는 G0가 더 높아 이득이 domain-dependent임을 확인했다.

## 결정과 한계 — 기존 기록 기준

- G0는 필수 skeleton fallback이다.
- G2는 배포 domain별 locked comparison을 거쳐 선택한다.
- 모든 domain에서 G2가 우월하다고 주장하지 않는다.
- 비식별 시스템에서는 global 특징을 edge에서 계산하고 원본 영상을 서버로 전송하지 않는 방식을
  우선한다.
