# S0-GT0 낙상 이후 회복 타깃 감사

- 문서 ID: `DOC-20260929-s0gt0-reconstruction-R2`
- 기준일: 2026-09-29
- 상태: `completed` — 학습·검증371개 sequence의 타깃 구성·검산 완료

## 확인된 표본

| 항목 | Train | Validation |
| --- | ---: | ---: |
| Sequences | 298 | 73 |
| Fall episodes | 2,405 | 520 |
| Contextual recovery episodes | 2,349 | 512 |
| Recovery frames | 114,435 | 25,635 |
| Post-fall frames | 206,315 | 32,993 |
| 일반 일어나기 hard-negative 구간 | 5,094 | 908 |

이 주요 표본수는 기존 기록과 일치했다. 모든8개 camera view에서 회복 표본이 존재하고,
학습·검증 subject의 중복은 없었다. 독립적인 두 타깃 생성 방식과 저장 결과가 일치했으며
원본 라벨을 변경하지 않았다. Test/OOD는 이 단계에서 사용하지 않았다.
Episode는 다중시점 sequence별 집계이므로 서로 독립된 물리적 낙상 사건 수와는 다르다.

이는 **회복을 학습할 문맥 라벨이 확보됐다는 근거**이며 회복 검출 성능을 입증한 결과가 아니다.
다음 S0-G0 endpoint 모델에서 타깃의 학습 가능성과 오경보·환경별 일반화를 평가해야 한다.

## 방법과 범위

단순한 일어나기와 낙상 이후 회복을 구분하기 위해 train/validation의 공식 라벨로 문맥 타깃을 구성한다.
상태는 background/falling/post_fall/recovering/ignore다. 낙상 이후 다음 낙상 전 첫 일어나기를 recovery로
정의하고, 그전 구간을post_fall로 둔다. 회복이 관측되지 않은 구간은ignore 처리한다.
낙상 이력이 없는 일어나기는background hard-negative로 남긴다.

원본 라벨은 수정하지 않는다. Test/OOD는 이 단계에서 사용하지 않고 subject 분리와 camera별
회복 표본을 검사한다. 이는 학습 annotation 구성이지 회복 검출 모델의 성능 결과가 아니며,
오프라인 GT 정의에서 사용한 미래 라벨을 실제 추론에 사용하지 않는다.

[기존 상태·회복 연구](2026-09-03_09_state_recovery_decoder_shared.md)의 타깃 정의를 기반으로 한
별도 재구현이다. 이 단계 이후 causal endpoint 회복 모델을 평가한다.
