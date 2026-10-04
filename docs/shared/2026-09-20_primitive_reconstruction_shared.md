# P0/P1: body-relative primitive 보완 실험

- 문서 ID: `DOC-20260920-primitive-reconstruction-R3`
- 기준일: 2026-09-20
- 상태: `completed` — P0 비교·미채택 및 P1 학습·독립 검산 완료

## 질문과 방법

학습된 skeleton 특징에 관절의 상대적인 자세·움직임을 더하면 낙상과 눕기를 더 잘 구분하는지
검사한다. 과거 실험의 구조는 유지하지만 소실된 세부는 새로 명시한 **별도 재구현**이다.
과거 성능과 수치적으로 같은 실험이라고 주장하지 않는다.

공통 신호에는 물리적 의미가 다른 데이터셋 간 root/world translation을 넣지 않는다.
torso·support 관계, 정규화된 관절 거리, knee/elbow 각도, shape covariance 고유값 비,
상대 관절속도·torso 각속도·초기 자세 대비 변화의12신호를 사용한다.
첫 유효 torso 방향과 유효 torso 길이의 중앙값을 기준으로 한다. 이 방향은 중력이 아니다.
영점/퇴화 자세는 신호0으로 보존하며 해당 경계를 가로지르는 속도도0이다.

P0는12신호의13통계로156차원 clip descriptor를 만든다. 평균·표준편차·최솟값·최댓값,
다섯 분위수, 처음·마지막·변화량·평균 절대 차분을 사용한다.
FU의 native30 DSTE temporal/spatial 특징과 동일한5 subject-disjoint folds에서
기존2048차원, primitive156차원, 결합2204차원을 비교한다.
primitive만 training fold 통계로 표준화한다. 개발 fold에서 최적 epoch를 고르므로
**development validation-OOF**이지 독립 일반화 성능은 아니다.
사전 point gate는 mean-fold F1 개선·lying 오탐 감소·fall TP 비감소의 동시 충족이다.
fold별 subject bootstrap10,000회로 paired 차이의 탐색적95%구간을 함께 보고한다.

P1은 기존 F1 특징 추출기를 동결하고 zero-init12→512 보정층과 복사한 classifier만 학습한다.
학습 전 모든 validation window에서 초기 출력이 F1과 정확히 같은지 검사한다.
SAFER training windows의1/4로10epochs, 초기 상태와2epoch 간격으로 검증하며
validation macro-F1→fall-vs-unstable AUPRC→lying F1 순으로 선택한다. 초기 모델도 선택 가능하다.
이번 feasibility에서는 test/OOD를 재평가하지 않는다.

## 상태와 한계

P0의3표현×5fold 비교와 P1의10epoch 학습·독립 검산이 모두 완료됐다.
P1은 validation 전체104,589개 window와 합친 timeline에서 초기 출력이 F1과 정확히 일치했다.

| 표현 | 개발 OOF F1% | AP% | fall TP/165 | lying FP/168 |
| --- | ---: | ---: | ---: | ---: |
| D0-TS | 95.575 | 98.802 | 162 | 12 |
| primitive only | 88.462 | 95.643 | 161 | 34 |
| D0-TS+primitive | 95.808 | 98.406 | 160 | 9 |

기존 D0-TS 비교군의 예측을 정확히 재현했다. 결합 특징은 mean-fold F1을0.304%p 높이고
lying 오탐을3개 줄였지만 fall TP를2개 잃었다. 따라서 사전 point gate를 통과하지 못해
P0 결합안은 **not_selected**다. mean-fold F1 차이의 paired subject bootstrap95%구간은
[−1.425,+2.008]%p로0을 포함한다. 소폭 F1 상승만으로 개선이 입증됐다고 해석하지 않는다.

P1은 초기 모델이 아니라 **epoch2**가 선택됐다.

| P1 validation | Macro-F1% | Fall F1% | Fall-vs-unstable AP% | Lying F1% |
| --- | ---: | ---: | ---: | ---: |
| 초기 F1과 동일 | 75.033 | 76.789 | 92.369 | 41.653 |
| P1 epoch2 | 75.313 | 77.419 | 93.309 | 41.819 |

Macro-F1 +0.280%p, fall F1 +0.630%p다. 843,768개 covered frames로 평가했으며
14,006개 uncovered frames는 제외했다. 원본 F1 동결 보존, 초기 동등성, 선택된 모델의 출력,
모든 validation의 겹침 평균·평가지표·최적 epoch 선택을 독립적으로 검산했다.
과거 기록의 best_epoch0과 다른 결과이며, 소실된 세부를 새로 고정한 재구현의 차이를 유지한다.
classifier도 함께 학습했으므로 개선을 primitive 자체의 효과로 분리할 수 없다.
test/OOD를 평가하지 않아 작은 validation 개선을 외부 일반화나 통계적 유의성으로 주장하지 않는다.
기존 F1을 자동 대체한 결과도 아니다. 다음 단계는 SAFER V2/V3 전처리와 matched controls다.

원장의 과거 수치를 이번 성능으로 인용하지 않는다.
통계·정규화에 관측 범위 전체를 사용하므로 causal/실시간 검증이 아니며, 짧은 window와
whole-clip 신호는 관측 범위가 다르다. proxy skeleton의 결손·단위·좌표계 차이가 남을 수 있다.
bootstrap은 재학습·epoch 선택을 반복하지 않으므로 독립적인 일반화 보장이 아니다.
큰 primitive 모델의 도입 여부는 이 결과만으로 결정하지 않는다.

관련 선행 연구와 전체 맥락은 [통합 연구 문서](2026-09-03_project_complete_summary_shared.md),
이전 특징 비교는 [D0/D1/D2 결과](2026-09-20_fu_probe_reconstruction_shared.md)를 참고한다.
