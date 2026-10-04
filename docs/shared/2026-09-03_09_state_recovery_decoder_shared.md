# E09 — S0 상태·회복 모델과 D1 event decoder

- 문서 ID: `DOC-20260903-exp09-state-recovery-R8`
- 기준일: 2026-09-30
- 연구 상태: `in_progress` — S0-E 완료·미채택, S0-F 진행; 문맥 회복·event decoder 잔여

## 현재 연구 진행

[S0-E](2026-09-30_s0e_reconstruction_shared.md) 72조건과고정평가를완료했으나채택후보는없다.
진단후보의validation Macro-F1 60.813%,lying_down F1 7.925%로기존 C를유지한다.
[S0-F](2026-09-30_s0f_reconstruction_shared.md)는35Dposterior기반보조모델로,
두가중치후보와최저camera-view우선선택을진행한다. 최종성능과후속문맥회복·D1평가는아직없다.

[S0-B/C 완료 분석](2026-09-29_s0bc_results_analysis_shared.md):B12epoch·C44조건·최종평가완료.
B는Test Macro-F1 68.783%,fall recall82.870%이나validation전환감소4.030%로단독미채택이다.
C는A+EMA0.5/연속3프레임/margin0.05선택이며Test SegmentF1@50 49.304%,
전환18.482/분(-81.589%)이다. 다만fall recall65.396%로A대비3.220%p낮아져
낙상경보안전성보장으로해석하지않는다. 학습·선택조건과별도실험계보는보존했다.
[GT0](2026-09-29_s0gt0_reconstruction_shared.md) train/val371개sequence의문맥타깃생성·검산은완료됐다.
Train2405fall/2349recovery,val520/512로기록과같은support를확인했으나회복모델성능은아직미검증이다.

[S0-A 상태 분류](2026-09-29_s0a_reconstruction_shared.md)의12epoch 학습·선택·고정 평가를 완료했다.
Sqrt epoch4의 Val/Test/OOD Macro-F1은66.417/64.685/36.440%다. 상태 구분 정보는 있지만
test SegmentF1@50 13.508%, 상태 전환/분100.382로 시간적 안정성이 부족하다.
상태 안정화 비교는 완료했고후속문맥회복·event평가는미완료다. 소실된 세부값을 사전 명시한 별도 실험이므로
아래 역사적 수치와 채택 판단을 현재 모델 결과로 간주하지 않는다.

## 연구 질문

window 낙상 점수를 안정된 시간 상태로 변환하고 낙상 이후 recovery를 별도 사건으로 검출할 수 있는지
평가했다.

## 기존 기록 기준 결과

| 실험 | 결과 | 결정 |
| --- | --- | --- |
| S0-A state linear | Val/Test/OOD Macro-F1 62.652/59.637/34.314% | 기준선 |
| S0-C EMA 0.70 | Test segment-F1 +19.467%p, edit +26.970%p, switches -68.73% | 상태 안정화 채택 |
| S0-E hard rule | lying-down F1 30.990→7.094% | `not_selected` |
| S0-F transition auxiliary | precision 2.535% | `not_selected` |

Recovery validation에서는 S0-G0 Precision 32.943%, Recall 82.422%, F1 47.072%를 얻었다. 후속
variant는 test/OOD recall 안전 조건을 충족하지 못했거나 untouched 평가가 없어 채택하지 않았다.

## 기존 기록의 결정과 한계

- 상태 안정화에는 S0-C를 사용한다.
- fall alert와 recovery memory를 분리하는 D1 구조를 유지한다.
- recovery head는 feasibility를 보였지만 외부 일반화가 충분하지 않아 배포 기능으로 주장하지 않는다.
- 상태 Macro-F1과 fall/recovery event metric을 별도로 보고한다.
