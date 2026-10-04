# E08 — OmniFall/OOPS front-end와 failure audit

- 문서 ID: `DOC-20260903-exp08-oops-R1`
- 기준일: 2026-09-03
- 연구 상태: `completed` diagnostic

## 연구 질문과 데이터

818 videos, 914 fall events와 299 recovery events에서 detector coverage, event decoder와 representation
실패를 분리했다. 모델 score와 event matching 계약을 유지한 matched comparison이다.

## Front-end 결과

조건부 detector fallback C2는 quality-pass videos를 592/818에서 685/818로 높였다.

| 경로 | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| 초기 B0+D0 | 76.820% | 43.873% | 55.850% |
| C2+D0 | 75.311% | 53.063% | 62.259% |

front-end 귀책 사건은 256개에서 145개로 감소했고, 병목은 downstream miss로 이동했다.

## D1 decoder 결과

| Decoder | Precision | Recall | F1 | TP/FP/FN |
| --- | ---: | ---: | ---: | --- |
| D0 | 75.311% | 53.063% | 62.259% | 485/159/429 |
| D1 | 54.893% | 67.505% | 60.550% | 617/507/297 |

D1은 fall alert와 recovery memory를 분리해 latch 귀책 miss를 146개에서 16개로 줄였다. Recall은
14.442%p 증가했지만 precision은 20.417%p 감소했다. D1은 최고 F1 선택이 아니라 구조 결함 수정이다.

## 잔여 실패와 한계

잔여 representation failure 137건은 최대 fall posterior가 모두 0.4912 이하였고 중앙값은
0.1586이었다. threshold나 front-end만의 문제가 아니므로 supervision과 표현 연구가 필요하다.
OOPS를 본 뒤 threshold를 조정해 같은 데이터에서 최종 성능으로 보고하지 않는다.
