# E07 — CAUCAFall 외부 진단

> 후속 검증: [현재 모델의 전체100개 평가](2026-10-03_cauca100_own_evaluation_shared.md)를 완료했다.
> 사건 F1은82.98%, 영상 정확도88.00%이며, 아래 과거 진단 수치와 구분한다.

- 문서 ID: `DOC-20260903-exp07-cauca-R1`
- 기준일: 2026-09-03
- 연구 상태: `completed` diagnostic, pristine test 아님

## 평가 역할

CAUCAFall은 RGB front-end와 skeleton domain shift를 반복 진단한 개발성 외부 데이터다. 같은 데이터에
여러 전처리와 모델을 적용했으므로 최종 봉인 성능으로 해석하지 않는다.

## 기존 기록 기준 결과

| 평가 | 결과 | 해석 |
| --- | --- | --- |
| Native 60-class ADL head | Top-1 26.667% | RGB-lifted skeleton의 ADL domain gap |
| 초기 E0 fall ranking | AUROC 53.880%, AUPRC 50.610% | 초기 전처리에서 분리력 부족 |
| corrected E1 ranking | AUROC 84.480%, AUPRC 85.917% | ranking은 회복, hard decision F1은 0% |

현재 RGB 경로의 100-sequence 진단:

| Model | Precision | Recall | F1 | AUROC | AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: |
| G0 | 92.453% | 98.000% | 95.146% | 97.840% | 98.482% |
| G2 | 95.833% | 92.000% | 93.878% | 96.600% | 97.251% |

## 해석과 한계

이 domain에서는 G0가 primary이고 G2는 precision이 높은 auxiliary다. 수치는 구현·domain 진단용이며
untouched external proof가 아니다. 최종 확증은 별도 봉인 데이터와 사전 등록된 threshold로 수행한다.
