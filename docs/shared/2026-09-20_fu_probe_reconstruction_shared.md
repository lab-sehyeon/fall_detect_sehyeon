# FU D0/D1/D2 — 원본 특징과 SAFER adapter 특징 비교

- 문서 ID: `DOC-20260920-fu-probe-reconstruction-R2`
- 기준일: 2026-09-20
- 연구 상태: `completed` — 전체 개발 비교·독립 검산 완료

## 방법

동일 native30/aligned25 window 입력에서 DSTE temporal1024, temporal+spatial2048,
SAFER F1 hidden512, TS+hidden2560의 네 표현을 비교한다. DSTE와 ADL·F1은 동결한다.
각 window의 실제 시간 output을 temporal max 또는 hidden mean으로 집계하고 spatial max를
사용하며, clip별 feature는 window들의 동일 가중치 평균이다. Padding output은 시간 집계에서
제외한다. 이 pooling 세부는 원본에서 확인된 값이 아닌 사전 고정한 별도 재구현 조건이다.

FU993 clips를 subject-disjoint5fold로 나누고 binary linear heads를50epochs 학습한다.
새 고정 조건은 AdamW lr0.001, batch32, weight decay0.0001, 일정 학습률, scaler 없음이다.
Class weight는 train fold의 inverse-sqrt 빈도로만 계산하며, validation F1→AUPRC→accuracy와
동률 시 이전 epoch 규칙으로 선택한다. Clip label을 frame label로 복제하지 않는다.

## 검증과 해석 경계

이 결과는 epoch 선택에 사용한 fold들의 development validation-OOF다. Nested FU나 외부
일반화 성능으로 주장하지 않는다. 모델 불변성·sampling·clip 집계·저장 head/예측·선택·OOF
coverage의 독립 검산을 통과했다. 각 표현에서993개 clip이 정확히 한 번씩 OOF 평가에 포함됐다.

두 시간 처리 × 네 표현 × 5fold를 각각50epochs 학습했다. 아래는 제한 실행이 아닌 전체 결과다.

| Profile | 표현 | 개발 OOF F1 | AUPRC | Lying FP/168 |
| --- | --- | ---: | ---: | ---: |
| native30 | D0 temporal | 95.522% | 98.543% | 10 |
| native30 | D0 temporal+spatial | 95.575% | 98.802% | 12 |
| native30 | D1 adapter hidden | 91.765% | 96.722% | 17 |
| native30 | D2 concat | 95.575% | 98.739% | 12 |
| aligned25 | D0 temporal | 96.341% | 99.161% | 5 |
| aligned25 | D0 temporal+spatial | 96.970% | 99.051% | 5 |
| aligned25 | D1 adapter hidden | 93.051% | 96.913% | 12 |
| aligned25 | D2 concat | 95.522% | 98.554% | 10 |

두 조건 모두 adapter hidden 단독보다 DSTE-TS가 높았고 concat의 추가 이득은 확인되지 않았다.
이 결과만으로 통계적 동등성이나 외부 일반화를 주장하지 않는다. 역사적 모델·pooling·학습과
완전히 같은 실험은 아니며, 후속 primitive·공동학습·nested 검증과 분리한다.

[선행 ZS0/ZS1](2026-09-20_fu_zs_reconstruction_shared.md) ·
[전체 후속 범위](2026-09-20_post_f1_execution_contracts_shared.md)
