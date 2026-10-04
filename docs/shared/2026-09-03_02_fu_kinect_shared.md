# E02 — FU-Kinect-Fall fall/intentional-lying 실험

- 문서 ID: `DOC-20260903-exp02-fu-kinect-R6`
- 기준일: 2026-09-20
- 연구 상태: 전처리·F0-FU·별도 ZS/D0/D1/D2·P0 재구현 `completed`, P0 미채택·J1 후속 대기

## 최신 별도 재구현

[P0 primitive 비교](2026-09-20_primitive_reconstruction_shared.md)도 전체 검산을 완료했다.
결합 F1은95.808%지만 fall TP가162→160으로 줄어 사전 채택 기준을 통과하지 못했다.
lying FP는12→9다. 과거 P0의 TP손실 없는 개선을 재현한 것으로 해석하지 않는다.

[ZS0/ZS1 전체 평가](2026-09-20_fu_zs_reconstruction_shared.md)와
[D0/D1/D2 전체 비교](2026-09-20_fu_probe_reconstruction_shared.md)의 독립 검산을 완료했다.
Native30 D0-TS/D1/D2 개발 OOF F1은95.575/91.765/95.575%다. 이는 validation으로 epoch를
선택한 개발 지표이며 아래 역사적 J1 nested 성능을 대체하지 않는다. 전체 파이프라인 복구 완료도 아니다.

## 연구 질문과 방법

Frozen DSTE가 native 3D Kinect 입력에서 falling과 intentional lying, bending, sitting, squatting을
구분할 수 있는지 평가했다. 품질 규칙을 통과한 993 clips와 21 subjects를 사용해 subject-disjoint
5-fold 및 outer subject를 선택에 사용하지 않는 nested protocol을 적용했다.

## 기존 기록 기준 결과

| 모델 | Precision | Recall | F1 | AUPRC | Lying FPR |
| --- | ---: | ---: | ---: | ---: | ---: |
| J0 | 89.412% | 92.121% | 90.746% | 97.429% | 10.714% |
| J1 | **94.268%** | 89.697% | **91.925%** | **97.581%** | **5.357%** |

J1은 fall true positive 4개를 잃는 대신 lying false positive를 18개에서 9개로 줄였다.

동일 frozen feature의 RBF-SVM은 F1 92.169%로 J1보다 0.243%p 높았다. J1을 최고 F1 모델로
주장하지 않으며, 장점은 높은 precision, 낮은 lying FPR과 SAFER와 공유하는 residual 표현이다.

## 해석과 한계

- J1은 recall 일부를 희생해 intentional-lying 오탐을 줄이는 보수적 specialist다.
- clip label은 onset, phase와 recovery supervision을 제공하지 않는다.
- FU는 반복 개발에 사용됐으므로 pristine external test가 아니다.

## 현재 F0-FU recovery control 결과

공식 배포 archive의 skeleton 1,006개를 전수 감사했다. 서로 다른 subject에 중복된 기록의 fold 누출을
차단하고, 같은 subject의 반복 사본과 공식 반복 범위를 벗어난 기록을 제외해 기존 기록과 같은 993개
품질 집합을 재구성했다. falling 165개와 intentional lying 168개를 포함하며 21 subjects가 모두
존재한다.

Kinect-v1 20 joints를 deterministic proxy NTU25로 변환하고 zero-frame 복구, SpineMid 중심화와
첫-frame shoulder 정렬을 적용했다. `(subject-1) mod 5`의 subject-disjoint 5 folds는 237, 188,
189, 189, 190 clips이며 각 fold의 train/validation subject overlap은 없다. 전처리 산출물의 shape,
finite 좌표, class 수와 fold 무결성 검사를 통과했다.

고정된 ADL DSTE 위에서 FU binary linear head만 학습하는 5-fold control을 완료했다.

| 평가 | 결과 |
| --- | ---: |
| Precision | 93.491% |
| Recall | 95.758% |
| F1 | 94.611% |
| AUPRC | 98.620% |
| Accuracy | 98.187% |
| Confusion | TP 158 / FP 11 / FN 7 / TN 817 |

993개 표본은 OOF 평가에 중복 없이 정확히 한 번 포함됐고, subject-disjoint split, frozen encoder와
linear-head-only 학습 조건을 모두 통과했다. 오탐 11건은 모두 intentional lying에서 발생해 lying
FPR은 6.548%였으며 walking, bending, sitting과 squatting의 오탐은 0건이었다. Frozen DSTE 표현에서
FU 낙상 분리 신호가 확인됐지만 intentional lying이 핵심 hard negative라는 결론은 유지한다.

과거 F0의 OOF F1 약 92.877%, AUPRC 98.357%와 유사한 범위지만, 과거 trainer 원문과 전체 학습 설정이
남아 있지 않아 현재 결과를 byte-identical 역사 재현이나 성능 개선으로 주장하지 않는다. 구조·데이터
split 계약을 복구한 별도 control로 사용한다.

## 다음 연구

SAFER 원본과 split 감사는 완료됐다. 기존 실험 순서에 따라 legacy `clean3d_v1`, F0A/F0B/F1과
corrected V2/V3를 차례로 복구한다. 그 뒤에만 동일 frozen feature와 사전 고정된 평가 계약으로
J0/J1을 순서대로 재현한다.
