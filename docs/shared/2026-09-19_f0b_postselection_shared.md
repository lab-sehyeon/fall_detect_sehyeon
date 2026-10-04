# SAFER F0B 고정 모델의 test/OOD 평가

- 문서 ID: `DOC-20260919-f0b-postselection-R3`
- 기준일: 2026-09-20
- 연구 상태: `completed` — 전체 평가와 독립 검산 통과

## 연구 질문과 고정 방법

reconstructed V1의 validation으로 선택한 temporal+spatial F0B head가 test/OOD에서 어떤 성능을
보이는지 검증한다. Frozen DSTE와 ADL head, epoch46 F0B head를 유지하고 test 216,768 windows와
OOD 71,964 windows 전체를 평가한다. 4-class argmax와 기존 metric 정의를 사용하며 후보 재선택이나
threshold fitting을 수행하지 않는다.

## 검증된 결과

선택 모델·입력 계보·평가 구현의 동일성과 실행 준비 검증을 완료했다. 기존 validation 예측에서도
선택 모델의 Macro-F1 69.913%, fall F1 74.345%, conditional fall-vs-lie AUPRC 95.814%를 재확인했다.
Validation은 기존 결과 재검산이며, 아래 test/OOD는 이번 전체 평가 결과다.

| Split | Windows | Macro-F1 | Fall F1 | All-action AUPRC | Fall-vs-lie AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Test | 216,768 | 71.505% | 79.228% | 85.853% | 95.573% |
| OOD | 71,964 | 53.452% | 52.744% | 49.830% | 85.659% |

입력 순서·전체 평가 범위·유한 출력과 DSTE/ADL/선택 head 불변성을 확인했다. 저장된 예측에서
지표를 독립적으로 다시 계산해 일치를 확인했다. 새로운 학습·threshold fitting·후보 재선택은 없었다.

## 한계와 다음 연구

V1 원 mapper와 역사적 optimizer는 유실됐으므로 이 실험은 직전 명시적 recovery control의 후속
평가다. 삭제 전 원본 실험의 완전 복원이나 과거 대비 인과적 성능 개선으로 해석하지 않는다.
OOD에서 fall F1과 conditional AUPRC가 낮아지고 lying-down F1은 test 32.037%, OOD 16.974%로
제한적이다. 정확도만으로 실환경 일반화를 주장할 수 없다.

다음은 [F1 문서 기반 재현 계약](2026-09-20_f1_reconstruction_contract_shared.md)이다.
[E03 실험 기록](2026-09-03_03_safer_preprocessing_shared.md)과
[전체 재현 현황](2026-09-19_recovery_restart_shared.md)을 함께 따른다.
