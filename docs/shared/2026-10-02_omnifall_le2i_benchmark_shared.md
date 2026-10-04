# OmniFall Le2i-CS 공식 시험 구간 비교

- 문서 ID: `DOC-20261002-omnifall-le2i-benchmark-R1`
- 기준일: 2026-10-02
- 상태: `completed`

공식 시험203구간에서 현재 고정 G0의 **Fall F1은53.33%, recall은36.36%**였다.
오경보는 없었지만 낙상22개 중14개를 놓쳤다. 그중9개는 자세 품질 기준 미통과,
5개는 품질을 통과한 뒤의 분류 오류였다. 동일 시험 구간을 사용한 논문과의 비교 및 독립 검산을 완료했다.

## 연구 질문과 평가 조건

현재 고정 모델을 OmniFall의 공식 Le2i-CS 시험 구간에 적용해 논문 보고값과 비교한다.
시험 대상은 피험자2/7의 **38영상·203동작 구간**으로, 낙상 동작22개와 나머지181개이다.
`fall`만 양성이고 낙상 후 누운 상태 `fallen`을 포함한 다른 클래스는 음성이다.
정답 구간마다 예측 하나를 생성하고 recall, specificity, Fall F1을 계산한다.
이는 정답 구간이 제공된 분류 평가이며 연속영상 사건검출이나 회복평가가 아니다.
[공식 시험 목록](https://huggingface.co/datasets/simplexsigil2/omnifall/blob/83572a37b9e3081df8c06a56874b1d1f2a19386c/splits/cs/le2i/test.csv),
[구간 정답](https://huggingface.co/datasets/simplexsigil2/omnifall/blob/83572a37b9e3081df8c06a56874b1d1f2a19386c/labels/le2i.csv),
[라벨 정의](https://huggingface.co/datasets/simplexsigil2/omnifall/blob/83572a37b9e3081df8c06a56874b1d1f2a19386c/LABELS.md).

## 예측 전에 고정한 적용 방법

공식 원본을 저자의 영상 변환 조건으로 처리한 뒤 각 구간을 공식 companion의 uniform64 모드로 읽는다.
YOLOv8x와 ViTPose-B로 자세를 얻고 기존 품질검사를 통과하면 MotionAGFormer, DSTE,
고정 J1 adapter와 별도 학습된 G0 분류기를 적용한다. 4클래스 argmax가 fall인 경우만 양성이다.
품질 기준은 검출·자세 coverage 각각0.8 이상, pelvis confidence 중앙값0.3 이상이며,
실패 구간도 제거하지 않고 무경보로 채점한다.
[저자 영상 변환](https://github.com/simplexsigil/omnifall-helper-scripts/blob/eb1443c1035ecdfbd15598dec16fae6aca71e7ea/video_conversion/le2i.sh),
[공식 companion](https://pypi.org/project/omnifall/0.2.0/).

종료시각이 영상 길이를 넘는 공식 구간은6개이며, 그중1개 비낙상 구간은 약0.301초,
나머지5개는 약0.224–12.0004밀리초 초과한다. 공식 로더의 마지막 프레임 반복 규칙을
그대로 적용하며, 구간 삭제나 정답시각 변경은 하지 않는다.
전체203구간 중121개에 반복 프레임이 있고, 구간당 고유 프레임은 최소8개다.
가장 가까운 원프레임을 고르는 특성상 일부 샘플은 GT 경계를 최대 약20밀리초 벗어난다.

J1은 SAFER/FU로, G0는 고정 특징에 대해 SAFER로 학습된 모델이다. Le2i를 통한 추가 학습,
임계값 보정, 모델 선택은 하지 않는다. G2의 움직임 특징은 물리적25fps를 가정하므로 구간을
64샘플로 시간 정규화하는 이번 주 비교에서는 사용하지 않는다. GT 구간과 offline lifting을
사용하므로 온라인 지연시간·실시간 성능으로 해석하지 않는다.

## 비교 범위와 한계

비교 기준은 [OmniFall v3 Table3](https://arxiv.org/html/2505.19889v3)의 Le2i Fall 열이다.
논문의 RGB 모델과 현재 skeleton 모델은 학습자료와 입력 표현·시간 표본화가 다르다.
따라서 시험 분할·정답·채점 지표를 맞춘 **시스템 비교**이며 동일 학습조건의 구조 우월성 검증이 아니다.
현재 모델의4클래스와 논문의16클래스 분류 문제도 다르므로 비교 지표는 Fall-vs-rest로 한정한다.
논문 수치는 보고값을 인용하며 비교 모델을 직접 다시 실행한 결과로 표시하지 않는다.
Le2i 학습을 포함하는 in-domain 결과는 외부 일반화 결과와 구분한다.
사전학습 데이터와 시험 데이터 사이의 완전한 중복 배제는 입증하지 않았다.

기존 127영상 사건검출 결과와 이번 203구간 분류 결과는 평가단위가 달라 직접적인 성능 증감으로
해석하지 않는다. Le2i가 과거 프로젝트 분석에도 사용됐으므로 완전히 처음 보는 blind test라는 주장도 하지 않는다.
원영상은 연구용 CC BY-NC-SA3.0 조건을 따르며 원본 재배포·비식별화 완료를 주장하지 않는다.

시험 구간에는 의도적으로 눕는 동작과 그 후 누운 상태의 정답 클래스가 없으며,
낙상 후 누운 상태는21개다. 따라서 이 시험 하나로 의도적인 눕기와 낙상의 구분 능력까지
검증했다고 해석하지 않는다. 양성은22개뿐이므로 한 건의 미탐 차이만으로도 recall이 약4.55%p 달라진다.

## 평가 결과와 해석

공식 시험 영상38개를 확보해 배포 파일의 무결성 기준과 대조했고, 저자의 영상 변환 조건 적용을 마쳤다.
203구간의 입력 프레임12,992장을 전수 대조했고, 고정 모델 평가와 별도 지표 검산을 완료했다.

| 모델 | 학습 조건 | Recall | Specificity | Fall F1 |
| --- | --- | ---: | ---: | ---: |
| 현재 DSTE + J1 + G0 | SAFER/FU adapter, SAFER G0; Le2i 추가 학습 없음 | 36.36% | 100.00% | 53.33% |
| Qwen3-VL-8B | Zero-shot | 9.1% | 99.4% | 16.0% |
| InternVL3.5-8B | Zero-shot | 27.3% | 98.3% | 38.7% |
| VideoMAE-K400 | CMDFall-CS | 77.3% | 97.8% | 79.1% |
| VideoMAE-K400 | OF-Synthetic | 59.1% | 100.0% | 74.3% |

첫 행만 이번 직접 실행·검산값이고, 나머지는 [OmniFall v3 Table3의 보고값](https://arxiv.org/html/2505.19889v3)이다.
현재 F1은 CMDFall로 학습한 VideoMAE보다 약25.77%p 낮다. Le2i 학습을 포함하는
OF-Staged와 Staged+Synthetic의 논문 F1은 각각100.0%,97.8%지만, 위 외부 일반화 비교와 구분한다.
정량표는 [비교 CSV](2026-10-02_omnifall_le2i_comparison.csv)로도 제공한다.

### 미탐과 품질 기준의 영향

전체 confusion count는 **TP8 / FP0 / FN14 / TN181**이다. Precision100%, binary accuracy93.10%,
AP59.81%이나, 높은 accuracy만으로 낙상 검출이 충분하다고 볼 수 없다.
최종 판정에 들어간176개 외에 품질 기준 미통과27개도 분모에 유지했다.

| 낙상22개에 대한 처리 | 개수 |
| --- | ---: |
| 품질 기준을 통과하고 낙상으로 분류 | 8 |
| 품질 기준을 통과했지만 비낙상으로 분류 | 5 |
| 품질 기준 미통과로 무경보 처리 | 9 |

품질 미통과 낙상9개 중8개는 사람 검출·자세 coverage 기준에 못 미쳤고,
나머지1개는 pelvis confidence만 기준 미달이었다. 따라서 현재 결과의 문제를 분류기 하나로만
설명할 수 없다. 같은 품질 기준을 유지하면 분류기로 전달되는 낙상은13개뿐이므로
전체 recall은 최대59.09%로 제한된다. 다만 gate를 완화하면 성능이 개선된다는 결론은 아직 검증하지 않았다.

비낙상181개 중18개도 품질 미통과로 무경보 처리됐다. Specificity100%는 이 처리까지 포함한
전체 시스템의 결과이지 모든 구간에서 자세 추정이 성공했다는 뜻이 아니다.

### 검증과 후속 범위

공식 분할·정답·입력 무결성, 자세 품질 계산, 가중치 불변성을 검사했다.
저장된 DSTE 특징에 대해 CPU에서 J1 adapter와 G0를 별도로 계산했으며176개 argmax가 모두 일치했다.
독립적인 지표 구현으로 confusion count와 보고 수치를 재계산해 일치함을 확인했다.
이는 전체 backbone의 독립 재실행이나 실환경 성능 보장은 아니다.

다음 연구 후보는 품질 미통과 원인 분석과 입력 시간축·문맥 비교다. 아직 실행하지 않았으며,
이번 시험 결과에 맞춰 품질 기준이나 임계값을 수정하면 별도 사후 실험으로 구분해야 한다.
기존127영상 사건검출 점수와 이번 구간 분류 점수를 직접적인 성능 증감으로 해석하지 않는다.

## 참고 자료

- Schneider 외, *OmniFall: From Staged Through Synthetic to Wild, A Unified Multi-Domain Dataset for Robust Fall Detection*, arXiv v3, 2026. [논문](https://arxiv.org/html/2505.19889v3).
- Charfi 외, *Optimized spatio-temporal descriptors for real-time fall detection: comparison of support vector machine and adaboost-based classification*, Journal of Electronic Imaging22(4), 041106, 2013. [원 데이터 논문](https://doi.org/10.1117/1.JEI.22.4.041106).
- 현재 모델의 학습 계보·구조·원 논문은 [방법론 상세](2026-09-29_paper_methodology_shared.md), 기존 사건검출 결과는 [127영상 Le2i 평가](2026-10-02_le2i_current_evaluation_shared.md), 같은 구간 적용 방법의 다른 데이터 결과는 [GMDCSA24-CS 평가](2026-10-02_omnifall_gmdcsa_benchmark_shared.md)에 정리했다.
