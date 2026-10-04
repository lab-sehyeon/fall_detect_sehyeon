# RGB와 재학습 모델 통합 검증

- 문서 ID: `DOC-20260928-rgb-integration-R2`
- 기준일: 2026-09-28
- 상태: `completed` — 고정3개 기술 연결 검사, 전체 외부 성능·회복 미평가

원영상에서 YOLOv8x·ViTPose-B로 사람과 관절을 추출하고 MotionAGFormer V3로 스켈레톤을 만든 뒤,
검증된 J1 및 G0/G1/G2로 연결하는 오프라인 추론을 재구현하고 기술 연결 검사를 완료했다.
사람 검출은 기존 C2 규칙에 따라 기본 검출이 없는 프레임에서만 저신뢰도 top1 후보로 보완한다.
스켈레톤과 131차원 전역 움직임은 최근 학습에 사용한 정의·정규화와 일치시킨다.
기존 모델을 다시 학습하거나 G2 미채택 결정을 변경하지 않는다.

OOPS의 고정된 이름순 첫3개 영상으로 기술 연결 검사를 수행했다. 기존 시간축 수정의
실제 관측 시각 기반25fps 정렬을 사용하며, 이는 과거 nearest-frame 방식과 구분한 별도 재구현이다.
품질 기준은 bbox/pose coverage 80% 이상과 pelvis 중앙 confidence 0.3 이상이다.
미통과 입력은 분류 결과 없이 남겼으며 결과를 보고 기준을 완화하지 않았다.

## 확인된 결과

| 영상 | 정렬 프레임 | bbox/pose coverage | pelvis 중앙 confidence | 처리 결과 |
| --- | ---: | ---: | ---: | --- |
| 고정 표본1 | 196 | 100.000% | 0.791 | 17개 window 추론·검산 완료 |
| 고정 표본2 | 302 | 76.821% | 0.760 | 품질 미달로 후속 분류 중단 |
| 고정 표본3 | 277 | 100.000% | 0.750 | 27개 window 추론·검산 완료 |

775프레임을 검사했으며 품질 통과2개 영상의44개 window에서 전체 추론 경로를 연결했다.
시간 정렬·추적 재생·3D overlap-add·131차원 독립 통계 재계산·CPU 분류기 비교를 통과했다.
모델 가중치와 원자료는 유지했고 추가 학습이나 임계값 조정은 하지 않았다.

통과한 두 영상에서 G0/G2의 fall-argmax window는 모두0개였다. 따라서 이번 연결 완료를
낙상 검출 성능의 성공으로 해석하면 안 된다. Event 정답·matching을 아직 구성하지 않았으므로
낙상 precision/recall/F1은 보고하지 않는다. G2 미채택 결정도 유지한다.

## 한계와 후속

이 검사는 기존 GMDCSA 실험의 복제나 외부 성능 평가가 아니다. 양방향 3D lifting과 sequence
정규화를 포함하므로 실시간 인과적 추론·지연 개선을 주장할 수 없다.
S0·회복 episode·전체 외부 event 평가는 이후 별도 단계이며 현재 성능 결과는 없다.
다음은 S0-A의16-class 상태 학습과 contextual recovery target 검증이다.

[학습된 모델과 한계](2026-09-28_global_motion_training_shared.md) ·
[전체 진행](2026-09-22_recovery_execution_shared.md) ·
[공식 ViTPose 구현](https://github.com/ViTAE-Transformer/ViTPose)
