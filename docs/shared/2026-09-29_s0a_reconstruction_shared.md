# S0-A 프레임 단위 상태 분류 실험

- 문서 ID: `DOC-20260929-s0a-reconstruction-R2`
- 기준일: 2026-09-29
- 상태: `completed` — 학습·선택·고정 평가 완료, 연속 상태 출력의 운영 채택 근거는 부족

## 결과와 판정

두 후보를 각각12epoch 학습한 후 validation 기준으로 **sqrt inverse-frequency CE의 epoch4**를
선택했다. 선택 이후 test/OOD 평가와 저장 예측의 지표 재검산을 완료했다. 기존 동결 모델은 유지됐다.

| 지표 | Validation | Test | OOD |
| --- | ---: | ---: | ---: |
| Frame Macro-F1-16 (%) | 66.417 | 64.685 | 36.440 |
| Fall frame F1 (%) | 70.727 | 74.712 | 51.370 |
| Fall frame recall (%) | 65.488 | 68.616 | 64.311 |
| lie_down: 눕는 동작 F1 (%) | 81.979 | 85.575 | 52.086 |
| lying_down: 누운 상태 F1 (%) | 38.145 | 32.644 | 18.114 |
| Getting_up frame F1 (%) | 73.350 | 71.731 | 53.138 |
| Segment F1@50 (%) | 12.981 | 13.508 | 6.505 |
| Edit (%) | 19.202 | 18.778 | 9.827 |
| 예측 상태 전환/분 | 106.522 | 100.382 | 154.741 |
| 정답 상태 전환/분 | 15.844 | 15.131 | 20.279 |

**상태 정보가 있는 기준 모델은 확보했지만, 시간적 일관성과 도메인 일반화는 부족하다.**
Test 예측 구간 수는 정답의 약6.60배, OOD는 약7.61배로 행동 구간이 과도하게 잘게 나뉜다.
Test의 누운 상태 정답 중45.38%를 '눕는 동작'으로 혼동했으며 OOD에서는46.59%다.
상태 전환 횟수는16개 상태 전체의 변동이지 낙상 오경보 횟수가 아니다.

Test에서 직접적인 낙상→누운 상태 전이는0/8, 누운 상태→일어나기는0/16을 검출했다.
이 수치는 ±12프레임 이내 direct-adjacency 평가이며 전체 회복 사건의 검출률을 뜻하지 않는다.
Getting_up 분류 성능을 낙상 후 회복 감지 성능으로 해석할 수 없다.

기존 기록보다 test Macro-F1은59.637→64.685%, OOD는34.314→36.440%로 높지만,
test fall recall은76.031→68.616%로 낮다. 재생성 좌표·학습 설정·세부 평가 조건 차이가 있어
과거 모델보다 개선됐다는 통제 비교나 원본 성능의 완전 복원으로 주장하지 않는다.
단일 seed 결과이며 통계적 유의성은 미검증이다. Frame 지표는 외부 event 경보 성능이 아니고,
미래를 포함하는 window를 사용하므로 온라인 causal 감지 성능도 입증하지 않는다.

## 방법

낙상 사건 분류와 구분되는 16개 행동 상태를 평가한다. 동결된 DSTE의 temporal 특징과
유효 관절의 spatial 평균을 결합하고 선형 분류기만 학습한다. 기존 ADL·낙상 모델은 변경하지 않는다.

SAFER의 원본16개 frame label을 사용하며 plain CE와 sqrt inverse-frequency CE를 12epoch 비교한다.
Train 빈도로만 class weight를 산출한다. Validation Macro-F1→SegmentF1@50→Edit 순으로 선택하고,
선택 후 test/OOD를 평가한다. 겹친 window의 raw logits를 평균하며 smoothing은 적용하지 않는다.

소실된 학습 설정은 SGD lr0.006, momentum0.9, batch128, seed0으로 사전에 명시한 별도 실험이다.
원본 실험이나 과거 성능과 동일하다고 주장하지 않는다. Frame/class/segment/edit/switch 지표와
낙상→누운 상태, 앉기→앉은 상태, 누운 상태→일어나기의 직접 전이를 함께 기록한다.
전이 평가는 ±12프레임 matching을 새로 명시한 보조 지표로, 과거 전이 수치와 직접 비교하지 않는다.

학습609,183개·검증106,680개 window를 사용하고 선택 이후 test219,896개·OOD71,964개를 평가했다.
평가 대상199개 sequence의 covered3,199,464frames를 검산했다. 끝의 미포함716frames는 기존
64frame/stride8 계약에서 생긴 tail이며 임의 라벨 수정이나 추가 padding은 하지 않았다.

## 다음 연구

선택 S0-A를 고정한 S0-B temporal correction, 이후 S0-C 상태 안정화 비교를 기존 순서대로 수행한다.
회복 검출은 낙상 이후 문맥을 포함하는 별도 GT0 target으로 평가해야 한다. 후속 실험은 아직
실행하지 않았으며 현재 테스트 결과에 맞춰 threshold나 선택 기준을 바꾸지 않는다.

## 근거

- [SAFER-Activities 공식 coarse label 정의](https://github.com/safer-activities/SAFER-Activities/blob/994ed688ce9e491245ee96c1665e948c6ce6c74d/preprocessing/mappings.json)
- [MS-TCN 공식 segment F1·Edit 평가](https://github.com/yabufarha/ms-tcn/blob/0e418c029c2de1e90f6c54f45a0c186d8d9977b0/eval.py)
- [기존 S0·회복 연구 기록](2026-09-03_09_state_recovery_decoder_shared.md)
- [전체 연구 및 참고문헌](2026-09-03_project_complete_summary_shared.md)
