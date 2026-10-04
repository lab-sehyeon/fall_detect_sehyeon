# 전체 연구 재구현 진행

- 문서 ID: `DOC-20260922-recovery-execution-R16`
- 기준일: 2026-09-30
- 상태: `in_progress` — S0-E 완료·미채택, S0-F 진행; 문맥 회복·외부 평가 잔여

## 최신 — S0-E 완료와 S0-F 진행

[S0-E](2026-09-30_s0e_reconstruction_shared.md) 72조건 비교와고정평가를완료했다.
8개기준을모두충족한후보는없어미채택이다. 진단후보의validation Macro-F1은60.813%,
lying_down F1은7.925%이며기존 C를유지한다.
[S0-F](2026-09-30_s0f_reconstruction_shared.md)는고정posterior35차원의전이보조모델이다.
누락조건을별도실험에사전명시하고cap20/50의12epoch비교와worst-view우선선택을진행한다.
아직최종성능결과는없다. 이후문맥회복모델·D1·외부event평가는남아있다.

## S0-B/C 완료 결과

[S0-B/C 완료 분석](2026-09-29_s0bc_results_analysis_shared.md):B12epoch와C44조건의선택·
고정평가·저장예측검산을완료했다. B는epoch7선택,Test Macro-F1 68.783%,fall recall82.870%이나
validation전환감소4.030%로단독미채택이다. C는A+EMA0.5/연속3프레임/margin0.05선택으로
Test SegmentF1@50 49.304%,전환18.482/분(-81.589%)이다. 다만fall recall65.396%로A대비
3.220%p하락해상태안정화기준통과를낙상경보안전성보장으로주장하지않는다.
실질batch1024/FP32/선택계약은유지했으며단계적실행계보를구분한다. 원본bit-exact동등성주장없음.
[GT0](2026-09-29_s0gt0_reconstruction_shared.md)는train/validation371개sequence에서타깃을완성했다.
Train2405개낙상중2349개,val520개중512개가회복으로연결되며독립검산과subject분리가통과했다.
이는회복모델학습을위한자료준비이며회복감지성능복원으로해석하지않는다.

## S0-A 완료 결과

[S0-A 상태 분류](2026-09-29_s0a_reconstruction_shared.md) 두 후보의12epoch 학습과 선택 이후
test/OOD 평가·저장 예측 검산을 완료했다. Sqrt epoch4의 Test/OOD Macro-F1은64.685/36.440%,
SegmentF1@50은13.508/6.505%다. 기존 모델은 유지하며 S0-A는 기준선으로만 확보한 상태다.
지속 상태·시간적 안정성과 OOD 일반화가 부족해 운영 모델이나 회복 감지 완료를 주장하지 않는다.
S0-B/C의최종판정은위완료분석에기록했다. S0-E는미채택으로완료했고S0-F는진행중이다.
문맥기반회복모델·D1과외부event평가는남아있다.

## 앞선 완료 결과

[RGB 기술 연결 검사](2026-09-28_rgb_integration_shared.md)에서 고정3개 영상775프레임을 처리했다.
2개44windows는 전체 추론·저장값 검산을 통과했고1개는 coverage76.821%로 후속 분류를 중단했다.
G0/G2의 fall-argmax window는 통과2개 모두0개였으며, 연결 완료를 낙상 성능 복원으로 해석하지 않는다.
전체 외부 event 평가와 S0/회복은 아직 미완료다.

[V3 R3](2026-09-22_v3_valid_support_shared.md)를 별도 post-hoc 실험으로 완료했다.
이전 미채택 결과는 유지한다. 원본 입력·모델·선택 기준은 그대로 두고 정의 가능한 2D 근거에서만
재투영·seam을 집계하며 전체 시간축의 3D 검사는 유지했다. 전체497개 전처리 검산과 후속controls·
공동학습·기존FU분류기대비고정예측비교가완료됐다.
[OOPS818·SAFER OOD30 자료 확보](2026-09-23_rgb_data_acquisition_shared.md)도 완료했다.
[입력 검사](2026-09-28_rgb_data_preparation_shared.md)에서 OOPS 전체 프레임 읽기와 라벨 정규화,
SAFER 전체497개 global source 확인을 마쳤다. 후속[정합·학습](2026-09-28_global_motion_training_shared.md)에서
별도 시간축·좌표 정합과 네 split 전체 1,007,723개 입력 검사를 완료했다.
G0/G1/G2 각 50epoch 학습·validation 선택·고정 test/OOD 평가·독립 검산도 완료했다.
G2의 validation Macro-F1 이득 +0.089%p는 기준 +0.5%p에 못 미쳐 미채택이다.
G0/G2의 test fall F1은 79.560/79.296%, OOD는 53.018/53.117%다.
OOD fall recall은 75.000/70.576%로 감소했으며 G2의 보편적 우위를 주장하지 않는다.

## 확인된 완료 범위

NTU60 ADL, FU F0, SAFER V1 controls·F1, FU ZS/probe, P0/P1에 이어 V2 전처리의 전체 생성과
독립 구조 검산이 완료됐다. V2는497 sequences/8,091,357 frames와1,007,723개 windows다.
Split별609,183/106,680/219,896/71,964개가 원본 sequence slice와 정확히 일치했다.
이는 3D 의미적 정확도나 낙상 성능 검증과는 구분한다. P0는 완료했으나 미채택이다.
모든 새 결과는 소실된 원본과 동일하다고 주장하지 않는 문서 기반 별도 재구현이다.

## 남은 연구

| 단계 | 내용 | 상태 |
| --- | --- | --- |
| V2 controls | 전체 완료, 고정 test/OOD fall F1 79.652/54.206% | `completed` |
| V1/V2 공동학습 | 두 계보 전체 학습·고정/nested 평가·최종 학습·독립 검산 완료 | `completed` |
| V3 이전 pilot | 원본 좌표0 프레임620개로 퇴화 입력 검사 미통과, 결과 보존 | `not_selected` |
| V3 R3 pilot | 유효 근거의 경계 이상 비율95.863% 감소, 고정 기준 통과 | `completed` |
| V3 전체 | 정책 고정 후497개 생성·전체 검산 완료 | `completed` |
| V3 후속 | matched controls·공동학습·독립 검산 완료 | `completed` |
| FU 분류기 비교 | 4종 nested·V3 고정 reference 비교 완료 | `completed` |
| 나머지 추가 비교 | LaDy·표현 baseline·NTU binary sanity의 누락 실험 | `planned` |
| RGB 자료 확보 | OOPS818·SAFER OOD30 확보·검증 | `completed` |
| RGB 입력 정합 | OOPS818개 시간축·SAFER30개 공식 resized pose공간 정합 수정·검사 | `completed` |
| Global G0/G1/G2 | 전체 입력·각50epoch·고정 평가 검증 완료, G2 사전 기준 미달 | `completed`, G2 `not_selected` |
| RGB end-to-end | 고정3개 기술 검사,2개44windows 전체 연결·검산,1개 품질 거부 | `completed` engineering smoke |
| S0-A/B/C·GT0 | 상태모델 비교·선택·평가와문맥타깃검산 완료 | `completed` |
| S0-E | 72조건 완료,채택가능후보0 | `not_selected` |
| S0-F | 전이보조모델 두후보의사전조건고정·실험진행 | `in_progress` |
| 회복·외부 | S0-G0/G1/G2·D1·모델 외부평가 | `planned` |
| VLM 관련 | 역사적 standalone 범위와 계획 단계 통합을 구분 | `planned` |

각 단계는 선행 검증을 통과한 결과만 사용한다. V3는 과거의 선택 정책을 바로 채택하지 않고
label-blind geometry 검증을 재실행한다. Development OOF와 nested 평가를 구분하고,
외부 진단을 sealed 성능으로 재해석하지 않는다. 기존의 미채택·negative result도 보존한다.

V1/V2 J1의 새 FU nested F1은93.373/92.683%다. V2는J0보다오탐과recall이함께줄었다.
FU4분류기 중 RBF-SVM의nestedF1은93.617%이며,완료된V3J0/J1은92.771/93.333%다.
이전V3 R2는유효성검사에서미채택됐고,별도사후변경R3는전체검산을통과했다.
전체후반연구가완료된것은아니다.

[현재 V2 controls](2026-09-22_safer_v2_controls_shared.md) ·
[V1/V2 공동학습](2026-09-22_joint_reconstruction_shared.md) ·
[V3 geometry](2026-09-22_safer_v3_reconstruction_shared.md) ·
[V3 controls](2026-09-22_safer_v3_controls_shared.md) ·
[V3 공동학습](2026-09-22_joint_v3_reconstruction_shared.md) ·
[FU 분류기 비교](2026-09-22_fu_classifier_reconstruction_shared.md) ·
[연구 통합본과 참고문헌](2026-09-03_project_complete_summary_shared.md)
