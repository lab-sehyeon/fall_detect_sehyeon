# SAFER V2: native timeline 3D 전처리 재구현

- 문서 ID: `DOC-20260920-safer-v2-reconstruction-R3`
- 기준일: 2026-09-22
- 상태: `completed` — 전체 생성·독립 구조 검산 완료

## 목적과 방법

기존 3D의 시간 대응·관절 배치 문제를 교정하기 위해 공식 COCO17 2D에서 3D를 다시 추론한다.
과거 문서로 확인되는 조건은 유지하고 소실된 세부는 실행 전에 고정한 별도 재구현이다.
삭제 전 원본과 수치적으로 동일한 복원을 주장하지 않는다.

MotionAGFormer-B에243프레임 연속 구간을 stride243으로 입력한다. 마지막 실제 frame까지
덮는 구간을 추가하고, 짧은 sequence만 마지막 자세로 padding한다. 실제 구간 출력만 보존하고
동일 frame의 중복 출력은 균등 평균한다. Flip 추론을 평균하며 pelvis를0으로 유지한다.
이후 H36M17→NTU25 proxy mapping, sequence torso 길이 정규화·SpineMid 중심화·어깨 yaw
정렬을 적용한다. 물리적 세계 이동이나 native Kinect 3D로 해석하지 않는다.

대상은497 sequences/8,091,357 frames다. Confidence의 비유한 값만0으로 바꾸며 좌표와 label은
임의 보간하지 않는다. 기존 legacy3D는 입력이나 frame 제외 기준으로 사용하지 않는다.
공식 SAFER batch의 confidence 순서 정책은 유지한다. XY만 H36M 순서로 바뀌는 해당 정책의
관절-confidence 의미 불일치는 남은 한계이며 추가 교정의 효과를 이번 결과에 섞지 않는다.

공식 subject split과 기존 validation subjects를 유지해64프레임/stride8 windows를 만든다.
완료된 train/val/test/OOD는609,183/106,680/219,896/71,964개다.
Label은 동일성 검사와 저장에만 사용하며 전처리 선택을 위한 성능 계산은 하지 않는다.

## 검증과 다음 단계

좌표·timeline coverage·padding·좌우 flip·mapping·중심화·source label identity와
학습/검증 subject 분리를 검사한다. 구조 검증 통과는 3D 의미적 정확도나 낙상 성능 검증이 아니다.
입력 전수 검사는 통과했다. 497 sequences/8,091,357 frames, 예정 lifting33,570 windows와
split별 window 수가 확인됐고 confidence 보정 대상은193 sequences/525 frames/8,925값이다.
전체497 sequences/8,091,357 frames의 생성·독립 구조 검산이 완료됐다. Coverage는1..2이며
중복 평균·관절 변환·label identity·동결 모델 불변성을 확인했다.
합1,007,723개 입력 windows 모두 원본 sequence slice와 정확히 일치했다.
[F0A/F0B 비교](2026-09-22_safer_v2_controls_shared.md)와 V3 label-blind geometry
pilot로 이어가며, 과거 V3 정책을 확인만 하고 검증 없이 채택하지 않는다.

## 근거 자료

- [SAFER 공식 3D lifting 설명](https://github.com/safer-activities/Safer-Activities/tree/main/preprocessing/3d_pose_lifting)
- [SAFER 공식 batch 구현](https://github.com/safer-activities/Safer-Activities/blob/994ed688ce9e491245ee96c1665e948c6ce6c74d/preprocessing/3d_pose_lifting/batch_3d_pose_processor.py)
- [MotionAGFormer 논문](https://openaccess.thecvf.com/content/WACV2024/html/Mehraban_MotionAGFormer_Enhancing_3D_Human_Pose_Estimation_With_a_Transformer-GCNFormer_Network_WACV_2024_paper.html) · [공식 구현](https://github.com/TaatiTeam/MotionAGFormer)
- [전체 연구 맥락과 참고자료](2026-09-03_project_complete_summary_shared.md)
