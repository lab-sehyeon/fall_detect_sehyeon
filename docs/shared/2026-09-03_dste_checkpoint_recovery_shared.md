# DSTE 사전학습 백본 복구 현황

- 문서 ID: `DOC-20260903-dste-checkpoint-recovery-R3`
- 기준일: 2026-09-03
- 연구 상태: 구조 검증 `completed`, 성능 재현 `completed`

## 목적

낙상 인식 파이프라인의 기반 표현 모델을 다시 학습하기 전에, 공식 NTU RGB+D 60 cross-subject DSTE 사전학습 백본을 확보하고 현재 코드와의 호환성을 확인한다.

## 대상 모델

- 데이터셋: NTU RGB+D 60
- 프로토콜: cross-subject
- 입력 모달리티: 3D joint skeleton
- 백본: Dense Spatio-Temporal Encoder(DSTE)
- 입력 길이: 64 frames
- 출력 표현: temporal·spatial 특징 결합

## 진행 결과

- 공식 연구 저장소가 배포한 `ntu60_xs_joint_dste.pth.tar`를 확보했다.
- 파일 크기와 SHA-256이 공식 배포 메타데이터와 일치함을 확인했다.
- 체크포인트는 451 epoch의 사전학습 결과이며, state dict의 182개 항목이 모두 정상적인 텐서임을 확인했다.
- 현재 DSTE 구현에 백본 파라미터가 누락 없이 로드됐다.
- downstream 분류기 파라미터만 체크포인트에 없으며, 이는 linear evaluation에서 새로 학습하는 정상 구성이다.
- 64-frame NTU60 입력에 대한 최소 순전파가 성공했고 60-class 출력이 유한한 값으로 생성됐다.

## 현재 판정

공식 DSTE 사전학습 백본의 무결성과 구조 호환성 검증은 완료됐다. NTU60 56,578개 표본을
Cross-Subject train 40,091개와 validation 16,487개로 구성하고 downstream 입력 로딩도 확인했다.
제한 배치 linear evaluation에서는 60-class head만 학습되고 DSTE encoder가 변경되지 않았다.

기존 가공 입력의 전체 linear evaluation은 Top-1 79.899%, Top-5 94.899%로 재현 기준을 통과하지
못해 전처리 불일치 대조군으로 보존했다. 이후 원본 NTU60에서 공식 UmURL 입력을 재생성해 동일
계약으로 평가했다. 기존 기록의 epoch 150 결과 Top-1 85.285%, Top-5 97.234%를 재현했고 전체 구간
best는 Top-1 85.298%, Top-5 97.234%였다. encoder 불변성과 전체 validation 산출물도 검증했다.

## 다음 연구 단계

1. 재현된 DSTE encoder와 60-class ADL head를 고정한다.
2. 완료된 FU control에 이어 SAFER legacy F0A/F0B/F1과 corrected V2/V3를 순서대로 재현한다.
3. J0/J1 학습 전후 전체 validation ADL logits와 parameter 불변성을 검증한다.

## 참고

- [FoundSkelModel 공식 저장소](https://github.com/wengwanjiang/FoundSkelModel)
- [USDRL 공식 저장소](https://github.com/wengwanjiang/USDRL)
- [UmURL 공식 전처리](https://github.com/HuiGuanLab/UmURL/blob/main/data_gen/preprocess.py)
- [NTU RGB+D 공식 데이터셋](https://github.com/shahroudy/NTURGB-D)
