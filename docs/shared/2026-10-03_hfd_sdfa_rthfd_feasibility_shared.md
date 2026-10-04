# HFD·SDFA·RTHFD 원본 학습 모델 비교 준비

> 후속 확인: [2026-10-04 원본 모델 평가](2026-10-04_original_baseline_evaluation_shared.md)에서 RTHFD 230개 평가·검산을 완료했다. HFD/SDFA는 원본 학습 파일 미확보로 미평가다.

문서 ID: DOC-20261003-hfd-sdfa-rthfd-feasibility-R2  
갱신일: 2026-10-04  
상태: 검토 `completed`, 원본 학습 모델 확보·평가 준비 `in_progress`

## 평가 대상과 범위

**저자가 이미 학습한 원본 모델을 확보하여 추가 학습 없이 외부 데이터셋에서 평가한다.**
이전 검토의 HFD SVM 신규 학습·SDFA 재학습 제안은 현재 계획에서 제외한다.
학습 완료 모델의 가중치와 공식 입력·판정 처리를 유지하고 동일한 외부 영상·정답·채점 기준으로 비교한다.

| 모델 | 확보하여 사용할 대상 | 현재 확인 상태 |
| --- | --- | --- |
| HFD 3D-CNN+SVM | 사전학습 C3D와 저자가 학습한 SVM 파라미터 | 공식 코드와 C3D 링크 확인; 학습된 SVM 파일 미확보 |
| SDFA | 저자가 학습한 SDFA checkpoint 및 해당 평가 설정 | 공식 코드 확인; 실제 checkpoint 미확보 |
| RTHFD | 공개 MoveNet Thunder와 저자의 고정 판정 코드 | 공식 판정 코드 확인; 모델 파일 호환성·실행 확인 필요 |
| Ours | 기존 고정 DSTE→J1→G0 | 현재 완료된 가중치와 평가 경로 유지 |

[HFD 공식 코드](https://github.com/ekramalam/HFD_3DCNN) ·
[SDFA 공식 코드](https://github.com/saniazahan/SDFA) ·
[RTHFD 공식 코드](https://github.com/ekramalam/RTHFD)

논문에서 모델을 학습했다는 사실과 해당 학습 파일을 실제로 확보했다는 사실을 구분한다.
코드 공개만으로 전체 학습 모델을 확보했다고 표시하지 않는다. 현재 두 가중치의 미확보는
새 학습의 근거가 아니며 공식 배포 파일과 provenance 확보가 필요한 상태다.

## 평가 절차

1. 저자 원본 checkpoint·공식 추론 코드·학습 데이터 출처를 확인한다.
2. 모든 모델이 학습에 사용하지 않은 공통 외부 시험 목록을 확인한다.
3. 가중치·전처리·판정 조건을 고정하고 모델별 원래 입력 처리를 실행한다.
4. 동일 구간 정답과 event matching으로 채점하고 실패 영상도 전체 집계에 포함한다.

특히 HFD 원 논문은 CAUCA를 학습·시험에 사용한 실험도 있으므로 확보한 SVM이 어느 학습본인지
확인해야 한다. CAUCA 학습본이면 CAUCA에서 학습 미사용 외부 평가라고 부를 수 없다.
[HFD 논문](https://arxiv.org/html/2506.03193v1)

현재 사용자 모델의 전체 평가 사건 F1은 Le2i130개 **91.01%**, CAUCA100개 **82.98%**다.
HFD·SDFA·RTHFD의 새 외부 점수는 아직 측정하지 않았다.
기존 ST-GCN++·MS-G3D·1D-CNN의 완료 결과는 보존한다.

[Le2i 전체 평가](2026-10-03_le2i130_reevaluation_shared.md) ·
[CAUCA 전체 비교](2026-10-03_cauca100_comparison_shared.md)
