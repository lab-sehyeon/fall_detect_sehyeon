# Skeleton-Based Fall Detection

FoundSkelModel의 skeleton representation을 기반으로 일상행동 인식 성능을 보존하면서 낙상과
intentional lying을 구분하는 낙상·이상행동 인식 연구 프로젝트입니다.

이 저장소는 원본 FoundSkelModel을 그대로 배포하는 저장소가 아닙니다. 현재 낙상 분류 구조는
**Frozen DSTE → J1 residual adapter → G0 분류기**입니다. 전역 움직임 및 G1·G2와 상태·회복
보조 모델은 현재 활성 경로에서 제외하고, 기존 비교 실험으로 보존합니다.

## 공개 범위

이 저장소는 연구 코드, 설정, 테스트와 공유용 결과 문서를 제공합니다. 데이터셋, 모델 가중치,
외부 저장소 사본, 내부 실험 기록과 데이터셋 영상에서 추출한 사례 이미지는 포함하지 않습니다.
실행에는 해당 자산을 별도로 확보해야 하며, 일부 실험 도구는 내부 manifest·실행 기록에도
의존하므로 이 저장소만으로 모든 과거 실험을 즉시 재실행할 수 있는 배포본은 아닙니다.

공개 사본에서는 서버별 절대경로를 상대경로·현재 Python interpreter로 바꾸었습니다.
연구 모델 설정과 보고된 결과 수치는 보존했습니다. 외부 구성요소의 라이선스 적용 범위는
[외부 라이선스 고지](THIRD_PARTY_NOTICES.md)를 확인하세요.

## 주요 구성

- **Frozen DSTE**: 사전학습된 skeleton 표현 추출기 고정
- **J1 adapter**: SAFER와 FU-Kinect로 학습한 낙상용 특징 보정
- **G0 classifier**: J1 특징 위에 SAFER로 학습한 4-class 분류기
- **Fall event decoder**: 연속 fall 예측의 시작을 낙상 경보로 변환
- **기존 연구 보존**: ADL, J0, global motion 및 상태·회복 비교 실험은 별도 유지
- **Privacy-first design**: edge에서 skeleton과 선택적 audio feature를 생성하고 서버에는 원본 영상을
  전달하지 않는 구조를 연구

```text
RGB video
  → person detection / pose estimation
  → whole-video quality gate
  → 3D skeleton
  → Frozen DSTE
  → J1 adapter
  → G0 (other / fall / lie_down / lying_down)
  → fall event
```

현재 구조 선택은 [active model config](configs/active_fall_model.json)에 명시했습니다.
새 평가는 [J1+G0 전용 실행 도구](scripts/run_j1_g0_evaluation_20261002.py)를 사용하며,
이전 실험 스크립트의 동작은 소급 변경하지 않습니다. 현재 평가에는 영상 전체 품질 검사와
비인과적3D 처리가 포함되므로 실시간·엣지 배포 검증을 완료한 구조는 아닙니다.

## 현재 상태

| 항목 | 상태 |
| --- | --- |
| 공식 NTU60 XSub 입력 생성·검증 | 완료 |
| 공식 DSTE checkpoint 구조·순전파 검증 | 완료 |
| SAFER skeleton 최소 자료 확보·무결성 검사 | 완료 |
| NTU60 ADL 분류기 학습 | 150epoch 완료; backbone 고정 |
| FU/SAFER J0·J1 재학습 | 문서 기반 재구현 완료 |
| G0 분류기 학습 | 50epoch 완료, validation 선택5epoch 사용 |
| J1+G0 단독 구조 평가 | Le2i38/127·URFD70 재평가 완료;5555창 기존 G0와 동일 |
| Global 및 상태·회복 보조 경로 | 현재 낙상 경로에 미사용; 기존 연구 보존 |

삭제 전 **역사적 결과**와 문서 기반 재구현의 **새 검증 결과**는 구분합니다. 모델 가중치가
소실 전 원본과 동일하다는 주장은 하지 않습니다. 현재 구조의 조건과 결과는
[J1+G0 단독 평가 문서](docs/shared/2026-10-02_j1_g0_only_shared.md)에 정리합니다.

## 문서

- [J1–G0 Related Work·사용 논문·BibTeX](docs/shared/2026-10-03_j1_g0_related_work_shared.md)
- [Git 협업 및 기여 가이드](CONTRIBUTING.md)
- [현재 J1+G0 구조와 평가](docs/shared/2026-10-02_j1_g0_only_shared.md)
- [프로젝트 전체 공유 통합본](docs/shared/2026-09-03_project_complete_summary_shared.md)
- [실험별 문서 목록](docs/shared/README.md)
- [파이프라인 재학습 순서](docs/shared/2026-09-03_pipeline_retraining_recovery_shared.md)
- [데이터 및 모델 확보 계획](docs/shared/2026-09-03_next_steps_dataset_download_shared.md)
- [연구 문서 작성 규칙](docs/documentation_rules.md)

## 데이터와 개인정보 보호

데이터셋, checkpoint와 생성된 feature는 저장소에 포함하지 않습니다. NTU RGB+D, FU-Kinect-Fall,
SAFER-Activities 등 각 데이터셋의 이용 조건과 라이선스를 별도로 따라야 합니다.

운영 목표는 RGB와 audio를 edge에서 처리하고 서버에는 skeleton, 점수, 상태 전이와 제한된 event
metadata만 전달하는 것입니다. 이 비식별 구조는 설계·검증 중이며 완전한 익명성을 보장하는 것으로
간주하지 않습니다.

## Upstream 및 참고 연구

본 프로젝트는 [FoundSkelModel 공식 구현](https://github.com/wengwanjiang/FoundSkelModel)과
[Foundation Model for Skeleton-Based Human Action Understanding](https://arxiv.org/abs/2508.12586)을
기반으로 합니다. YOLOv8, ViTPose, MotionAGFormer와 데이터셋을 포함한 전체 참고문헌은
[공유 통합본의 참고문헌](docs/shared/2026-09-03_project_complete_summary_shared.md#27-전체-참고문헌)을
확인해주세요.

Upstream 코드와 논문을 프로젝트 자체 기여로 주장하지 않으며, 프로젝트 확장과 원본 구현의 출처를
구분합니다.

## License

기반인 FoundSkelModel은 **Apache-2.0**으로 배포되며, 이 저장소는 원본
[Apache License 2.0](LICENSE)을 보존합니다. 외부 구성요소의 출처, 확인한 코드 라이선스와
재배포 시 고려할 사항은 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)에 정리했습니다.

RGB 전처리에 사용하는 **Ultralytics YOLOv8에는 AGPL-3.0 또는 별도 Enterprise 계약 조건**이
적용됩니다. YOLO를 포함한 결합물의 공개·배포 범위에 맞는 조건을 확인해야 하며, 루트 LICENSE가
전체 파이프라인에 대한 Apache-2.0 단독 허가를 의미하지는 않습니다.

외부 코드의 라이선스·저작권 고지를 보존해야 하며, 가중치, 데이터셋, 원본 영상과 해당 영상에서
추출한 사례 이미지의 재배포 조건은 자산별로 별도 확인해야 합니다.
