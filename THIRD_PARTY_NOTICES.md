# Third-party notices

확인일: 2026-10-04

이 문서는 본 프로젝트의 주요 외부 구성요소와 확인한 **코드 라이선스**를 안내합니다.
FoundSkelModel에서 계승한 [Apache License 2.0](LICENSE)을 보존하며, 제3자 코드의 권리와
조건은 각 원본 라이선스를 따릅니다. 이 목록 자체가 새로운 이용 허가를 부여하거나 각 라이선스
전문을 대체하지는 않습니다. 패키지의 모든 간접 의존성에 대한 완전한 목록은 아닙니다.

## 주요 파이프라인

| 구성요소와 출처 | 역할 | 확인한 코드 라이선스 |
| --- | --- | --- |
| [FoundSkelModel](https://github.com/wengwanjiang/FoundSkelModel) | 기반 코드와 DSTE skeleton encoder | [Apache-2.0](https://github.com/wengwanjiang/FoundSkelModel/blob/main/LICENSE) |
| [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) | RGB 영상의 사람 검출; YOLOv8 사용 | [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/v8.4.138/LICENSE) 또는 별도 Enterprise 계약 |
| [ViTPose](https://github.com/ViTAE-Transformer/ViTPose) | 2D 관절 추정 | [Apache-2.0](https://github.com/ViTAE-Transformer/ViTPose/blob/c050ed29112da7704797cc1a65af0234b525010d/LICENSE) |
| [MotionAGFormer](https://github.com/TaatiTeam/MotionAGFormer) | 2D 관절에서 3D skeleton 추정 | [Apache-2.0](https://github.com/TaatiTeam/MotionAGFormer/blob/4756fd1eb7cc73f0e991f091ff2280e030ab85f3/LICENSE) |

ViTPose LICENSE의 `Copyright 2018-2020 Open-MMLab. All rights reserved.` 고지를
보존해야 합니다. 다른 파일에 포함된 관련 저작권·출처 고지도 유지합니다.
Ultralytics 링크는 확인한 설치 버전 8.4.138을 가리키며, 다른 버전을 배포할 때는 그 버전의
실제 라이선스도 확인해야 합니다.

### Ultralytics가 포함되는 배포

[Ultralytics 공급사 안내](https://www.ultralytics.com/license)는 AGPL-3.0에 따른 공개 또는
별도 Enterprise 계약을 안내합니다. 현재 RGB 처리 코드는 `ultralytics.YOLO`를 직접 사용합니다.
이를 포함한 배포에서는 적용되는 AGPL 조건에 따라 라이선스와 필요한 대응 소스를 제공할 범위를
확인해야 합니다. 전체 결합물을 Apache-2.0만으로 배포할 수 있다고 이 문서가 보증하지 않습니다.

외부 코드를 Git에서 제외하거나 설치 의존성으로 선언하는 것만으로 결합물에 대한 조건이
자동으로 해소되지는 않습니다. 이 고지를 추가하는 작업으로 프로젝트 전체를 AGPL로
재라이선스하거나 Enterprise 계약을 취득한 것도 아닙니다.

## 주요 비교 모델

다음 구현은 비교 실험에 사용한 별도의 구성요소입니다. 사용자 모델의 자체 구현으로
표시하지 않으며, 코드 라이선스와 모델 가중치의 배포 조건을 구분합니다.

| 구성요소와 출처 | 확인한 코드 라이선스 | 고지 또는 미확인 사항 |
| --- | --- | --- |
| [HFD 3D-CNN+SVM](https://github.com/ekramalam/HFD_3DCNN/tree/b74c13524979302641c1e55fc8582939daa58800) | 미확인 | 확인한 revision의 루트 라이선스 파일을 찾지 못함. 저자 코드와 C3D·SVM 가중치의 재배포 권한을 별도 확인해야 함 |
| [Privacy X3D-UDA](https://github.com/1015206533/privacy_supporting_fall_detection) | [Apache-2.0](https://github.com/1015206533/privacy_supporting_fall_detection/blob/55e784df987eff05e12f2e0aeda3867aa4e91b1b/LICENSE) | Copyright 2018-2019 Open-MMLab. All rights reserved. |
| [FLASH](https://github.com/Tresor-Koffi/FLASH-Impact-Fall-Detection) | [MIT](https://github.com/Tresor-Koffi/FLASH-Impact-Fall-Detection/blob/04a1215314472085aea68009159c5f9237e9e7cf/LICENSE) | Copyright (c) 2026 Tresor Y. Koffi |

이 공개 사본에 포함된 Privacy X3D-UDA 평가 설정에는 원본 저작권과 변경 고지를 표시했고,
[원본 라이선스 전문](licenses/Privacy-X3D-UDA-LICENSE.txt)을 함께 제공합니다.

## 보조 구현과 관련 자료

아래 항목은 추가 비교, 이전 연구 또는 도구에 관련된 구성요소이며, 모두가 현재 낙상 추론의
필수 구성요소라는 의미는 아닙니다.

| 구성요소와 출처 | 확인한 코드 라이선스 | 보존할 원본 고지 |
| --- | --- | --- |
| [SAFER fall_detection / pyskl 기반 구현](https://github.com/senior-action-recognition/fall_detection) | [Apache-2.0](https://github.com/senior-action-recognition/fall_detection/blob/d7f2476245f8e9164acfa98fa47d386ecd617ca3/LICENSE) | Copyright 2018-2019 Open-MMLab. All rights reserved. |
| [LaDy](https://github.com/HaoyuJi/LaDy) | [MIT](https://github.com/HaoyuJi/LaDy/blob/bdba27dc03670f0ad54927a7b50d0d0d04b22997/LICENSE) | Copyright (c) 2025 Haoyu Ji |
| [DistillH-Mamba](https://github.com/Tresor-Koffi/DistillH-Mamba) | [MIT](https://github.com/Tresor-Koffi/DistillH-Mamba/blob/b0e7b66e150933f451f5618f320849d2a00f965b/LICENSE) | Copyright (c) 2025 Tresor-Koffi |
| [Impact fall detection ST-GCN](https://github.com/Tresor-Koffi/impact-fall-detection-stgcn) | [MIT](https://github.com/Tresor-Koffi/impact-fall-detection-stgcn/blob/7f5e00637c7f27fcb5a4d23894930cf228cd68ea/LICENSE) | Copyright (c) 2025 Tresor-Koffi |

[RTHFD](https://github.com/ekramalam/RTHFD), [SDFA](https://github.com/saniazahan/SDFA),
[Modeling Human Skeleton Joint Dynamics](https://github.com/saniazahan/Modeling-Human-Skeleton-Joint-Dynamics-for-Fall-Detection-)
및 [UP-Fall 3D skeleton 자료](https://github.com/Tresor-Koffi/3D_skeletons-UP-Fall-Dataset)도
확인한 checkout의 루트 라이선스 파일을 찾지 못했습니다. 관련 코드·가중치·자료의 재배포 권한은
미확인입니다. GitHub 공개 열람만으로 일반적인 재배포 허가를 가정하지 않습니다.
[GitHub의 라이선스 안내](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository)

## 코드를 포함해 배포하는 경우

- **Apache-2.0**: 라이선스 전문을 제공하고 관련 저작권·특허·상표·출처 고지를 보존합니다.
  수정한 파일에는 변경 사실을 표시하고, 원본에 `NOTICE`가 있으면 관련 고지를 전달합니다.
  자세한 조건은 [Apache-2.0 제4조](https://www.apache.org/licenses/LICENSE-2.0)를 따릅니다.
- **MIT**: 원본 LICENSE 전문과 저작권 고지를 해당 코드의 사본 또는 상당 부분과 함께 보존합니다.
- **AGPL-3.0**: 실제 결합·배포 형태에 적용되는 원문 조건에 따라 라이선스, 고지와 필요한 대응
  소스를 제공합니다. 구체적인 범위가 불분명하면 배포 전 확인합니다.
- 외부 코드를 복사하거나 수정해 포함할 때 이 표의 링크만 남기는 것으로 원문 동봉 의무를
  대신하지 않습니다. 파일별로 추가 고지가 있으면 함께 보존합니다.

## 가중치, 데이터와 사례 이미지

이 저장소는 외부 checkout, 데이터셋과 checkpoint를 기본 Git 추적에서 제외하도록 구성되어
있습니다. 이 제외 설정 자체가 재배포 권한 확인을 대신하지는 않습니다.

코드 라이선스만으로 pretrained checkpoint, 새로 학습한 가중치, 데이터셋, 원본 영상,
추출 프레임과 사례 이미지의 공개 조건을 확정할 수 없습니다. 공개할 자산별로 배포처의 이용
조건과 관련 권리를 확인하고 출처를 표시해야 합니다. 확인되지 않은 자산을 이 저장소의
Apache-2.0 라이선스 대상으로 묶어 안내하지 않습니다.

논문 인용과 소프트웨어 라이선스 준수는 별개입니다. 참고문헌을 기재하더라도 필요한 라이선스
원문·저작권 고지 또는 별도 이용 허가를 대신하지는 않습니다.
