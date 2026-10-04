# 최신 비교 모델 후보 검토

- 문서 ID: `DOC-20261004-recent-baseline-candidates-R1`
- 기준일: 2026-10-04
- 상태: 조사 `completed`; 새 모델 평가 미실행

낙상 특화 모델을 대상으로 한 후속 취득·검증 결과는 [외부 평가 준비 결과](2026-10-04_fall_specialized_baselines_shared.md)에 정리했다.

## 1. 선정 기준

최신성 요구를 반영해2025~2026 연구를 우선하고2024 연구를 보조 후보로 조사했다.
논문·모델의 연도와 코드·가중치 재배포 날짜를 구분했다. 저자 코드와 학습된 모델의
공개 수준, 낙상 과제와의 관계, Le2i·CAUCA 학습 미사용 조건을 함께 검토했다.

**이번 조사에서는 CascadeFormer를 우선 검토 후보로 선정했다.** 최신 연구이며,
저자 배포 목록에 학습된 행동 분류기까지 포함되어 있기 때문이다. 아직 가중치를
모델에 로딩하거나 외부 평가한 것은 아니다.

## 2. 후보 비교

| 모델 | 연도·학술 출처 | 연구상 역할 | 확인된 상태 |
| --- | --- | --- | --- |
| **CascadeFormer** | **ICPR2026 채택 공지**,2025 사전공개 | NTU 행동인식에서 낙상으로의 전이 비교 | 저자 코드와 학습된 분류기를 포함한 모델 파일 공개 확인 |
| **FLASH** | **ICIP2026 채택 공지** | Hypergraph+Mamba 기반 낙상 impact 검출 | 저자 코드 공개. 확인한 저장소·배포 목록에서 학습된 가중치 미확인 |
| **DistillH-Mamba** | **IEEE Sensors Journal2026** | Hypergraph+Mamba+지식증류 기반 impact 검출 | 저자 코드 공개. 확인한 저장소·배포 목록에서 학습된 가중치 미확인 |
| **SkateFormer** | **ECCV2024** | Transformer 기반 NTU 행동인식의 낙상 전이 비교 | 저자 코드와 학습된 모델 파일 공개 확인. 최신성 측면에서 보조 후보 |

CascadeFormer와 FLASH의 학회 채택은 저자 공식 저장소 공지를 근거로 한다.
DistillH-Mamba는 저장소 제목에2025가 있으나 DOI 등록 정보의 정식 권호는
2026년26권10호이므로2026으로 표시했다. SkateFormer의2026년 가중치 재배포는
기반 방법의 발표 연도2024와 구분했다.
[CascadeFormer 공식 구현](https://github.com/Yusen-Peng/CascadeFormer),
[FLASH 공식 구현](https://github.com/Tresor-Koffi/FLASH-Impact-Fall-Detection),
[DistillH-Mamba 출판 논문](https://doi.org/10.1109/JSEN.2025.3620575),
[SkateFormer 공식 구현](https://github.com/KAIST-VICLab/SkateFormer).

## 3. 실제 비교에 필요한 조건

**CascadeFormer**는 masked pretraining과 두 단계 Transformer를 사용하는 행동인식
모델이다. 공개된 NTU 모델은 분류기까지 학습되어 있으므로 A43 낙상 행동을 이용하는
전이 기준선 후보가 된다. 원래부터 낙상 이진 분류기인 것은 아니다. 현재 사용자 모델의
17관절 입력을 그대로 연결할 수 있다고 확인한 상태도 아니므로 NTU 관절·좌표·정규화와
시간 처리 조건을 맞춰야 한다. 학습 데이터 명세 기준으로 Le2i·CAUCA 미사용 후보이며,
실제 선택 가중치의 학습 이력을 확정한 후 평가한다.
[논문](https://arxiv.org/abs/2509.00692),
[저자 공개 모델](https://huggingface.co/YusenPeng/CascadeFormerCheckpoints).

**FLASH와 DistillH-Mamba**는33관절3D 스켈레톤을 이용하는 UP-Fall 기반 impact 검출
연구다. 낙상 중 충돌 순간이라는 목표를 영상 낙상 유무나 사건 검출 평가와 연결하는
규칙을 명시해야 한다. 이번에 확인한 공식 저장소의 파일 목록과 배포 항목에서는
학습 완료 가중치를 찾지 못했으므로, 원본 가중치로 바로 실행 가능한 모델로 소개하지
않는다. 모든 외부 미러와 과거 배포를 전수 조사했다는 뜻은 아니다.
[FLASH 논문](https://arxiv.org/abs/2607.25791),
[DistillH-Mamba 공식 구현](https://github.com/Tresor-Koffi/DistillH-Mamba).

**SkateFormer**는 상대적으로 오래되었지만 공개된 완성 모델이 명확한 대안이다.
원본 NTU25관절을 읽고24관절로 재배열하는 구조여서 역시17관절 입력과의 차이가 있다.
[저자 공개 모델](https://huggingface.co/JeonghyeokDo/SkateFormer).

## 4. 권고와 한계

최신성과 공개 가중치를 함께 요구하면 **CascadeFormer의 입력·모델 로딩 검증을 우선**한다.
낙상 전용 최신 연구를 반드시 포함하려면 **FLASH 또는 DistillH-Mamba의 원본 가중치 확보**가
먼저다. 가중치 미확인 상태에서 새로 학습한 모델을 저자 원본 모델로 보고하지 않는다.

OmniFall의 최신 RGB 표현 모델도 관련 연구지만 합본 학습에 Le2i·CAUCA가 들어갈 수
있어 source별 학습 이력을 확인해야 한다. SAFER2026에서 학습한 기존 구조 역시 최신
데이터셋 연구와 최신 모델 구조를 구분한다.

이번 조사는 후보 선정이며 새 낙상 성능은 없다. 외부 평가를 진행할 때는 동일한
Le2i130개·CAUCA100개, 동일 정답·실패 포함 집계, target 추가 학습·판정 기준 조정
없음의 조건을 유지한다. 행동인식 정확도나 impact 논문의 수치를 해당 외부 데이터셋의
낙상 F1로 대신 사용하지 않는다.
