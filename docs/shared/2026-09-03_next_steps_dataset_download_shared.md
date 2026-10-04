# NTU60 전처리 이후 실행·데이터 확보 연구 진행 공유

- 문서 ID: `DOC-20260903-next-steps-dataset-download-R9`
- 기준일: 2026-09-04
- 연구 상태: `in_progress`

## 1. 결론

현재 NTU60 raw skeleton과 공식 missing-sample 목록, 공식 DSTE XSub backbone은 이미 확보됐다.
따라서 NTU60이나 DSTE weight를 다시 받을 필요는 없다. 다음 core 학습을 막지 않도록 지금 먼저
준비할 항목은 다음과 같다.

1. FU-Kinect-Fall 원본·전처리와 F0-FU control 완료
2. SAFER-Activities 497-sequence lineage와 legacy split/window-count gate 완료; mapping 검증 대기
3. RGB 확장용 YOLOv8x·ViTPose-B·MotionAGFormer-B 공식 weight 확보
4. 개발용 외부 데이터의 version·license·annotation 선점
5. AI-Hub 71641은 접근 신청만 조기에 진행하고 모델과 평가는 봉인

SAFER 전체 173GB, OOPS 45GB와 모든 외부 데이터를 한꺼번에 받을 필요는 없다. SAFER core skeleton
재현에 직접 필요한 최소 묶음은 공식 tree 기준 pose archive 1개, split 8개, normal/non-lab CSV
65개로 총 74개·2,899,821,254 bytes(약 2.70 GiB)다. RGB 원본은 Final J1 뒤에 확장한다.

## 2. 현재 NTU60 상태와 완료 조건

공식 NTU60은 56,880개 raw skeleton과 missing/incomplete 302개를 제외한 56,578개 유효 표본 계약을
충족했다. 공식 UmURL 좌표계의 XSub 입력 생성도 완료됐고, train 40,091개와 validation 16,487개의
배열·label·frame count 및 preprocessing manifest를 확보했다.

완료는 다음 조건을 모두 만족할 때만 선언한다.

| 항목 | 완료 계약 |
| --- | --- |
| XSub train | 40,091 samples, `(N,3,300,25,2)` |
| XSub validation | 16,487 samples, `(N,3,300,25,2)` |
| 부속 파일 | 두 split의 label과 frame-count 길이 일치 |
| 좌표계 | official UmURL center/shoulder-axis normalization |
| provenance | preprocessing revision·profile·split을 manifest에 기록 |
| 품질 검사 | finite 좌표, split/class 무결성, downstream loader와 frozen DSTE smoke 통과 |

데이터 생성, 공식 구현 출력 동등성, downstream loader, frozen DSTE 제한 실행과 전체 ADL linear
evaluation까지 통과했다. epoch 150에서 기존 기록의 Top-1 85.285%, Top-5 97.234%를 재현했다.
FU binary linear control까지 완료했으며 OOF F1 94.611%, AUPRC 98.620%를 기록했다. SAFER 원본도
497 sequences·8,091,357 frames와 split/annotation lineage를 확인했다. legacy window-count gate는
V1 총 993,307 windows로 통과했고, 다음 gate는 mapping 검증과 SAFER F0A/F0B control이다.

## 3. 전체 실행 순서

```text
NTU 전처리 완료
  → official input 검증
  → Frozen DSTE ADL smoke
  → NTU60 60-class 전체 linear evaluation
  → FU provenance·split 및 F0-FU control 완료
  → SAFER provenance·split 고정 완료
  → legacy split/window-count gate 완료 → mapping 검증 → clean3d_v1 → SAFER F0A/F0B/F1 controls
  → corrected V2/V3
  → J0 → J1 → locked/nested evaluation
  → Final J1
  → G0/G1/G2
  → RGB development external
  → sealed external 1회
  → YAMNet/VLM/TrackMemory
```

ADL, F0-FU와 SAFER raw-data gate는 통과했다. legacy 입력 및 SAFER control을 확인하기 전에는 J0/J1
학습으로 이동하지 않는다. 외부평가를 본 뒤 threshold, checkpoint나 decoder를 바꾸지 않는다.

## 4. 데이터 다운로드 우선순위

### P0 — 이미 확보됨

| 자산 | 상태 | 결정 |
| --- | --- | --- |
| NTU RGB+D 60 3D skeleton | 공식 raw와 missing list 확보 | 다시 받지 않음 |
| FoundSkelModel DSTE NTU60 XSub | 구조 호환성과 순전파 확인 | 다시 받지 않음 |
| NTU RGB+D 120 | 현재 연구 질문 밖 | 추가 전처리 보류 |

### P1 — 지금 준비

| 자산 | 연구 역할 | 공식 배포 상태 | 지금 할 일 |
| --- | --- | --- | --- |
| [FU-Kinect-Fall](https://github.com/MuzafferAslan23/Fall-Detection-Dataset) | fall 대 intentional lying hard negative | 원본 감사·993개 전처리·F0-FU 완료 | 추가 다운로드 없음 |
| [SAFER-Activities](https://huggingface.co/datasets/SAFER-Activities/SAFER-Activities) | dense fall/state/recovery | gated access, 전체 repository 173GB | 74 files·2,899,821,254 bytes의 pose archive·공식 split·normal/non-lab CSV 확보 및 무결성 검사 완료 |
| [YOLOv8x](https://docs.ultralytics.com/models/yolov8) | RGB person detector | 공식 pretrained model 제공 | release·version·hash를 고정해 weight 확보 |
| [ViTPose-B](https://github.com/ViTAE-Transformer/ViTPose) | COCO17 2D pose | 여러 공식 variant 제공 | `Multi COCO17 256×192` 계약과 일치하는 weight만 확보 |
| [MotionAGFormer-B](https://github.com/TaatiTeam/MotionAGFormer) | H36M17 3D lifting | H3.6M과 MPI-INF-3DHP weight 분리 | H3.6M 243-frame base weight 확보 |

SAFER의 새 wheelchair subset은 유용하지만 삭제 전 497-sequence 결과 재현에는 포함하지 않는다.
우선 기존 normal 467 sequences와 non-lab OOD 30 sequences의 lineage를 고정하고, wheelchair 연구는
별도 revision으로 분리한다.

최소 묶음은 고정된 공식 revision에서 수신했으며 파일 수·총 byte size, pose archive와 ZIP 무결성이
일치했다. Normal 467개와 non-lab OOD 30개의 sequence·frame·split lineage도 기존 기록과 일치했다.
전체 RGB 영상과 extracted RGB feature는 받지 않았고, archive에 포함된 wheelchair pickle도 삭제 전
497-sequence 결과 재현에는 사용하지 않는다.

### P2 — Core 학습 중 확보할 development external

| 데이터 | 역할 | 공식 제공 상태 | 우선순위 메모 |
| --- | --- | --- | --- |
| [CAUCAFall v5](https://data.mendeley.com/datasets/7w7fccy7ky/5) | RGB/domain diagnostic | CC BY 4.0, 100 videos와 frame label | version 5 고정 |
| [Le2i FDD](https://search-data.ubfc.fr/FR-13002091000019-2024-04-09_Fall-Detection-Dataset.html) | 저해상도 engineering external | 8.95GB, 191 videos, CC BY-NC-SA | 공식 archive와 annotation 확보 |
| [HQFSD](https://iiw.kuleuven.be/onderzoek/advise/datasets) | 현실적 nursing-home development | meta·example·full video 링크 제공 | 개발용으로만 사용 |
| [MCFD](https://www.iro.umontreal.ca/~labimage/Dataset/) | 8-view cross-view external | 공식 연구실 host | view protocol을 먼저 고정 |
| [URFD](https://fenix.ur.edu.pl/~mkepski/ds/uf.html) | RGB/depth/accelerometer engineering | 70 sequences, CC BY-NC-SA | 필요한 modality와 sync data만 선택 |

CAUCA, OOPS와 Le2i는 이미 반복 진단에 사용된 계열이므로 pristine final proof로 표현하지 않는다.

### P3 — 대용량·후순위·봉인

| 데이터 | 공식 규모·접근 | 시작 조건 |
| --- | --- | --- |
| [OOPS](https://oops.cs.columbia.edu/data/) | video+annotation 45GB, CC BY-NC-SA | Final J1과 RGB front-end 고정 후 OOPS-Fall audit |
| [OmniFall](https://huggingface.co/datasets/simplexsigil2/omnifall) | repository 9.76GB, OOPS-Fall 818 videos, synthetic video 약 9.1GB | label/split 우선, synthetic video는 선택 |
| [EDF/OCCU](https://zenodo.org/records/15494102) | 26.9GB, depth occlusion benchmark | 기본 외부평가 뒤 occlusion 확장 |
| [AI-Hub 71641](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&dataSetSn=71641&topMenu=100) | 22,672 video/sensor pairs, 승인·안심존 요건 | 신청은 조기 준비, protocol lock 뒤 단일 sealed 실행 |
| SAFER full repository | 173GB | RGB/global/fusion 재현을 시작할 때 선택 확장 |

AI-Hub 71641은 8개 촬영 방향, paired sensor와 fall start/end annotation을 제공하지만 내국인 신청,
소속 증빙과 보안·안심존 절차가 적용될 수 있다. 접근 승인을 받더라도 threshold와 model selection에
사용하지 않고 pristine sealed 후보로 분리한다.

## 5. 데이터가 있어도 별도로 구현해야 하는 부분

현재 repository에는 학습 결과를 자동 복원할 전체 trainer가 남아 있지 않다. 다음 항목은 데이터
입수 후에도 구현 또는 정확한 복구가 필요하다.

1. 새 official UmURL NTU 입력을 명시적으로 선택하는 downstream data profile — 완료
2. FU 20-joint→proxy NTU25, quality exclusion과 subject-disjoint fold 생성 — 완료
3. SAFER dense label alignment, V2 relifting과 V3 overlap-add cache
4. FU F0 control — 완료; SAFER F0A/F0B, J0/J1, locked/nested, Final J1 trainer/evaluator는 복구 필요
5. G0/G1/G2와 train-only scaler의 matched comparison
6. raw RGB에서 YOLOv8x→ViTPose-B→MotionAGFormer-B를 실행하는 front-end
7. core 완료 뒤 YAMNet audio-only/late-fusion 평가

따라서 “데이터 확보”는 원상복구의 필요조건이지만 충분조건은 아니다. 과거 checkpoint의 bit-exact
복원 대신 동일 데이터·split·전처리·학습·평가 계약을 갖춘 reproduced result를 만드는 것이 목표다.

## 6. 입수 직후 공통 체크리스트

- 공식 배포 URL, dataset version, license와 citation 기록
- 원본 archive의 byte size·SHA-256·file count 기록
- raw와 derived data 분리
- subject/view/action/split count 확인
- FPS, frame count, annotation time unit과 decode failure 감사
- 원본과 파생 산출물의 manifest 연결
- 외부 데이터 추론 전에 checkpoint·threshold·window·decoder 고정

공식 설명과 실제 archive 수가 다르면 임의로 맞추지 않는다. 특히 FU의 공식 1,008 clips와 기존
기록의 1,006→993, Le2i 공식 record의 191 videos와 다른 파생 benchmark의 190 videos 차이는 각각
별도 provenance 항목으로 남긴다.

## 7. YAMNet 관련 다운로드 판단

YAMNet pretrained inference를 위해 AudioSet 전체를 내려받을 필요는 없다. 지금은 core skeleton과
RGB 재현이 우선이며, YAMNet은 이후 edge-only audio 보조 증거로 평가한다. 그때 skeleton-only,
audio-only와 late fusion을 같은 event split에서 비교하고, raw audio는 서버로 전송하거나 장기
보존하지 않는다.

## 8. 참고문헌·공개 자료

- [NTU RGB+D 공식 배포 페이지](https://rose1.ntu.edu.sg/dataset/actionRecognition/)
- [NTU RGB+D 공식 코드·missing list](https://github.com/shahroudy/NTURGB-D)
- [SAFER-Activities 공식 코드](https://github.com/safer-activities/SAFER-Activities)
- [CAUCAFall 논문](https://doi.org/10.1016/j.dib.2022.108610)
- [OOPS 논문](https://openaccess.thecvf.com/content_CVPR_2020/html/Epstein_Oops_Predicting_Unintentional_Action_in_Video_CVPR_2020_paper.html)
- [OmniFall 논문](https://arxiv.org/abs/2505.19889)
- [Le2i FDD 논문](https://doi.org/10.1117/1.JEI.22.4.041106)
- [HQFSD 논문](https://doi.org/10.1049/htl.2015.0047)
- [MCFD 공식 배포 페이지](https://www.iro.umontreal.ca/~labimage/Dataset/)
- [URFD 논문](https://doi.org/10.1016/j.cmpb.2014.09.005)
- [EDF/OCCU 논문](https://doi.org/10.1007/978-3-319-14364-4_19)
