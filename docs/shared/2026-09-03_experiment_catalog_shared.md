# FoundSkelModel 낙상·이상행동 연구 전체 정리

- 문서 ID: `DOC-20260903-experiment-catalog-R28`
- 기준일: 2026-09-22
- 연구 상태: `paused` — 주요 스켈레톤 비교 완료, V3 평가 정의·후반 데이터 검토 필요
- 전체 단일 문서: [공유 통합본](2026-09-03_project_complete_summary_shared.md)

## 최신 연구 진행 — 완료 결과와 남은 검증

[전체 진행](2026-09-22_recovery_execution_shared.md)에서 현재 상태를 구분한다.
[V2 대조 모델](2026-09-22_safer_v2_controls_shared.md)과
[J0/J1 V1/V2](2026-09-22_joint_reconstruction_shared.md)의 전체 평가·독립 검산을 완료했다.
V1/V2 J1의FU nestedF1은93.373/92.683%이며, V2에서는오탐과recall이함께줄었다.
[FU4분류기](2026-09-22_fu_classifier_reconstruction_shared.md)도nested검산을완료했지만V3reference는없다.
[V3pilot](2026-09-22_safer_v3_reconstruction_shared.md)은원본좌표0프레임620개로
별도정의한퇴화입력검사를미통과했다. full생성·V3후속학습은중단했고평가정의추가검토가필요하다.
원본과동일한수치복원이나후반연구전체완료를뜻하지않는다. 아래이전진행은당시이력이다.

## 이전 진행 — R27 당시 상태

[V2 전처리](2026-09-20_safer_v2_reconstruction_shared.md)는 전체497sequences/
8,091,357frames와1,007,723windows 생성·전수 검산을 완료했다.
[Matched controls](2026-09-22_safer_v2_controls_shared.md)는 전체 실행 중이며,
[J0/J1](2026-09-22_joint_reconstruction_shared.md)은 방법·구현 검증 단계다.
V3 pilot와 후반 연구는 남아 있고 새 낙상 성능은 아직 확정하지 않는다.

FU [ZS0/ZS1](2026-09-20_fu_zs_reconstruction_shared.md)과
[D0/D1/D2](2026-09-20_fu_probe_reconstruction_shared.md) 전체 실행·독립 검산을 완료했다.
Native30 D0-TS/D1/D2 개발 OOF F1은95.575/91.765/95.575%다. 별도 재구현의 개발 지표이며
역사적 J1/nested/외부 성능이 아니다. [P0 비교](2026-09-20_primitive_reconstruction_shared.md)도 완료했다.
결합 F1 95.808%·lying FP9지만 fall TP가162→160으로 줄어 미채택이다.
P1도 전체 학습·독립 검산을 완료했다. epoch2 validation Macro-F1 75.313%, fall F1 77.419%이며
test/OOD는 미평가다. V2 후속 비교·V3·J0/J1·RGB/외부는 아직 남아 있다.

F1은 별도 문서 기반 조건에서 네 후보 20-epoch 학습·고정 평가·독립 검산을 완료했다.
Temporal+spatial/sqrt epoch17의 test/OOD Macro-F1은 75.278/56.516%, fall F1은
79.063/54.271%다. [F1 실행 결과](2026-09-20_f1_run_shared.md)를 따르며 원본 동등성이나
전체 복구 완료를 주장하지 않는다. F0B와 평가 단위가 달라 직접적인 개선량으로 비교하지 않는다.

고정 F0B의 test/OOD 전체 평가와 독립 검산을 완료했다. Test/OOD Macro-F1 71.505/53.452%,
fall F1 79.228/52.744%이며 원본과 동일한 실험이라는 주장은 하지 않는다. 이전 revision의 평가
대기 상태는 이 결과로 갱신한다. [최신 전체 현황](2026-09-19_recovery_restart_shared.md)과
[F1 재현 계약](2026-09-20_f1_reconstruction_contract_shared.md)을 현재 시작점으로 사용한다.

## 1. 연구 목표

기존 skeleton 기반 60-class 일상행동 인식 성능을 보존하면서 낙상 전문 분기를 추가하고, RGB 영상의
전역 움직임과 시간 상태를 결합해 intentional lying과 낙상을 구분하는 것이 목표다. 장기적으로는
사람별 bounded memory와 VLM을 이용해 낙상·배회·비활동 사건을 설명하되, 낙상 경보 권한은 전문
detector에 유지한다.

## 2. 전체 구조

```text
RGB → 사람 검출·2D pose → 3D pose lifting → proxy NTU25
                                           ↓
NTU60 3D → Frozen DSTE ─┬→ Frozen 60-class ADL head
                        └→ J1 fall specialist ─┐
RGB 전역 움직임 131-D ─────────────────────────┤→ G0/G2 → D1 event decoder
SAFER 상태·회복 분기 ──────────────────────────┘

장기 확장: tracking → bounded TrackMemory → event VLM interpretation
```

## 3. 실험별 문서

| ID | 실험 | 역사적 상태 | 핵심 결론 |
| --- | --- | --- | --- |
| [E01](2026-09-03_01_ntu60_adl_shared.md) | NTU60 ADL 재현·보존 | 역사적·현재 `completed` | official epoch 150에서 Top-1 85.285%, Top-5 97.234% 재현 |
| [E02](2026-09-03_02_fu_kinect_shared.md) | FU-Kinect fall/lying | 역사적·현재 F0 `completed` | recovery control OOF F1 94.611%; FP 11건 모두 lying |
| [E03](2026-09-03_03_safer_preprocessing_shared.md) | SAFER 전처리·temporal | V1 controls·F1·P1 완료, V2 `in_progress` | V2 전체 생성·검산 후 matched controls·V3 pilot |
| [E04](2026-09-03_04_joint_j0_j1_shared.md) | J0/J1 공동학습 | `completed` | Frozen ADL을 유지한 shared residual specialist 확립 |
| [E05](2026-09-03_05_lady_ablation_shared.md) | LaDy/physics ablation | `not_selected` | FU hard-negative 일반화에 안전한 상보성 없음 |
| [E06](2026-09-03_06_rgb_global_motion_shared.md) | RGB·Global Motion | `completed` 후보 | G2 이득은 dataset에 따라 달라 G0 fallback 필요 |
| [E07](2026-09-03_07_caucafall_shared.md) | CAUCAFall 진단 | `completed` diagnostic | 현재 RGB 경로는 작동하나 pristine test는 아님 |
| [E08](2026-09-03_08_omnifall_oops_shared.md) | OOPS failure audit | `completed` diagnostic | C2는 coverage를 개선했고 D1은 latch 결함을 분리 |
| [E09](2026-09-03_09_state_recovery_decoder_shared.md) | 상태·회복·D1 | 부분 완료 | 상태 안정화는 가능, recovery 일반화는 부족 |
| [E10](2026-09-03_10_le2i_shared.md) | Le2i engineering external | `completed` smoke | 저해상도 경로 실행 가능성 확인 |
| [E11](2026-09-03_11_external_protocol_shared.md) | 외부평가 프로토콜 | `planned` | 적격성 감사와 봉인 실행을 분리 |
| [E12](2026-09-03_12_vlm_trackmemory_shared.md) | VLM·TrackMemory | 설계 완료, 통합 `planned` | detector-authoritative 해석 계층과 bounded memory |
| [E13](2026-09-03_13_benchmarks_provenance_shared.md) | 비교 모델·provenance | `completed` audit | modality·split·실행 출처가 다른 수치를 분리 |

## 4. 기존 기록 기준 핵심 결과

| 평가 | 결과 | 주장 범위 |
| --- | --- | --- |
| NTU60 ADL | Top-1 85.285%, Top-5 97.234% | native Kinect skeleton XSub |
| FU nested J1-V3 | Precision 94.268%, Recall 89.697%, F1 91.925%, lying FPR 5.357% | FU는 개발에 사용된 내부 skeleton domain |
| SAFER locked J1-V3 | Test fall F1 75.692%, OOD fall F1 50.297% | domain gap과 recall trade-off 존재 |
| SAFER G2 | G0 대비 Test/OOD Macro-F1 +2.997/+2.959%p | OOD fall recall은 3.909%p 감소 |
| CAUCA current RGB | G0 F1 95.146%, G2 F1 93.878% | 반복 diagnostic, final proof 아님 |
| OOPS C2+D1 | Precision 54.893%, Recall 67.505%, F1 60.550% | decoder 결함 수정 후 matched diagnostic |
| Le2i locked G2 | Precision 84.946%, Recall 82.292%, F1 83.598% | engineering smoke |

위 수치는 삭제 전 산출물을 대조해 고정한 **기존 기록 기준 역사적 결과**다. 현재 재현 결과로
간주하지 않으며, 새 실행은 별도 `reproduced result`로 보고한다.

## 5. 현재 업데이트

- 공식 NTU60 XSub DSTE 백본의 구조 호환성과 최소 순전파 검증을 완료했다.
- NTU60 56,578개 표본을 train 40,091개와 validation 16,487개로 구성했다.
- 전체 60-class ADL head 학습에서 DSTE 불변성은 확인했으나 Top-1 79.899%, Top-5 94.899%로 역사적
  성능 gate를 통과하지 못했다.
- 현재 가공 입력의 프레임별 정규화가 공식 UmURL 좌표계와 달라 원본 NTU60 skeleton 입력을
  복구했다. NTU 전용 normalization과 전체 XSub 입력 생성을 완료했으며 공식 구현과의 출력
  동등성, 표본 수, class 범위와 유한값 검사를 통과했다.
- 새 official 입력을 선택하는 downstream profile과 frozen DSTE 제한 실행도 통과했다. 제한 표본
  정확도는 성능 결과로 사용하지 않는다.
- 전체 linear evaluation의 epoch 150에서 기존 기록과 동일한 Top-1 85.285%, Top-5 97.234%를
  재현했다. 전체 구간 best는 Top-1 85.298%, Top-5 97.234%이며 encoder 불변성도 통과했다.
- FU-Kinect skeleton 1,006개를 감사해 기존 기록과 같은 993개 품질 집합과 21-subject 5 folds를
  재구성했으며 전처리 무결성 검사를 통과했다.
- Frozen DSTE 기반 FU binary linear control은 OOF Precision 93.491%, Recall 95.758%, F1 94.611%,
  AUPRC 98.620%를 기록했다. 993개 표본의 OOF·subject split과 encoder 불변성 검사를 통과했고,
  오탐 11건은 모두 intentional lying이었다.
- SAFER 원본 감사에서 기존 기록과 같은 497 sequences·8,091,357 frames와 legacy 3D non-finite
  193 sequences·93,256 frames를 확인했다.
- epoch-150 matched head로 문서 근거 legacy 후보를 subject-disjoint validation에서 비교했지만 사전
  gate를 통과한 후보는 없었다. test/OOD는 열지 않았고 metric에 맞춘 추가 탐색은 종료했다.
- 보존 기록과 공식 전처리에 가장 직접적인 구조적 복원안을 한계와 함께 고정했고 전체 993,307개
  입력의 무결성 검사를 통과했다.
- reconstructed V1 F0A 전체 평가에서 Validation/Test/OOD fall F1은 0.000/1.784/1.925%, conditional
  AUPRC는 55.184/69.560/54.271%였다. 고정 ADL decision은 실패했지만 순위 신호가 남아 있어 새 SAFER
  경계를 학습하는 F0B가 다음 단계다.
- F0B는 frozen DSTE의 temporal-only/temporal+spatial linear control과 validation-only 선택 계약을 먼저
  확정한 뒤 진행하며, 이후 F1, corrected V2/V3, J0/J1, Final J1, G0/G2와 RGB 외부평가로 이동한다.
- VLM, TrackMemory, YAMNet은 core detector 복구 후 별도 revision에서 평가한다.

## 6. 비식별·안전 원칙

원본 RGB와 audio는 가능한 한 edge에서 처리하고 서버에는 축약된 event·quality·coarse time만
전송한다. Skeleton도 체형과 보행 패턴을 포함할 수 있으므로 좌표 정규화, session-scoped ID,
짧은 보존기간, 암호화와 접근통제를 함께 적용한다. VLM과 YAMNet은 보조 해석기이며 detector alert를
자동 취소할 권한이 없다.

## 7. 다음 연구

1. Validation에서 고정한 SAFER F0B temporal+spatial head의 test/OOD 단일 평가
2. SAFER F1 temporal adapter 재현
3. Corrected V2/V3와 J0/J1 matched 재학습
4. locked/nested 평가 후 Final J1 확정
5. G0/G1/G2와 RGB 외부평가 재연결
6. 봉인 외부 데이터에서 사전 등록된 단일 확증 평가
7. edge 비식별 정책 아래 VLM/YAMNet/TrackMemory 추가가치 평가

## 8. 공개 참고자료

- [FoundSkelModel 공식 저장소](https://github.com/wengwanjiang/FoundSkelModel)
- [Foundation Model for Skeleton-Based Human Action Understanding](https://arxiv.org/abs/2508.12586)
