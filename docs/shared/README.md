# 공유용 문서 디렉터리

이 디렉터리에는 팀원 또는 외부 연구 협력자에게 그대로 전달할 수 있는 **연구 내용과 연구 진행
내용만** 둔다.

작성 전 [연구 문서 분리 규칙](../documentation_rules.md)과
[공유용 템플릿](../templates/shared_research_progress_template.md)을 따른다.

로컬 경로, 명령어, 로그, 환경 복구, 설치 과정, 역할·일정, 접근 정보와 개인 메모는 이곳에
기록하지 않는다.

## 공유 문서 인덱스

### 전체 현황과 재현 진행

| 문서 | 내용 | 상태 |
| --- | --- | --- |
| [연구 강점과 논문 작성 근거](2026-10-04_research_strengths_paper_rationale_shared.md) | ADL 고정·J0/J1·외부 전이·실패 분석을 잇는 기여 서술과 원고 초안; 근거·주장 범위 구분 | `completed` |
| [외부 비교·원본 기준모델 종합](2026-10-04_external_comparison_summary_shared.md) | R4: 사용자·원본 기준·선정3종의 영상 Precision·Recall·F1; Le2i130·CAUCA100 | `completed` |
| [FLASH 직접 학습·외부 평가](2026-10-04_flash_reproduction_shared.md) | 공식 모델 source-only 재현과 동일230개 외부 평가 | `completed` |
| [최신 낙상 특화 모델 평가 준비](2026-10-04_fall_specialized_baselines_shared.md) | FLASH·Distill 가중치 미확인; STGCN-GRU-BiLSTM74텐서 로딩 성공, 정규화 미확정 | 준비 검증 `completed`, 평가 `paused` |
| [최신 비교 모델 후보](2026-10-04_recent_baseline_candidates_shared.md) | CascadeFormer·FLASH·DistillH-Mamba2026, SkateFormer2024; 실제 배포·출판 연도 대조 | 조사 `completed`, 새 평가 미실행 |
| [추가 비교 모델 후보](2026-10-04_additional_baseline_candidates_shared.md) | PoseC3D 전체 확장·DICTA2021·UMDR RGB 후보; 원본 가중치 접근·입력 조건 구분 | 조사 `completed`, 새 평가 미실행 |
| [USDRL 외부 평가](2026-10-04_usdrl_external_evaluation_shared.md) | 공식 백본과 NTU60 재학습 분류기; 사건 F1 Le2i81.82/CAUCA54.05%, A43 출력·독립 검산 완료 | `completed` |
| [HFD·Privacy X3D 외부 비교](2026-10-04_hfd_x3d_external_comparison_shared.md) | 세 모델 동일230개 영상 비교·X3D 저자 원본 실행·독립 검산 완료 | `completed` |
| [HFD·SDFA 학습 파일 추가 조사](2026-10-04_original_checkpoint_search_shared.md) | C3D 원본 확보·검증, 선행 연구6개 가중치 확보; HFD SVM·SDFA 원본은 미확보 | 취득·검증 `completed`, 평가 `paused` |
| [저자 원본 RTHFD 외부 평가](2026-10-04_original_baseline_evaluation_shared.md) | RTHFD 원본230개 실제 평가·독립 검산; HFD SVM·SDFA checkpoint 미확보 | RTHFD `completed`, 두 모델 `paused` |
| [HFD·SDFA·RTHFD 후보 검토](2026-10-03_hfd_sdfa_rthfd_feasibility_shared.md) | 저자 학습 원본 checkpoint만 평가; 신규 학습 제외, 가중치 확보 중 | 검토 `completed`, 준비 `in_progress` |
| [CAUCA 전체100개 네 모델 비교](2026-10-03_cauca100_comparison_shared.md) | 사건 F1 사용자82.98/ST86.79/MS87.23/CNN80.77%; 동일100개, 독립 검산 | `completed` |
| [CAUCA 전체100개 사용자 평가](2026-10-03_cauca100_own_evaluation_shared.md) | 전체100개·낙상50건; 사건 F1 82.98%, 독립 검산 완료 | `completed` |
| [Le2i130 원본 정답 재평가](2026-10-03_le2i130_reevaluation_shared.md) | 전체 130개·99낙상; F1 사용자91.01/ST97.51/MS97.00/CNN94.53%, 독립 검산 | `completed` |
| [Le2i 제외3개 원본 정답 확인](2026-10-03_le2i_missing_annotation_search_shared.md) | 원본 중간의 정답 확인,130영상/99낙상 목록 확보; 후속 재평가 완료 | 확인 `completed` |
| [저자 원본 평가 사전 검토](2026-10-03_author_evaluation_feasibility_shared.md) | 공개 프로그램과 논문 조건 차이; 정답 문제 후속 해소, 전체 재실행 미완료 | 검토 `completed` |
| [학습 미사용 외부 비교 확장](2026-10-03_external_comparison_expansion_shared.md) | Le2i·GMDCSA·CAUCA에서 세 구조와 사용자 모델 비교 | `completed` |
| [서로 다른 모델 3종·새 데이터셋 3종](2026-10-03_three_models_three_new_datasets_shared.md) | 네 모델×EDF·OCCU·OOPS 직접 추론·독립 검산 | `completed` |
| [외부 저성능 원인 진단](2026-10-03_external_failure_diagnosis_shared.md) | 품질 거부·추론 후 미탐 분해, 동일 부분집합 비교·실제 출력 진단 | `completed` |
| [EDF confidence 완화 진단](2026-10-03_edf_confidence_ablation_shared.md) | 사후 기준 0.30→0.05, 기존 미탐 8건 중 1건 추가 검출·전체 F1 64.29% | `completed` |
| [J1–G0 Related Work 전체 정리](2026-10-03_j1_g0_related_work_shared.md) | 입력·변환·DSTE layer·adapter 출처, 40개 인용·BibTeX·구성요소 대응표·원고 초안 | 문헌 정리 `completed` |
| [양쪽 학습 미사용 외부 비교](2026-10-03_shared_unseen_external_shared.md) | 공식 PoseC3D 두 학습모델과94영상 비교; SAFER 대비 GMDCSA·CAUCA F1우세, Le2i열세 | `completed` |
| [시간축 수정 후 연속 평가](2026-10-02_temporal_continuous_evaluation_shared.md) | 사건 F1: Le2i90.71/78.95%, GMDCSA70.97%, CAUCA87.50%; 독립 검산 | `completed` |
| [현재 모델의 시간 입력 감사](2026-10-02_temporal_input_audit_shared.md) | 64/8 순서 보존·구간 시간 변형·Le2i 시각 비교 문제 확인 | 감사 `completed`, 수정·재평가 미실행 |
| [학습 미포함 외부 성능 목표](2026-10-02_cross_dataset_objective_shared.md) | 입력 정합·품질 거부에 따른 F1 상한·후속 검증 순서 | 검토 `completed`, 개선 미실행 |
| [CAUCA 공식 구간 평가](2026-10-02_omnifall_cauca_benchmark_shared.md) | 전체47 F150.00%, 실패 분해·독립 검산·3개 데이터셋 비교 | `completed` |
| [J1과 G0 단독 구조](2026-10-02_j1_g0_only_shared.md) | 현행 구조, Le2i38/127·URFD70 F1 78.95/90.71/58.82%,5555창 동등성 검증 | `completed` |
| [Le2i 공식 시험 연속 처리](2026-10-02_omnifall_le2i_continuous_shared.md) | 38영상 G0/G2 사건 F1 78.95%, TP15/FP1/FN7, 품질 거부 및 구간 비교 진단 | `completed` |
| [OmniFall Le2i 공식 구간 비교](2026-10-02_omnifall_le2i_benchmark_shared.md) | 203구간 G0 F1 53.33%, 미탐14개·품질 거부 영향·논문 비교 | `completed` |
| [추가 벤치마크 후보 조사](2026-10-02_additional_benchmark_search_shared.md) | TST·공식 Le2i/CAUCA 후보, 5개 시험 메타데이터·기존 높은 결과 검토 | 조사 `completed`, 새 점수 미측정 |
| [GMDCSA24 낮은 성능의 원인 후보](2026-10-02_omnifall_error_diagnosis_shared.md) | 시간·문맥 차이와 오류 진단, 라벨 의미 정정 | 진단 `completed`, 원인별 실험 미실행 |
| [OmniFall 공식 구간 비교](2026-10-02_omnifall_gmdcsa_benchmark_shared.md) | GMDCSA24-CS 93구간 고정 평가·논문 보고값 비교 | `completed` |
| [현재 모델 Le2i 평가](2026-10-02_le2i_current_evaluation_shared.md) | 전체127 G2/G0 F1 90.608/90.710%, 과거 대비 +7.010/+6.500%p | 평가·검산 `completed` |
| [Le2i 외부평가 데이터](2026-10-02_le2i_acquisition_shared.md) | 영상130개·주석 확보; 후속127개 평가 완료 | 확보·후속 평가 `completed` |
| [논문 근거·비판적 검토](2026-10-02_paper_evidence_review_shared.md) | URFD 전체70 평가·실제사례·관련연구6편·인용21개·FU 독립검산 | 검토·URFD `completed`, MCFD `paused` |
| [논문 벤치마크 정리](2026-09-03_13_benchmarks_provenance_shared.md) | R3: NTU/FU 검산·Le2i 출처·URFD 완료·MCFD 보류 | 검산·URFD `completed` |
| [S0-E 규칙 비교](2026-09-30_s0e_reconstruction_shared.md) | 72조건 완료,채택가능후보0 | `not_selected` |
| [S0-F 전이보조모델](2026-09-30_s0f_reconstruction_shared.md) | 고정35Dposterior·두후보비교·worst-view우선선택 | `in_progress` |
| [S0-F 자동 진행](2026-09-30_s0f_reconstruction_run_shared.md) | 전체실험현재단계 | `in_progress` |
| [핵심 Methods 본문](2026-09-29_paper_methods_main_shared.md) | R4: 실제 수행한 Methods 3.1–3.5절, 정규화 대상·J0/J1 출력·절 구성 정리 | 본문 정리, 연구 `in_progress` |
| [방법론 상세 연구 문서](2026-09-29_paper_methodology_shared.md) | R5: 수행 절차·재현 조건·완료 결과·별도 미실행 제안, 본문 근거 연결 | 문서 `completed`, 연구 `in_progress` |
| [S0-B/C 완료 분석](2026-09-29_s0bc_results_analysis_shared.md) | 상태안정화와낙상재현율상충관계·OOD한계 | `completed` |
| [S0-B 병렬 계승](2026-09-29_s0b_acceleration_shared.md) | 12epoch완료, 전환감소기준미달 | `not_selected` |
| [S0-B 실행 기록](2026-09-29_s0b_acceleration_run_shared.md) | 학습·선택·평가완료 | `completed` |
| [S0-C 상태 안정화](2026-09-29_s0c_acceleration_shared.md) | 44조건완료, A+EMA0.5/d3/m0.05선택 | `completed` |
| [S0-C 실행 기록](2026-09-29_s0c_acceleration_run_shared.md) | 고정평가·지연진단완료 | `completed` |
| [GT0 문맥 회복 타깃](2026-09-29_s0gt0_reconstruction_shared.md) | train/val371seq 생성·검산 완료 | `completed` |
| [S0-A 상태 분류](2026-09-29_s0a_reconstruction_shared.md) | 두 후보12epoch·고정 평가 완료, 시간적 안정성·OOD 한계 | `completed` baseline |
| [S0-A 실험 진행](2026-09-29_s0a_run_shared.md) | 학습·선택·평가 완료 | `completed` |
| [RGB 통합 기술 검증](2026-09-28_rgb_integration_shared.md) | 고정3개 중2개44windows 연결·검산,1개 품질 거부 | `completed` engineering smoke |
| [정합 수정·Global 학습](2026-09-28_global_motion_training_shared.md) | 데이터 수정·각50epoch·고정 평가 완료 | `completed`, G2 `not_selected` |
| [Global 학습 완료 기록](2026-09-28_global_motion_run_shared.md) | 전체 학습·최종 감사 완료 | `completed` |
| [RGB 입력 검사·정합](2026-09-28_rgb_data_preparation_shared.md) | 초기 검사 완료·후속 정합 수정 연결 | `completed` |
| [RGB·외부 데이터 준비](2026-09-23_rgb_data_acquisition_shared.md) | OOPS818·SAFER OOD30 확보·검증 완료 | `completed` |
| [전체 복구 실행 진행](2026-09-22_recovery_execution_shared.md) | S0-E 완료·미채택, S0-F 진행; 회복·외부 평가 잔여 | `in_progress` |
| [V3 R3 유효 근거 검산](2026-09-22_v3_valid_support_shared.md) | 전체 전처리·후속 검산 완료·사후 변경 한계 | `completed` |
| [V2 기준 모델 비교](2026-09-22_safer_v2_controls_shared.md) | A043 zero-shot·동일 조건 linear heads·고정 평가 | `completed` |
| [J0/J1 V1/V2](2026-09-22_joint_reconstruction_shared.md) | 개발·고정 평가·nested·최종 학습 방법 | `completed` |
| [V3 geometry](2026-09-22_safer_v3_reconstruction_shared.md) | Pilot 완료·좌표0 프레임 검사 미통과 | `not_selected` |
| [V3 controls](2026-09-22_safer_v3_controls_shared.md) | 전체 학습·고정 평가·독립 검산 완료 | `completed` |
| [V3 공동학습](2026-09-22_joint_v3_reconstruction_shared.md) | 개발·고정/nested평가·최종학습 완료 | `completed` |
| [FU 분류기 비교](2026-09-22_fu_classifier_reconstruction_shared.md) | 4종 nested·V3 reference 비교 완료 | `completed` |
| [최신 전체 재현 현황](2026-09-19_recovery_restart_shared.md) | 9월 20일 F0B·F1 검증 완료와 E01–E13 재현 범위 | `paused` |
| [F0B 후속 평가 결과](2026-09-19_f0b_postselection_shared.md) | 고정 test/OOD 평가와 독립 검산 | `completed` |
| [F1 문서 기반 계약](2026-09-20_f1_reconstruction_contract_shared.md) | 고정한 별도 재구현 설정·coverage·검증 결과 | `completed` |
| [F1 실행 기록](2026-09-20_f1_run_shared.md) | 네 후보 학습·고정 평가·독립 검산 완료 | `completed` |
| [F1 이후 실행 조건](2026-09-20_post_f1_execution_contracts_shared.md) | 후속 전체 단계의 별도 재구현 계약 | `paused` |
| [FU ZS0/ZS1](2026-09-20_fu_zs_reconstruction_shared.md) | 전체 무학습 전이 평가·독립 검산 | `completed` |
| [FU D0/D1/D2](2026-09-20_fu_probe_reconstruction_shared.md) | 동일 시간 처리의 특징 비교·독립 검산 | `completed` |
| [P0/P1 primitive](2026-09-20_primitive_reconstruction_shared.md) | P0 미채택·P1 학습·독립 검산 완료 | `completed` |
| [SAFER V2 전처리](2026-09-20_safer_v2_reconstruction_shared.md) | 전체3D 생성·1,007,723windows 전수 검산 완료 | `completed` |
| [SAFER V2 실행 진행](2026-09-20_safer_v2_run_shared.md) | 전처리 생성·검증 완료 | `completed` |
| [공유 통합본](2026-09-03_project_complete_summary_shared.md) | 연구 전체와 사용·검토한 논문·공개자료 46개를 한 파일로 정리한 공유 시작 문서 | 최신 통합본 R27 |
| [전체 연구 정리](2026-09-03_experiment_catalog_shared.md) | E01–E13 구조, 역사적 결과, 현재 업데이트와 다음 연구 | `paused` |
| [파이프라인 재학습 순서](2026-09-03_pipeline_retraining_recovery_shared.md) | 현재 재현 상태와 모델 의존성 | `paused` |
| [DSTE 백본 검증](2026-09-03_dste_checkpoint_recovery_shared.md) | 공식 백본의 구조 호환성과 ADL 평가 | 구조·성능 `completed` |
| [NTU 이후 실행·데이터 확보](2026-09-03_next_steps_dataset_download_shared.md) | NTU/FU 완료 상태와 SAFER 이후 데이터·weight 우선순위 | `in_progress` |

### 실험별 문서

| ID | 문서 | 상태 |
| --- | --- | --- |
| E01 | [NTU60 ADL 재현·보존](2026-09-03_01_ntu60_adl_shared.md) | 역사적·현재 `completed` |
| E02 | [FU-Kinect fall/lying](2026-09-03_02_fu_kinect_shared.md) | 역사적·현재 F0 `completed`, SAFER 단계로 이동 |
| E03 | [SAFER 전처리·temporal](2026-09-03_03_safer_preprocessing_shared.md) | V1·F1·P1·V2 controls 완료, V3 pilot `not_selected` |
| E04 | [J0/J1 공동학습](2026-09-03_04_joint_j0_j1_shared.md) | V1/V2 `completed`, V3 `paused` |
| E05 | [LaDy ablation](2026-09-03_05_lady_ablation_shared.md) | `not_selected` |
| E06 | [RGB·Global Motion](2026-09-03_06_rgb_global_motion_shared.md) | Global 재구현 `completed`, 새 G2 `not_selected` |
| E07 | [CAUCAFall](2026-09-03_07_caucafall_shared.md) | `completed` diagnostic |
| E08 | [OOPS](2026-09-03_08_omnifall_oops_shared.md) | `completed` diagnostic |
| E09 | [상태·회복·D1](2026-09-03_09_state_recovery_decoder_shared.md) | 상태 완료, recovery `not_selected` |
| E10 | [Le2i](2026-09-03_10_le2i_shared.md) | `completed` smoke |
| E11 | [외부평가 프로토콜](2026-09-03_11_external_protocol_shared.md) | `planned` |
| E12 | [VLM·TrackMemory·비식별](2026-09-03_12_vlm_trackmemory_shared.md) | 설계 완료, 통합 `planned` |
| E13 | [비교 모델·provenance](2026-09-03_13_benchmarks_provenance_shared.md) | R3 검산·URFD `completed`, MCFD `paused` |

공유 문서의 과거 수치는 삭제 전 실험 기록에 근거한 **기존 기록 기준**이다. 현재 재현이 완료된
수치와 혼동하지 않으며, 복구 후 새 결과는 별도 revision으로 갱신한다.

처음 공유하는 경우 [공유 통합본](2026-09-03_project_complete_summary_shared.md) 하나만 전달해도
전체 목표·실험·결론·현재 상태·비식별 설계와 다음 순서를 파악할 수 있다.
