# S0-B 시간축 보정 실험

- 문서 ID: `DOC-20260929-s0b-reconstruction-R2`
- 기준일: 2026-09-29
- 상태: `paused` — 원래 실행을 보존하고 마이크로배치512 실행으로 계승

현재 학습은 [512 전환 실행](2026-09-29_s0b_mb512_shared.md)에서 이어진다. 실질 배치1024와
모델·평가 기준은 유지한다. 아래는 원래 실행의 방법과 진행 이력이다.

S0-A가 가진 행동 정보를 유지하면서 시간적 일관성을 개선할 수 있는지 검증한다.
S0-A와 DSTE는 동결하고 hidden256, dilation1/2/4 residual temporal correction만 학습한다.
출력을zero-init하여 학습 전에는 S0-A와 정확히 같아야 한다. 두 모델은 같은 입력·평가 기준으로 비교한다.

12epoch, effective batch1024, sqrt CE를 적용하며 validation Macro-F1→SegmentF1@50→Edit로 선택한다.
S0-A보다 tuple이 좋은 epoch가 없으면 초기 모델을 유지한다. 채택하려면 validation에서
Macro-F1 하락≤0.5%p, SegmentF1@50/ Edit 각각≥2%p 개선, 상태 전환≥25% 감소,
fall recall 하락≤2%p를 모두 만족해야 한다. Test/OOD는 선택 후 평가하며 기준을 사후 조정하지 않는다.

소실된 optimizer·block 세부는 AdamW lr0.001, seed0, dropout0.5 등 사전 고정한 별도 설정이다.
과거 모델과 수치 동등성을 주장하지 않는다. 기존 [S0-A 결과](2026-09-29_s0a_reconstruction_shared.md)는
유지하며, 결과가 나오기 전 채택이나 실사용 성능을 주장하지 않는다.
이후 S0-C에서 두 모델의 같은 상태 안정화 조건을 비교한다.

1,024개 window 제한 실행에서 초기 동등성·실제 학습·동결 모델 불변이 통과했다.
전체 validation의857,528개 평가 프레임에서도 학습 전 logits와 지표가 S0-A와 정확히 일치했다.
이 검증을 통과해12epoch 학습을 시작했고 이후 선택·고정 평가를 진행한다.
[현재 실행 단계](2026-09-29_s0b_run_shared.md)를 별도로 기록하며 정식 성능 결과는 아직 없다.

근거: [MS-TCN 공식 residual block](https://github.com/yabufarha/ms-tcn/blob/0e418c029c2de1e90f6c54f45a0c186d8d9977b0/model.py),
[역사적 상태 연구](2026-09-03_09_state_recovery_decoder_shared.md),
[전체 참고문헌](2026-09-03_project_complete_summary_shared.md).
