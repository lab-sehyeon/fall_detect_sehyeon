# S0-C 상태 안정화 비교 결과

- 문서 ID: `DOC-20260929-s0c-acceleration-R2`
- 기준일: 2026-09-29
- 상태: `completed` — 44조건 비교·선택·고정 평가·검산 완료

[병렬계승 S0-B](2026-09-29_s0b_acceleration_shared.md)와동결S0-A에대해
동일44조건의상태안정화비교를완료했다. [원래방법](2026-09-29_s0c_reconstruction_shared.md)의
후보·채택기준·validation선택·선택후test/OOD평가·지연진단은변경하지않았다.

6개eligible중**S0-A+EMA0.5/연속3프레임/margin0.05**가선택됐다.
Test SegmentF1@50 49.304%,Edit66.851%,전환18.482/분으로A대비전환81.589%감소다.
다만test낙상재현율65.396%로A보다3.220%p낮아졌으므로validation기준통과를
낙상경보안전성보장으로해석하지않는다. OOD Macro-F1도36.380%로일반화한계가남는다.

[상세 결과·지연 진단·연구 한계](2026-09-29_s0bc_results_analysis_shared.md) ·
[최종 실행 기록](2026-09-29_s0c_acceleration_run_shared.md).
