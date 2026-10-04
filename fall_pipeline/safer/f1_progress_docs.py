"""Generate paired experiment progress artifacts from the running F1 reports.

These dedicated generated documents never overwrite the manually maintained
contract/overview. Internal is written before its sanitized shared companion.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


def atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def publish(root, config, stage, epoch=None, metrics=None, failed=False, audited=False, project_root=None):
    project = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    date = datetime.now(ZoneInfo("Asia/Seoul")).date()
    revision_path = root / "doc_revision.json"
    previous = json.loads(revision_path.read_text())["revision"] if revision_path.exists() else 0
    shared_path = project / "docs/shared/2026-09-20_f1_run_shared.md"
    if shared_path.exists():
        match = re.search(r"DOC-20260920-f1-run-R(\d+)", shared_path.read_text())
        if match:
            previous = max(previous, int(match.group(1)))
    revision = previous + 1
    doc_id = f"DOC-20260920-f1-run-R{revision}"
    state = "paused" if failed else "completed" if audited else "in_progress"
    names = {"extract": "고정 DSTE dense feature 추출", "train": "네 후보 전체 학습·validation 선택",
             "epoch": "전체 validation epoch 결과(예비)", "evaluate": "고정 모델 test/OOD 평가",
             "complete": "전체 실험·독립 감사 완료", "failed": "실행 중단; 결과 미확정"}
    stage_title = names.get(stage, stage)
    shared = ["# SAFER F1 문서 기반 재구현 — 실행 결과\n\n",
              f"- 문서 ID: `{doc_id}`\n- 기준일: {date}\n"
              f"- 연구 상태: `{state}` — {stage_title}\n\n",
              "이 문서는 실제 실행 산출물에서 생성되는 진행 기록이다. 원본 코드·checkpoint의 완전 복원이"
              " 아니라 사전 고정한 별도 재구현 실험이며, 역사적 수치를 새 결과로 복사하지 않는다.\n\n",
              "## 방법\n\nFrozen DSTE/ADL, 64-frame temporal/temporal+spatial 표현과 plain/sqrt CE의 네 후보를"
              " 비교한다. Hidden512 residual block, batch128, AdamW lr0.001, 20epochs의 새 조건을 사용한다. "
              "Validation timeline macro-F1 → fall-vs-unstable AUPRC → lying F1로 선택하고 이후 고정 모델만"
              " test/OOD에 평가한다. Raw logits를 overlap mean한 뒤 softmax를 적용한다.\n"]
    if epoch is not None:
        shared.append(f"## 진행 중 결과\n\n전체 train/validation epoch {epoch}/20을 완료했다. "
                      "아래 수치는 예비 validation 결과이며 최종 모델 선택·holdout 결과가 아니다.\n")
    if metrics:
        shared.append("| 후보/평가 | Macro-F1 | Fall F1 | Fall-vs-unstable AUPRC | Lying F1 |\n| --- | ---: | ---: | ---: | ---: |\n")
        for name, row in metrics.items():
            def pct(value):
                return "N/A" if value is None else f"{100 * value:.3f}%"
            shared.append(f"| {name} | {pct(row['macro_f1'])} | {pct(row['fall_f1'])} | "
                          f"{pct(row['fall_vs_unstable_auprc'])} | {pct(row['lying_down_f1'])} |\n")
    if audited:
        report = json.loads((root / "final_report.json").read_text())
        shared.append(f"\n선택: {report['selection_lock']['candidate']}, epoch {report['selection_lock']['epoch']}. "
                      "입력·모델 불변성, 저장 prediction 지표 및 overlap 재구성의 독립 감사를 통과했다.\n")
    elif failed:
        shared.append("\n실행이 중단되어 결과를 확정하지 않는다. 완료된 선행 결과는 보존하고 실패 범위를 검토한다.\n")
    shared.append("\n## 해석 경계\n\n미커버 frame은 평가에서 제외하며 coverage를 별도 보고한다. Event 지표는 "
                  "사전 정의한 segment IoU≥0.1의 재구현 지표로서 역사적 event F1과 동일하지 않다. "
                  "아직 최종 감사 전이면 성능 재현 완료로 해석하지 않는다.\n\n"
                  "[재현 계약](2026-09-20_f1_reconstruction_contract_shared.md) · "
                  "[선행 F0B 결과](2026-09-19_f0b_postselection_shared.md)\n")
    internal = ["# F1 실행 자동 기록 — 내부\n\n", f"- 문서 ID: `{doc_id}`\n- 기준일: {date}\n- 연구 상태: `{state}`\n",
                f"- Updated UTC: {datetime.now(timezone.utc).isoformat()}\n- Run: `{root}`\n",
                f"- Config: `configs/f1_safer_document_reconstruction_v2.json`\n- Stage: `{stage}`\n",
                "- 이 전용 파일은 자동 생성되며 수동 계약 원장을 대체하지 않는다.\n",
                "- 실행: `CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 "
                "OPENBLAS_NUM_THREADS=2 conda run --no-capture-output -n fall_detect python -m "
                "fall_pipeline.safer.run_f1_reconstruction --stage all`; 동일 계약 재개에만 `--resume` 추가.\n",
                "- Evidence: `run_contract.json`, `status.json`, `cache/report.json`, `training/epoch_*.json`, "
                "`training/report.json`, `selection_lock.json`, `final_report.json`, `independent_audit.json`.\n"]
    if failed and (root / "failure.json").exists():
        internal.extend(["\n## Failure\n\n```json\n", (root / "failure.json").read_text(), "\n```\n"])
    internal.extend(["\n## 공유 반영 내용\n\n", "".join(shared).replace("](2026-", "](../shared/2026-")])
    atomic_text(project / "docs/internal/2026-09-20_f1_run_internal.md", "".join(internal))
    atomic_text(project / "docs/shared/2026-09-20_f1_run_shared.md", "".join(shared))
    atomic_text(revision_path, json.dumps({"revision": revision}) + "\n")
