"""Export audited numeric tables only; never modify frozen evaluation outputs."""
import csv
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fall_pipeline.external import rgb_document_io as io


def main():
    out = ROOT/'data/fall_processed/RGB/usdrl_external_20261004_r1'
    audit = io.read(out/'independent_audit.json')
    io.require(audit['passed'] and audit['evaluation_sha256'] == io.sha(out/'evaluation.json'), 'unverified evaluation')
    io.require(audit['cases_sha256'] == io.sha(out/'cases.csv'), 'unverified cases')
    evaluation = io.read(out/'evaluation.json')
    rows = []
    for model, scopes in evaluation['summaries'].items():
        for scope, s in scopes.items():
            rows.append(dict(model=model, dataset=scope, videos=s['videos'], processed=s['processed'],
                             gt_fall_events=s['fall_events'], predicted_fall_videos=s['predicted_fall_videos'],
                             fall_windows=s.get('fall_windows', ''), windows=s.get('windows', ''),
                             event_tp=s['event']['tp'], event_fp=s['event']['fp'], event_fn=s['event']['fn'],
                             event_precision_percent=100*s['event']['precision'],
                             event_recall_percent=100*s['event']['recall'], event_f1_percent=100*s['event']['f1'],
                             video_tp=s['video']['tp'], video_fp=s['video']['fp'], video_fn=s['video']['fn'],
                             video_tn=s['video']['tn'], video_f1_percent=100*s['video']['f1'],
                             video_accuracy_percent=100*s['video']['accuracy']))
    shared = ROOT/'docs/shared'
    results_path = shared/'2026-10-04_usdrl_external_results.csv'
    cases_path = shared/'2026-10-04_usdrl_external_cases.csv'
    with results_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (out/'cases.csv').open(newline='') as stream:
        reader = csv.DictReader(stream)
        cases = list(reader)
        fields = reader.fieldnames
    io.require(len(cases) == 230, 'case denominator')
    with cases_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(cases)
    io.require(io.sha(cases_path) == audit['cases_sha256'], 'case export differs')
    with results_path.open(newline='') as stream:
        actual = list(csv.DictReader(stream))
    io.require(len(actual) == 4, 'summary rows')
    for row, expected in zip(actual, rows):
        io.require(row == {k: str(v) for k, v in expected.items()}, 'summary export differs')
    receipt = dict(passed=True, evaluation_sha256=audit['evaluation_sha256'],
                   independent_audit_sha256=io.sha(out/'independent_audit.json'),
                   exported_summary_sha256=io.sha(results_path), exported_cases_sha256=io.sha(cases_path),
                   report_code_sha256=io.sha(Path(__file__)), videos=230, windows=6286)
    io.save(out/'report_validation.json', receipt)
    io.save(out/'status.json', dict(stage='completed', time=datetime.now(timezone.utc).isoformat(),
                                   videos=230, windows=6286, independent_audit_passed=True,
                                   report_validation_passed=True, no_training=True))
    print(receipt)


if __name__ == '__main__':
    main()
