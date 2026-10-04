"""Descriptive paired-case diagnosis, not a new segment scoring protocol."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fall_pipeline.external import rgb_document_io as io
from fall_pipeline.external import omnifall_le2i_continuous_20261002 as run
from fall_pipeline.external import omnifall_le2i_20261002 as segment


def main():
    cfg=run.setup();io.locked(cfg)
    current=io.read(run.OUTPUT/'evaluation.json');audit=io.read(run.OUTPUT/'independent_audit.json')
    io.require(audit['passed'] and audit['evaluation_sha256']==io.sha(run.OUTPUT/'evaluation.json'),'audit missing')
    previous=io.read(segment.OUTPUT/'evaluation.json')
    plans={r['id']:r for r in io.read(segment.OUTPUT/'plan.json')}
    positive={plans[r['id']]['path']:r for r in previous['rows'] if r['fall']}
    rows=[]
    for row in current['rows']:
        if not row['episodes']:continue
        prior=positive[row['path']];gt=row['episodes'][0];head=row['heads']['G0']
        alarms=[p['time'] for p in head['predictions']]
        inside=[t for t in alarms if gt['fall_start']<=t<=gt['fall_end']]
        rows.append(dict(path=row['path'],segment_positive=bool(prior['prediction']),
            segment_quality=prior['quality_passed'],segment_class=prior['predicted_class'],
            continuous_quality=row['quality_passed'],continuous_event_tp=head['tp'],
            continuous_alarms=alarms,alarms_inside_gt=inside,
            late_after_end_seconds=[t-gt['fall_end'] for t in alarms if t>gt['fall_end']],
            gt=gt))
    io.require(len(rows)==22,'paired fall scope')
    gained=[r for r in rows if not r['segment_positive'] and r['continuous_event_tp']]
    result=dict(descriptive_only=True,causal_ablation=False,segment_reducer_added=False,
        caveat='same22 fall cases but segment classification and tolerated event matching are different decisions',
        positive_cases=22,prior_segment_positive=sum(r['segment_positive'] for r in rows),
        continuous_event_tp=sum(r['continuous_event_tp'] for r in rows),
        newly_detected_cases=len(gained),newly_missed_cases=sum(r['segment_positive'] and not r['continuous_event_tp'] for r in rows),
        newly_detected_with_alarms_inside_gt=sum(bool(r['alarms_inside_gt']) for r in gained),
        newly_detected_quality_rejection_resolved=sum(not r['segment_quality'] for r in gained),
        newly_detected_prior_quality_pass=sum(r['segment_quality'] for r in gained),
        all_detected_with_alarms_inside_gt=sum(bool(r['alarms_inside_gt']) for r in rows),rows=rows,
        provenance={str(p.relative_to(ROOT)):io.sha(p) for p in [run.OUTPUT/'evaluation.json',
            run.OUTPUT/'independent_audit.json',segment.OUTPUT/'evaluation.json',segment.OUTPUT/'plan.json',Path(__file__)]})
    io.save(run.OUTPUT/'paired_case_diagnosis.json',result)
    print({k:v for k,v in result.items() if k not in ['rows','provenance']})


if __name__=='__main__':main()
