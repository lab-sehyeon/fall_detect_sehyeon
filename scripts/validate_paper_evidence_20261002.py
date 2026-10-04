"""Cross-check deliverable claims, links, figures, and locked experiment provenance."""
from pathlib import Path
import sys,re,json,xml.etree.ElementTree as ET,csv
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fall_pipeline.external import rgb_document_io as io
from fall_pipeline.external.paper_external_20261002 import setup

def main():
    internal=ROOT/'docs/internal';shared=ROOT/'docs/shared';audit=internal/'2026-10-02_evidence_audit';checks={}
    a=internal/'2026-10-02_paper_evidence_review_internal.md';b=shared/'2026-10-02_paper_evidence_review_shared.md'
    texts={p:p.read_text() for p in [a,b]};docid='DOC-20261002-paper-evidence-review-R1'
    checks['paired_document_id']=all(docid in t for t in texts.values())
    checks['paired_completed_scope']=all('`completed`' in t and 'MCFD 평가 `paused`' in t for t in texts.values())
    benchmark=[shared/'2026-09-03_13_benchmarks_provenance_shared.md',internal/'experiments/2026-09-03_13_benchmarks_provenance_internal.md']
    checks['benchmark_r3']=all('DOC-20260903-exp13-benchmark-provenance-R3' in p.read_text() for p in benchmark)
    for p in [b,benchmark[0]]:
        text=p.read_text();checks[p.stem+'_no_internal_paths']=not any(x in text for x in ['/home/','/tmp/','docs/internal/','CUDA_VISIBLE_DEVICES','sha256','conda '])
        bad=[]
        for dest in re.findall(r'\]\(([^)]+)\)',text):
            if dest.startswith(('https://','http://','#')):continue
            target=(p.parent/dest.split('#')[0]).resolve()
            if not target.exists():bad.append(dest)
        checks[p.stem+'_relative_links']=not bad
        if bad:print('missing links',bad)
    fu=io.read(audit/'fu_checkpoint_replay.json');raw=io.read(audit/'raw_feature_replay.json');urfd=io.read(audit/'urfd_independent_audit.json')
    checks['independent_audits']=fu['passed'] and raw['passed'] and urfd['passed']
    checks['fu_numbers']=abs(fu['f1']['j1']-.9333333333333333)<1e-12 and '−1.013~+2.189' in texts[b]
    checks['urfd_numbers']=urfd['metrics']['tp']==15 and urfd['metrics']['fp']==6 and urfd['metrics']['fn']==15 and urfd['metrics']['tn']==34 and '58.824%' in texts[b]
    checks['full_denominator']=urfd['full_denominator']==70 and sum(x['total'] for x in urfd['coverage'].values())==70
    checks['mcfd_held']=not io.read(internal/'2026-10-02_mcfd_evaluation_scope.json')['sequence_level_approved'] and not (ROOT/'data/fall_processed/RGB/external_mcfd_20261002_r1/evaluation.json').exists()
    checks['urfd_user_scope']=io.read(internal/'2026-10-02_urfd_input_scope.json')['adl-37']=='retain_as_processing_failure'
    exports=[]
    for name in ['safer_ood_cases','urfd_cases']:
        for ext in ['png','pdf','svg']:
            p=ROOT/f'docs/images/paper_evidence_20261002/{name}.{ext}';assert p.stat().st_size>1000
            if ext=='svg':ET.parse(p)
            if ext=='png':assert p.read_bytes().startswith(b'\x89PNG')
            if ext=='pdf':assert p.read_bytes().startswith(b'%PDF')
            exports.append(dict(path=str(p.relative_to(ROOT)),sha256=io.sha(p),bytes=p.stat().st_size))
    checks['six_figure_exports']=len(exports)==6
    checks['actual_case_provenance']=io.read(audit/'case_provenance.json')['passed'] and io.read(audit/'urfd_case_provenance.json')['passed']
    bib=(shared/'2026-10-02_paper_core_references.bib').read_text()
    keys=re.findall(r'^@\w+\{([^,]+),',bib,flags=re.M)
    checks['bib21_unique']=len(keys)==len(set(keys))==21 and bib.count('{')==bib.count('}')
    with (shared/'2026-10-02_related_work_matrix.csv').open() as f:rows=list(csv.DictReader(f))
    checks['six_related_papers']=len(rows)==6 and all(r['citation_key'] in keys for r in rows)
    snapshots=io.read(audit/'benchmark_r2_snapshot.json')
    checks['historical_r2_snapshots']=all(io.sha(audit/'benchmark_r2_snapshot'/Path(p).name)==h for p,h in snapshots.items())
    config,root=setup('urfd');io.locked(config);checks['final_experiment_contract']=True
    report=dict(passed=all(checks.values()),checks=checks,figures=exports,document_sha256={str(p.relative_to(ROOT)):io.sha(p) for p in [a,b,*benchmark]},
        evaluation_sha256=io.sha(root/'evaluation.json'))
    io.save(audit/'deliverable_validation.json',report);print(json.dumps({k:v for k,v in report.items() if k!='figures'},indent=2));assert report['passed']

if __name__=='__main__':main()
