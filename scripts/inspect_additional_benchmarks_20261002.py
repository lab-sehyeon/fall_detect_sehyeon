"""Read fixed official test metadata to assess compatibility, without inference."""
from pathlib import Path
import csv
import hashlib
import json
import statistics
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'data/source_archives/OmniFall/gmdcsa24_cs_20261002_r1'
OUT = ROOT/'data/source_archives/OmniFall/candidate_metadata_20261002_r1'
PIN = '83572a37b9e3081df8c06a56874b1d1f2a19386c'
sys.path.insert(0, str(OLD/'runtime'))
import pyarrow.parquet as pq


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    available = {r['rfilename'] for r in json.loads((OLD/'hf_info.json').read_text())['siblings']}
    files = []; reports = []
    for dataset in ['le2i', 'caucafall', 'up_fall', 'edf', 'occu']:
        wanted = [f'splits/cs/{dataset}/test.csv', f'labels/{dataset}.csv',
                  f'parquet/{dataset}-cs/test-00000-of-00001.parquet']
        for name in wanted:
            assert name in available, name
            path = OUT/name
            url = f'https://huggingface.co/datasets/simplexsigil2/omnifall/resolve/{PIN}/{name}'
            if not path.exists():
                request = urllib.request.Request(url, headers={'User-Agent':'FoundSkelModel-benchmark-inspection/1.0'})
                with urllib.request.urlopen(request, timeout=60) as response:
                    data = response.read(12*1024*1024+1)
                assert len(data) <= 12*1024*1024, 'metadata size limit'
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            files.append(dict(path=name,url=url,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        rows = pq.read_table(OUT/wanted[2]).to_pylist()
        split = list(csv.DictReader((OUT/wanted[0]).open()))
        raw = list(csv.DictReader((OUT/wanted[1]).open()))
        paths = {r['path'] for r in split}
        assert {r['path'] for r in rows} == paths
        target = [r for r in raw if r['path'] in paths]
        # Float32 Parquet times require a tolerance comparison, not exact decimal text equality.
        a = sorted(target, key=lambda r:(r['path'],float(r['start']),int(r['label'])))
        b = sorted(rows, key=lambda r:(r['path'],float(r['start']),int(r['label'])))
        assert len(a) == len(b)
        for x,y in zip(a,b):
            assert all(str(x[k]) == str(y[k]) for k in ['path','label','subject','cam'])
            assert all(abs(float(x[k])-float(y[k])) <= 1e-4*max(1,abs(float(x[k]))) for k in ['start','end'])
        fall = [r for r in rows if r['label']==1]
        durations = [float(r['end'])-float(r['start']) for r in fall]
        nonpositive = [r for r in rows if float(r['end']) <= float(r['start'])]
        report = dict(config=dataset+'-cs',split='test',videos=len(paths),segments=len(rows),
                      fall=len(fall),nonfall=len(rows)-len(fall),subjects=sorted({r['subject'] for r in rows}),
                      cameras=sorted({r['cam'] for r in rows}),
                      fall_duration_min=min(durations),fall_duration_median=statistics.median(durations),
                      fall_duration_max=max(durations),fall_shorter_than_2_52=sum(d<2.52 for d in durations),
                      paths=sorted(paths),csv_parquet_scope_match=True,model_evaluated=False,
                      nonpositive_duration_segments=nonpositive)
        reports.append(report)
        print(json.dumps({k:v for k,v in report.items() if k not in ['paths','nonpositive_duration_segments']}
                         | {'nonpositive_duration_count':len(nonpositive)}),flush=True)
    final = dict(passed=True,hf_revision=PIN,candidates=reports,files=files,
                 model_inference=False,downloaded_bytes=sum(x['bytes'] for x in files),
                 passed_scope='metadata cross-check only; annotation anomalies retained without exclusion')
    (OUT/'report.json').write_text(json.dumps(final,indent=2,ensure_ascii=False)+'\n')


if __name__ == '__main__':
    main()
