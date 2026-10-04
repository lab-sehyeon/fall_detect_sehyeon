"""Write paired CAUCA100 reports only after the full independent audit passes."""
from pathlib import Path
import csv, hashlib, json, re, shutil

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/fall_processed/RGB/cauca100_20261003_r1'
STEM='2026-10-03_cauca100_own_evaluation'


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def pct(x):return f'{x*100:.2f}%'
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def main():
    audit=read(OUT/'independent_audit.json');ev=read(OUT/'evaluation.json');plan=read(OUT/'plan.json')
    assert audit['passed'] and audit['evaluation_sha256']==sha(OUT/'evaluation.json')
    assert audit['contract_sha256']==sha(OUT/'contract.json') and read(OUT/'status.json')['stage']=='completed'
    journal=read(OUT/'runner_journal.json');assert len(journal)==6 and all(r['returncode']==0 for r in journal)
    cfg=read(ROOT/'configs/cauca100_20261003_r1.json');old=read(ROOT/'configs/temporal_continuous_20261002_r1.json')
    unchanged={k:cfg[k]==old[k] for k in ['detector','pose_config','quality','lifting','inference','model','sampling']}
    assert all(unchanged.values())
    rows=[r for r in ev['rows'] if r['scope']=='cauca100'];summary=ev['datasets']['cauca100'];event=summary['event'];video=summary['video']
    assert len(rows)==summary['videos']==100 and summary['positive']==50
    gate=[r for r in rows if not r['quality_passed']];gate_fn=sum(r['event']['fn'] for r in gate)
    inferred_fn=[r for r in rows if r['quality_passed'] and r['event']['fn']]
    silent=[r for r in inferred_fn if not r['predictions']];timing=[r for r in inferred_fn if r['predictions']]
    assert gate_fn+len(inferred_fn)==event['fn']
    nonfall_fp=sum(r['event']['fp'] for r in rows if not r['episodes'])
    detected_fall_extra=sum(r['event']['fp'] for r in rows if r['event']['tp'])
    coverage_only=bool(gate) and all(r['quality']['bbox_coverage']<.8 and r['quality']['pose_coverage']<.8 and
                    r['quality']['pelvis_median']>=.3 and not r['quality']['insufficient_frames'] for r in gate)
    gate_note='거부된 모든 영상은 사람·관절 검출률이80% 미만이었다. 골반 confidence 중앙값0.3 기준과 최소 길이는 통과했다.' if coverage_only else '각 품질 거부 원인은 영상별 결과에 기록했다.'
    primary=table(['범위','영상 수','낙상 수','TP','FP','FN','정밀도','재현율','사건 F1'],
        [[name,ev['datasets'][scope]['videos'],ev['datasets'][scope]['positive'],
          *[ev['datasets'][scope]['event'][k] for k in ['tp','fp','fn']],
          *[pct(ev['datasets'][scope]['event'][k]) for k in ['precision','recall','f1']]]
         for name,scope in [('전체 CAUCA','cauca100'),('기존 시험 부분집합','cauca19')]])
    secondary=table(['영상 정확도','영상 정밀도','영상 재현율','영상 F1','특이도','영상 AP'],
                    [[pct(video[k]) for k in ['accuracy','precision','recall','f1','specificity','ap']]])
    by_id={p['id']:p for p in plan};groups=[]
    for subject in range(1,11):
        selected=[r for r in rows if by_id[r['id']]['subject']==subject]
        tp,fp,fn=[sum(r['event'][k] for r in selected) for k in ['tp','fp','fn']]
        assert len(selected)==10
        groups.append(dict(subject=subject,videos=len(selected),fall_events=sum(bool(r['episodes']) for r in selected),
                      quality_pass=sum(r['quality_passed'] for r in selected),tp=tp,fp=fp,fn=fn,
                      precision=tp/(tp+fp) if tp+fp else 0,recall=tp/(tp+fn),f1=2*tp/(2*tp+fp+fn)))
    subjects=table(['피험자','영상 수','품질 통과','TP','FP','FN','사건 F1'],
                   [[g['subject'],g['videos'],g['quality_pass'],g['tp'],g['fp'],g['fn'],pct(g['f1'])] for g in groups])
    errors=[]
    for r in rows:
        if not r['event']['fn'] and not r['event']['fp']:continue
        reason=('품질 거부' if not r['quality_passed'] else '낙상 경보 없음' if not r['predictions'] else
                '비낙상 영상에서 낙상 경보' if not r['episodes'] else '낙상 탐지와 추가 경보' if r['event']['tp'] else '경보 시각 불일치')
        errors.append([r['path'],r['event']['tp'],r['event']['fp'],r['event']['fn'],reason,'; '.join(f"{p['time']:.2f}" for p in r['predictions']) or '—'])
    error_table=table(['영상','TP','FP','FN','출력상 구분','경보 시각(초)'],errors)
    action_rows=[]
    for label,prefix in [('후방 낙상','backwards/'),('전방 낙상','forward/'),('왼쪽 낙상','side/FallLeft'),('오른쪽 낙상','side/FallRight'),('앉은 상태에서 낙상','side/FallSitting'),('비낙상 행동','adl/')]:
        selected=[r for r in rows if r['path'].startswith(prefix)]
        tp,fp,fn=[sum(r['event'][k] for r in selected) for k in ['tp','fp','fn']]
        action_rows.append([label,len(selected),tp,fp,fn,pct(tp/(tp+fn)) if tp+fn else '—'])
    action_table=table(['행동 유형','영상 수','TP','FP','FN','낙상 재현율'],action_rows)
    internal=ROOT/'docs/internal'/f'{STEM}_internal.md';shared=ROOT/'docs/shared'/f'{STEM}_shared.md'
    complete=f'''

## 전체 실행 완료와 독립 검산

상태: `completed`. 원본100개 AVI와100개 구간 정답의 이름·크기·CRC·SHA를 검증했다.
신규81개에 전체 영상 시간순 pipeline을 실행했고 기존19개 입력·예측은 검증 재사용했다.
실제 전체 분류 실행은 {summary['quality_pass']}개, 품질 거부는 {len(gate)}개다.
품질 거부도 전체100개 분모에 유지했다. 본 실행에 비교 모델 신규 추론은 없다.

{primary}

{secondary}

영상 confusion은 TP{video['tp']}/FP{video['fp']}/FN{video['fn']}/TN{video['tn']}이다.
전체 미탐 {event['fn']}건은 품질 거부 {gate_fn}건, 분류 후 무경보 {len(silent)}건,
경보는 있으나 정답 시간과 대응하지 않는 {len(timing)}건이다.
{gate_note}
오탐 경보 중 비낙상 영상 경보는 {nonfall_fp}건, 낙상 탐지 영상의 추가 경보는 {detected_fall_extra}건이다.

{action_table}

{subjects}

{error_table}

실행 명령:
```bash
CUDA_VISIBLE_DEVICES='' python scripts/prepare_cauca100_20261003.py all
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python scripts/run_cauca100_20261003.py all
CUDA_VISIBLE_DEVICES='' python scripts/report_cauca100_20261003.py
```

원본 취득은 공식 Range 접근에 필요한 승인된 네트워크 실행, 모델 단계는 승인된 물리GPU0에서 실행했다.
runner가 CPU 준비·채점·감사 단계에서는 CUDA_VISIBLE_DEVICES를 비운다.
기존 설정 일치: `{unchanged}`.

단계별 실행 기록:
```json
{json.dumps(journal,ensure_ascii=False,indent=2)}
```

독립 검산:
```json
{json.dumps(audit,ensure_ascii=False,indent=2)}
```

새 출력의 `evaluation.json`, `independent_audit.json`, `comparison_19_to_100.json`, `cases.csv`,
`contract.json`, `cache_reuse.json`, `source_snapshot.json`, `logs/`에 근거를 보존했다.
19개 부분집합의 영상별 결과와 전체 통계는 이전 temporal 평가와 정확히 같았다.
3D/분류 원시 출력, 품질 gate, PTS, 경보와 모든 집계는 CPU 별도 코드로 검산했다.
영상100개 결과를 공식19개 시험 점수나 과거100개 개발 진단 점수와 혼합하지 않았다.
공유본에는 결과·정답 체계·실패·해석 한계를 반영하고 내부 명령·경로·해시는 제외했다.
'''
    previous=internal.read_text().split('\n## 전체 실행 완료와 독립 검산')[0]
    internal.write_text(previous.replace('상태: `in_progress`','상태: `completed`',1)+complete)
    shared.write_text(f'''# CAUCA 전체 100개 영상의 사용자 모델 평가

문서 ID: DOC-20261003-cauca100-own-evaluation-R1  
갱신일: 2026-10-03  
상태: `completed`

**고정 DSTE→J1→G0 모델의 전체 CAUCA 낙상 사건 F1은 {pct(event['f1'])}다.**
원영상 100개(낙상 50개·비낙상 50개)를 평가했으며 {event['tp']}건 탐지, {event['fn']}건 미탐,
{event['fp']}건 오탐 경보가 발생했다. 전체 처리·채점과 독립 검산을 완료했다.

## 범위와 평가 방법

[CAUCAFall v4 공식 배포](https://data.mendeley.com/datasets/7w7fccy7ky/4)의 원본100개 영상을 확보했다.
기존 19개 평가와 같은 [고정 OmniFall 구간 정답](https://huggingface.co/datasets/simplexsigil2/omnifall/blob/83572a37b9e3081df8c06a56874b1d1f2a19386c/labels/caucafall.csv)을 사용했다.
원본 프레임별 이진 주석과 OmniFall의 시간 구간 주석은 구분된다. 이번 사건 채점에는 후자를 유지했다.
기존 시험 분할에 포함되지 않은 세 영상도 원본·시간 정합성을 확인해 포함했다.

기존19개 입력·예측은 동일성을 검증해 재사용했고 나머지81개는 새로 전처리·품질 판정했다.
모델·가중치·품질 기준·입력 순서·채점 규칙을 유지했으며 추가 학습이나 임계값 조정을 하지 않았다.
공식 OmniFall 분할의 train/val 영상도 이번에는 고정 모델의 평가 입력으로만 사용했다.
그 분할 이름이 이번 모델의 학습 데이터라는 의미는 아니다.

전체 영상의 실제 시간축에서 25Hz 입력을 구성하고, 64프레임 창을 8프레임 간격으로 분류했다.
낙상 클래스로 전환되는 경보를 정답 시작0.5초 전부터 종료3초 후까지 일대일로 대응해 사건 지표를 계산했다.
품질 거부는 무경보로 처리하되 전체 분모에 포함했다. ADL·회복 분기는 이번 평가에 포함되지 않는다.

## 낙상 사건 탐지 결과

{primary}

기존19개 영상의 예측·점수는 그대로 유지되었다. 전체100개와 기존19개는 시험 구성이 다르므로
두 점수 차이를 모델 자체의 개선이나 악화로 해석하지 않는다.

## 영상 단위 낙상/비낙상 분류

{secondary}

영상 안에서 낙상 예측이 한 번이라도 나오면 양성으로 판정한 보조 지표다.
영상 혼동행렬은 TP{video['tp']}/FP{video['fp']}/FN{video['fn']}/TN{video['tn']}이다.
반복 경보와 시각을 반영하는 사건 지표와 계산 단위가 다르다.

## 미탐과 품질 기준

전체100개 중 {summary['quality_pass']}개에서 분류를 실행했고 {len(gate)}개는 기존 관절 품질 기준에 걸렸다.
전체 미탐 {event['fn']}건 중 품질 거부는 {gate_fn}건, 분류를 실행했으나 낙상 경보가 없는 경우는
{len(silent)}건, 경보 시각이 정답 허용 구간과 대응하지 않는 경우는 {len(timing)}건이다.
{gate_note}
오탐 {event['fp']}건 중 비낙상 영상의 경보는 {nonfall_fp}건, 낙상 탐지 영상의 추가 경보는 {detected_fall_extra}건이다.

{error_table}

## 행동 유형별 결과

{action_table}

## 피험자별 결과

{subjects}

## 검증과 해석 범위

공식 원본100개 대응, 구간 정답, 프레임 시간순서, 추적·품질 판정, 3D 변환과 분류 연산,
영상별 경보와 지표를 독립 검산했다. 처리 실패 영상을 제거하여 성능을 높이지 않았다.
3D 변환은 전체 영상 문맥을 사용하는 오프라인 처리이므로 실시간 성능을 입증하는 결과는 아니다.
CAUCA는 과거 개발 진단 이력이 있어 처음 관찰하는 blind test로 주장하지 않는다.

이번에는 사용자 모델만 전체100개에서 평가했다. 비교 모델의 기존19개 점수를 이번100개 수치와
동일한 시험 범위의 비교표에 섞지 않는다. 저자의 원래 전체 평가 프로그램을 재현한 결과도 아니다.

[모델 결과 CSV]({STEM}_results.csv) · [100개 영상별 결과]({STEM}_cases.csv) ·
[피험자별 결과]({STEM}_subjects.csv)
''')
    exported=[]
    for scope,s in ev['datasets'].items():
        r=dict(model='DSTE_J1_G0',scope=scope,videos=s['videos'],fall_videos=s['positive'],quality_pass=s['quality_pass'])
        r.update({'event_'+k:v for k,v in s['event'].items()});r.update({'video_'+k:v for k,v in s['video'].items()});exported.append(r)
    for suffix,data in [('results',exported),('subjects',groups)]:
        with (shared.parent/f'{STEM}_{suffix}.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    shutil.copy2(OUT/'cases.csv',shared.parent/f'{STEM}_cases.csv')
    for p in [ROOT/'docs/README.md',ROOT/'docs/internal/README.md',ROOT/'docs/shared/README.md']:
        text=p.read_text();lines=text.splitlines()
        for i,line in enumerate(lines):
            if STEM in line:
                if line.startswith('|'):lines[i]=line.replace('전체 원영상 확보·기존19개 재사용·추가81개 DSTE/J1/G0 평가',f"전체100개·낙상50건; 사건 F1 {pct(event['f1'])}, 독립 검산 완료").replace('`in_progress`','`completed`')
                else:lines[i]=line.split(':',1)[0]+f": 공식100개 원본 확보·고정 DSTE/J1/G0 평가·독립 검산 완료, 사건 F1 {pct(event['f1'])}"
        p.write_text('\n'.join(lines)+'\n')
    print(primary);print(secondary);print('quality failure',len(gate),'gate FN',gate_fn,'inferred FN',len(inferred_fn))


if __name__=='__main__':main()
