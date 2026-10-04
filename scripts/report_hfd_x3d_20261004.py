"""Publish only audited HFD results; never fill an unexecuted X3D score."""
import csv,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/fall_processed/RGB/hfd_reproduction_20261004_r1'
E=ROOT/'docs/internal/2026-10-04_hfd_x3d_external_comparison_evidence'
def read(p):return json.loads(p.read_text())
def pct(v):return f'{v*100:.2f}%'
def report():
    audit=read(OUT/'independent_audit.json');assert audit['passed'];assert read(E/'x3d_preparation_verification.json')['passed'];ev=read(OUT/'evaluation.json');sm=ev['summaries'];source=read(OUT/'source_validation.json')
    with (OUT/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['model','scope','videos','processed','video_tp','video_fp','video_fn','video_tn','video_f1_percent','video_accuracy_percent','event_tp','event_fp','event_fn','event_f1_percent','provenance']);w.writeheader()
        for model,ds in sm.items():
            for scope,r in ds.items():
                v,e=r['video'],r['event'];w.writerow(dict(model=model,scope=scope,videos=r['videos'],processed=r['processed'],**{'video_'+k:v[k] for k in ['tp','fp','fn','tn']},video_f1_percent=v['f1']*100,video_accuracy_percent=v['accuracy']*100,**{'event_'+k:e[k] for k in ['tp','fp','fn']},event_f1_percent=e['f1']*100,provenance='Frozen existing result' if model=='own' else 'Official code reproduction; GMDCSA32 newly fitted SVM'))
    with (OUT/'cases.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['model','scope','id','processed','video_prediction','tp','fp','fn','alarm_times']);w.writeheader()
        for r in ev['rows']:w.writerow(dict(model=r['model'],scope=r['scope'],id=r['id'],processed=r['processed'],video_prediction=r['video_prediction'],**{k:r['event'][k] for k in ['tp','fp','fn']},alarm_times=';'.join(str(p['time']) for p in r['predictions'])))
    video='| 모델 | Le2i130 영상 F1 | 정확도 | CAUCA100 영상 F1 | 정확도 |\n| --- | ---: | ---: | ---: | ---: |\n'
    event='| 모델 | Le2i130 사건 TP/FP/FN | 사건 F1 | CAUCA100 사건 TP/FP/FN | 사건 F1 |\n| --- | ---: | ---: | ---: | ---: |\n'
    for model,title in [('own','사용자 DSTE–J1/G0'),('hfd_reproduction','HFD 공식 코드 재학습')]:
        a,b=sm[model]['le2i130'],sm[model]['cauca100'];video+=f'| {title} | {pct(a["video"]["f1"])} | {pct(a["video"]["accuracy"])} | {pct(b["video"]["f1"])} | {pct(b["video"]["accuracy"])} |\n'
        ae,be=a['event'],b['event'];event+=f'| {title} | {ae["tp"]}/{ae["fp"]}/{ae["fn"]} | {pct(ae["f1"])} | {be["tp"]}/{be["fp"]}/{be["fn"]} | {pct(be["f1"])} |\n'
    video+='| Privacy X3D-UDA, RGB 입력 | 미실행 | — | 미실행 | — |\n'
    ip=ROOT/'docs/internal/2026-10-04_hfd_x3d_external_comparison_internal.md';s=ip.read_text().replace('상태: `in_progress`','상태: HFD `completed`, X3D 외부 평가 `paused`',1)
    s+='\n## HFD 완료 및 독립 검산\n\n'+video+'\n'+event
    s+=f'\n총{audit["videos"]}개·{audit["frames"]} native frames·{audit["chunks"]} chunks 검산 완료.\n'
    s+='전체 선형 SVM 결정을 계수 내적으로 독립 재계산, 모든 PTS/rising-edge와 TP/FP/FN 검산,\n영상 confusion/F1/정확도는 sklearn으로 재계산했다. 세 영상은 원본 extractFeatures 함수로 재추출했다.\n'
    s+='원본230개 목록·정답·사용자 결과를 재사용하며 사용자·RTHFD 기존 실험은 덮어쓰지 않았다.\n원본 metric: data/fall_processed/RGB/hfd_reproduction_20261004_r1/evaluation.json,\nindependent_audit.json, comparison.csv, cases.csv. source_features.npz, svm.joblib,\nmodel_receipt.json과 각 영상 features.npz/prediction.json을 보존한다.\n'
    s+='실행: scripts/evaluate_hfd_reproduction_20261004.py freeze → train → evaluate;\nscripts/audit_hfd_reproduction_20261004.py로 독립 감사. 환경은 evidence/runtime_receipt.json.\n'
    s+='X3D는 원본 UDA 파일·공식 코드·입력/config 준비까지만 완료하며 선택 답변 전 외부 추론 없음.\n계획은 원본 tools/test.py와 own_scipt/evaluation/evaluation.py로 영상단위 평가다.\n공유에는 연구조건·완료점수·source/target 구분·한계만 반영하며 경로/로그/해시는 제외한다.\n';ip.write_text(s)
    sp=ROOT/'docs/shared/2026-10-04_hfd_x3d_external_comparison_shared.md';s=sp.read_text();s=s.replace('상태: HFD `in_progress`, X3D 평가 `paused`','상태: HFD `completed`, X3D 평가 `paused`');s=s.replace('외부230개 평가는 진행 중이다.','외부230개 평가와 독립 검산을 완료했다. 전체230개에서 HFD 처리가 완료되었다.')
    s=s.replace('아직 외부 성능을 산출하지 않았다.','같은230개 영상의 공식 입력 준비와 순서·픽셀 보존 검증까지 완료했다. 아직 외부 성능을 산출하지 않았다.')
    start=s.index('## 현재 검증된 사용자 결과');end=s.index('## 공개 근거',start)
    s=s[:start]+'''## 동일 영상에서의 비교 결과

영상 판정은 낙상 존재 여부를 평가한다. 사용자·HFD는 양성 경보 또는 clip이 하나라도 있으면
해당 영상을 양성으로 판정한다. X3D 원래 프로그램은 전체 영상의5개 clip 점수를 평균한다.
특징 추출과 분류는 각 모델의 공개 구현을 유지하고, 영상 집계는 위와 같이 명시한 전체 pipeline 비교다.

'''+video+'\n## 별도 사건 평가\n\n'+event+'''
사건 지표는 시간 구간과 경보 시각을 일대일 대응한다. HFD는16프레임 판정이 음성에서
양성으로 바뀌는 시점을 사용하며, 경보 시각은 해당 clip의 마지막 입력 프레임이다.
X3D의 원래 영상 판정에는 사건시각이 없어 위 사건 표에 넣지 않았다.

사용자 모델의 입력 처리 실패도 전체 분모에 남겼다. HFD 전 영상의 입력 픽셀·순서를 확인하고,
분류 출력·시간 대응·혼동 행렬을 독립 검산했다. 이 비교는 source 학습 자료가 서로 다른
전체 pipeline의 전이 성능이며, 동일 학습량에서 adapter 자체의 효과를 검증한 실험은 아니다.
X3D의 전체영상5clip 방식은 오프라인 분류이므로, 이를 실시간 사건 검출 능력과 동일시하지 않는다.
Le2i와 CAUCA는 과거 진단에서 관찰한 데이터이므로 ‘처음 보는 시험 데이터’로 표현하지 않는다.
이번 평가에서는 가중치·판정 조건을 고정하고 두 데이터셋으로 학습하거나 조정하지 않았다.
과거78.95%/87.50%는 각각38개/19개 부분집합 결과로 현재 전체 점수와 구분한다.

'''+s[end:];sp.write_text(s)
    for p in [ROOT/'docs/README.md',ROOT/'docs/shared/README.md']:
        s=p.read_text();s=s.replace('HFD 원본32개 source 검증99.57%·전체 외부 평가 중; X3D UDA 원본 확보 | HFD `in_progress`, X3D `paused`','HFD source 재학습·외부230개 완료·독립 검산; X3D 원본·입력 준비 | HFD `completed`, X3D `paused`');p.write_text(s)
    p=ROOT/'docs/internal/README.md';s=p.read_text().replace('HFD GMDCSA32 source 재현·동결 SVM 외부230개 실행 중; X3D 원본583tensor 검증','HFD GMDCSA32 source 재학습·외부230개 완료·독립 감사; X3D 원본583tensor·입력 준비');p.write_text(s)
    assert not any(x in sp.read_text() for x in ['/home/','data/source_archives','CUDA_VISIBLE_DEVICES','scripts/','sha256'])
    checks=dict(passed=True,audit=hashlib.sha256((OUT/'independent_audit.json').read_bytes()).hexdigest(),source_f1=source['mean_fall_f1'],x3d_target_evaluation_executed=False,shared_path=str(sp.relative_to(ROOT)),internal_path=str(ip.relative_to(ROOT)),shared_sha256=hashlib.sha256(sp.read_bytes()).hexdigest())
    (E/'report_validation.json').write_text(json.dumps(checks,indent=2)+'\n');print(video+'\n'+event)
if __name__=='__main__':report()
