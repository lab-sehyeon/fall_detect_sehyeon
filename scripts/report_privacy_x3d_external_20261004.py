"""Publish audited, same-video Ours/HFD/original X3D comparison (document R2)."""
import json,csv,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/fall_processed/RGB/privacy_x3d_external_20261004_r1'
HFD=ROOT/'data/fall_processed/RGB/hfd_reproduction_20261004_r1'
E=ROOT/'docs/internal/2026-10-04_hfd_x3d_external_comparison_evidence'
NAME='2026-10-04_hfd_x3d_external_comparison'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def pct(x):return f'{x*100:.2f}%'
def report():
    audit=read(OUT/'independent_audit.json');assert audit['passed']
    assert audit['evaluation_sha256']==sha(OUT/'evaluation.json')
    x=read(OUT/'evaluation.json');h=read(HFD/'evaluation.json');assert read(HFD/'independent_audit.json')['passed']
    summaries={name:{scope:dict(r['video'],videos=r['videos'],processed=r['processed']) for scope,r in ds.items()} for name,ds in h['summaries'].items()}
    summaries['privacy_x3d_uda_rgb']=x['summaries']
    names={'own':'사용자 DSTE–J1/G0','hfd_reproduction':'HFD 공식 코드 재학습','privacy_x3d_uda_rgb':'Privacy X3D-UDA, RGB 입력'}
    provenance={'own':'Frozen previous evaluation; NTU60 encoder, SAFER+FU fall learning',
      'hfd_reproduction':'Author C3D weights/code; new GMDCSA32 SVM; no target training',
      'privacy_x3d_uda_rgb':'Author UDA checkpoint and test.py; Kinetics700 RGB+depth; RGB external inference; prior resume ancestry not fully available'}
    with (OUT/'comparison.csv').open('w',newline='') as f:
        fields=['model','scope','videos','processed','tp','fp','fn','tn','precision_percent','recall_percent','f1_percent','accuracy_percent','provenance'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for model,ds in summaries.items():
            for scope,r in ds.items():w.writerow(dict(model=model,scope=scope,**{k:r[k] for k in ['videos','processed','tp','fp','fn','tn']},**{k+'_percent':100*r[k] for k in ['precision','recall','f1','accuracy']},provenance=provenance[model]))
    truth={r['id']:int(bool(r['episodes'])) for r in read(OUT/'ground_truth.json')}
    cases=[]
    for row in h['rows']+x['rows']:
        gt=truth[row['id']];pred=row['video_prediction'];tag='TP' if gt and pred else 'FN' if gt else 'FP' if pred else 'TN'
        cases.append(dict(model=row['model'],scope=row['scope'],id=row['id'],processed=row['processed'],ground_truth=gt,prediction=pred,case=tag))
    assert len(cases)==690
    with (OUT/'cases.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(cases[0]));w.writeheader();w.writerows(cases)
    (OUT/'comparison.json').write_text(json.dumps(dict(unit='video',summaries=summaries,provenance=provenance,cases=cases),ensure_ascii=False,indent=2)+'\n')
    table='| 모델 | Le2i130 영상 F1 | 정확도 | CAUCA100 영상 F1 | 정확도 |\n| --- | ---: | ---: | ---: | ---: |\n'
    for model,ds in summaries.items():
        a,b=ds['le2i130'],ds['cauca100'];table+=f'| {names[model]} | {pct(a["f1"])} | {pct(a["accuracy"])} | {pct(b["f1"])} | {pct(b["accuracy"])} |\n'
    conf='| 데이터셋 | TP | FP | FN | TN | Precision | Recall |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n'
    for scope,r in x['summaries'].items():conf+=f'| {scope} | {r["tp"]} | {r["fp"]} | {r["fn"]} | {r["tn"]} | {pct(r["precision"])} | {pct(r["recall"])} |\n'
    ip=ROOT/'docs/internal'/f'{NAME}_internal.md';s=ip.read_text();s=s.replace('상태: HFD `completed`, X3D 외부 평가 `in_progress`','상태: HFD·X3D 외부 평가 `completed`',1)
    s+='\n### R2 실제 실행 완료 및 검산\n\n'+table+'\nX3D 혼동행렬:\n\n'+conf
    s+='\n두 데이터셋의 저자 tools/test.py와 own_scipt/evaluation/evaluation.py가 모두 exit0으로 완료했다.\n전체230개 원본 logits, 예측·정답 목록과 sklearn 혼동행렬/F1/정확도를 독립 재계산했다.\n저자 scorer는 precision/recall을4자리로 반올림한 뒤 F1을 계산하므로, 표는 정수 혼동행렬의\nF1 원값을 사용하고 원본 출력도 함께 보존했다. 모델·설정·선택 PNG 등 계약 파일의 실행 전후\n해시 동일성을 확인했다. 입력 목록/정답/사용자 결과는 HFD 비교와 정확히 같다.\n'
    s+=f'공식583tensor 전체 exact 로딩, {len(audit["replays"])}개 영상의 직접 공식 모델 재추론으로 원본 test.py 출력과 대조했다.\n'
    s+='검산 표본은 데이터셋별 첫·마지막 영상 및 판정 경계에서 가장 가까운 영상이며 파라미터 변경 없음.\n'
    s+='원본 산출물은 data/fall_processed/RGB/privacy_x3d_external_20261004_r1 아래\nle2i130_results.pkl, cauca100_results.pkl, *_author_test.log, *_author_metrics.log, evaluation.json,\nindependent_audit.json, comparison.json/csv, cases.csv다. 독립 감사 코드는\nscripts/audit_privacy_x3d_external_20261004.py, 보고 코드는 scripts/report_privacy_x3d_external_20261004.py다.\n'
    s+='이전 HFD 산출물·R1 보고 검증 파일은 보존했다. R1 문서 스냅샷은 docs/internal에 보존하며\n현재 공유·내부 문서는 R2다. 공유에는 실행 조건·검증된 점수·한계만 옮겼다.\n'
    s+='X3D는 학습된 원본 파일을 그대로 사용했으며 새 학습·Le2i/CAUCA 적응·threshold 변경을 하지 않았다.\n원 논문의 depth-domain source 성능 재현까지 완료한 것으로 주장하지 않는다. 메타에 있는\nepoch37 선행 checkpoint는 미공개라 전체 학습 계보의 독립 인증 한계는 남는다.\n';ip.write_text(s)
    sp=ROOT/'docs/shared'/f'{NAME}_shared.md'
    event='| 모델 | Le2i130 사건 F1 | CAUCA100 사건 F1 |\n| --- | ---: | ---: |\n'
    for model in ['own','hfd_reproduction']:
        event+=f'| {names[model]} | {pct(h["summaries"][model]["le2i130"]["event"]["f1"])} | {pct(h["summaries"][model]["cauca100"]["event"]["f1"])} |\n'
    shared='''# HFD·Privacy X3D 외부 비교

문서 ID: DOC-20261004-hfd-x3d-external-comparison-R2  
날짜: 2026-10-04  
상태: HFD·X3D 외부 평가 `completed`

## 동일 영상의 낙상 분류 결과

Le2i130개(낙상99/비낙상31), CAUCAFall100개(낙상50/비낙상50)를 동일한 정답으로 비교했다.
사용자 모델의 기존 고정 결과를 유지하고, HFD 및 Privacy X3D를 실제 실행했다.
모든 영상이 분모에 포함되며 사용자 모델의 입력 처리 실패도 제외하지 않았다.
HFD와 X3D는230개 모두 처리가 완료되었다.

'''+table+'''
위 값은 **영상 단위** 낙상 F1과 정확도다. 사용자·HFD는 영상 안에 양성 경보 또는 clip이
하나라도 있으면 양성으로 판정한다. X3D는 원래 추론 방식대로5개 clip 점수를 평균한 뒤 판정한다.
특징 추출·분류 방법이 다른 전체 pipeline을 동일한 영상 정답으로 비교한 결과다.

## 실제 사용한 모델과 학습 출처

| 모델 | 사용한 자산·방법 | 낙상 학습 출처 | 이번 외부 평가에서의 변경 |
| --- | --- | --- | --- |
| 사용자 DSTE–J1/G0 | 기존 고정 인코더·adapter·분류 경로 | SAFER+FU, 인코더는 NTU60 사전학습 | 기존 검증 결과 재사용 |
| HFD | 저자 C3D 가중치·공식 특징 추출 함수, 새로 학습한 SVM | GMDCSA32, C3D는 Sports1M 사전학습 | source 전체로 최종 SVM 학습 및 영상·사건 출력 집계 추가 |
| Privacy X3D-UDA, RGB 입력 | 저자의 학습된 UDA 가중치와 원래 추론·평가 프로그램 | 공개 설정 기준 Kinetics-700 RGB+depth, X3D는 Kinetics400 사전학습 | RGB 외부 영상 적용, 추가 학습·적응 없음 |

HFD는 **공식 코드 기반 재학습(official-code reproduction)** 결과이며 저자가 배포한 최종 SVM
가중치의 결과로 표시하지 않는다. 원본 GMDCSA에 해당하는32개 영상의 파일명·프레임 수를
공식 실행 기록과 대조했다. 16프레임 특징360개로 수행한 공식 방식의5회70:30 검증에서
평균 낙상 F1은99.57%, 정확도는99.63%였다. 같은 영상의 조각이 학습·검증 양쪽에 들어갈 수
있으므로, 이 source 점수는 영상 독립 검증이나 외부 일반화 성능으로 해석하지 않는다.

Privacy X3D는 **순수 RGB 단독 학습본이 아닌 RGB+depth 공동 학습 UDA 모델**이다.
저자 원래 프로그램에서16프레임·간격5의 clip5개를 사용하고, RGB256×256 및 원래 정규화를
유지했다. 평균 분류 점수의 softmax 낙상 확률이0.5를 초과하면 양성으로 판정했다.
Le2i·CAUCA에 대한 학습, 무라벨 적응, 판정 기준 조정은 수행하지 않았다.

## X3D 혼동행렬

'''+conf+'''
## 검증과 해석의 범위

X3D의 두 데이터셋 모두 저자 원래 추론 프로그램과 평가 프로그램으로 실행했다.
저장된 전체230개 출력을 독립적으로 재계산하고, 대표 영상의 재추론 출력도 대조했다.
입력 영상 목록·정답·프레임 순서·픽셀 보존과 학습 가중치가 유지됨을 확인했다.
저자 평가 프로그램의 중간 반올림을 피하기 위해 표의 F1은 정수 혼동행렬에서 직접 계산했다.

이 조건에서는 사용자 모델의 영상 F1과 정확도가 두 비교 모델보다 높다. 다만 학습 데이터와
영상 집계 방식이 다르므로 동일 학습량에서 구조만의 효과를 입증한 비교로 해석하지 않는다.
X3D는 영상 전체를 이용하는 오프라인 분류이므로 실시간 사건 검출 성능을 입증한 것이 아니다.

X3D의 공개 설정·학습 목록에는 Le2i와 CAUCA가 없다. 이어서 학습하기 전의 선행 가중치는
공개되지 않아 전체 학습 이력을 독립적으로 완전히 인증한 것으로 표현하지 않는다.
이번 외부 평가 완료와 원 논문의 depth-domain 성능 재현 완료는 구분한다.
Le2i·CAUCA는 이전 진단에서 관찰한 자료이므로 ‘처음 보는 시험 데이터’라고 표현하지 않는다.

## 별도 사건 평가

'''+event+'''
사건 F1은 정답 시간 구간과 경보 시각의 일대일 대응으로 계산한다. HFD는 양성 clip의
상승 경보를 마지막 입력 프레임의 시각에 대응시켰다. X3D 원래 출력에는 사건시각이 없어
사건 F1을 임의로 생성하지 않았으며, 메인 세 모델 비교는 영상 단위 지표로 통일했다.
과거78.95%/87.50%는 각각38개/19개 부분집합 결과이므로 현재 전체 점수와 구분한다.

## 공개 근거

- [HFD 공식 코드](https://github.com/ekramalam/HFD_3DCNN)
- [GMDCSA-24 데이터 논문](https://pmc.ncbi.nlm.nih.gov/articles/PMC11416611/)
- [Privacy X3D 공식 코드](https://github.com/1015206533/privacy_supporting_fall_detection)
- [UMA-FD 논문](https://arxiv.org/abs/2308.12049)
'''
    sp.write_text(shared)
    assert not any(t in shared for t in ['/home/','CUDA_VISIBLE_DEVICES','scripts/','sha256','best_top1_acc_epoch'])
    for p in [ROOT/'docs/README.md',ROOT/'docs/shared/README.md']:
        s=p.read_text().replace('HFD 외부230개 완료; 승인된 X3D UDA 원본 RGB 평가 실행 중 | HFD `completed`, X3D `in_progress`','세 모델 동일230개 영상 비교·X3D 저자 원본 실행·독립 검산 완료 | `completed`');p.write_text(s)
    p=ROOT/'docs/internal/README.md';s=p.read_text().replace('HFD 외부230개 완료; X3D UDA 원본 승인·공식 프로그램 RGB 평가 실행 중','R2: 세 모델 동일230개 비교·X3D 저자 원본 실행·독립 감사 완료');p.write_text(s)
    receipt=dict(passed=True,document_id='DOC-20261004-hfd-x3d-external-comparison-R2',
      x3d_target_evaluation_executed=True,comparison_rows=6,case_rows=len(cases),
      x3d_audit_sha256=sha(OUT/'independent_audit.json'),hfd_audit_sha256=sha(HFD/'independent_audit.json'),
      shared_sha256=sha(sp),internal_sha256=sha(ip),comparison_sha256=sha(OUT/'comparison.csv'))
    (E/'report_validation_r2.json').write_text(json.dumps(receipt,indent=2)+'\n');print(table+'\n'+conf)
if __name__=='__main__':report()
