"""Write the paired research report only from three independently audited runs."""
from pathlib import Path
import csv,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fall_pipeline.external import rgb_document_io as io
OUT=ROOT/'data/fall_processed/RGB/three_by_three_20261003_r1'
TOPIC='2026-10-03_three_models_three_new_datasets'
NAMES=dict(own='사용자 모델 DSTE/J1/G0',stgcnpp='ST-GCN++',msg3d='MS-G3D',cnn1d='1D-CNN')
DATASETS=dict(edf='EDF-CS',occu='OCCU-CS',OOPS='OOPS-Fall 시험 분할')

def write_csv(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    summaries=[];cases=[];evidence={};table=[];details=[]
    for dataset,title in DATASETS.items():
        path=OUT/dataset;audit=io.read(path/'independent_audit.json');evaluation=io.read(path/'evaluation.json')
        assert audit['passed'] and audit['evaluation_sha256']==io.sha(path/'evaluation.json')
        evidence[dataset]=dict(audit_sha256=io.sha(path/'independent_audit.json'),evaluation_sha256=audit['evaluation_sha256'])
        plan=io.read(path/'plan.json');seconds=sum(r['source_duration_seconds'] for r in plan)
        cases+=list(csv.DictReader((path/'cases.csv').open()))
        values=[]
        for model,name in NAMES.items():
            s=evaluation['summaries'][model][dataset];v=s['event'];values.append(f"{100*v['f1']:.2f}%")
            row=dict(dataset=dataset,model=model,videos=s['videos'],fall_events=s['fall_events'],processed=s['processed'],**v,
                     video_f1=s['video']['f1'],video_ap=s['video']['ap'],video_specificity=s['video']['specificity'],
                     duration_hours=seconds/3600,false_alarms_per_hour=v['fp']/(seconds/3600))
            summaries.append(row)
            details.append(f"| {title} | {name} | {s['processed']}/{s['videos']} | {v['tp']} | {v['fp']} | {v['fn']} | {100*v['precision']:.2f} | {100*v['recall']:.2f} | {100*v['f1']:.2f} |")
        table.append('| '+title+' | '+' | '.join(values)+' |')
    summary=ROOT/f'docs/shared/{TOPIC}_results.csv';casefile=ROOT/f'docs/shared/{TOPIC}_cases.csv'
    write_csv(summary,summaries);write_csv(casefile,cases)
    text='''# 서로 다른 비교 모델 3종의 새 외부 데이터 평가

문서 ID: DOC-20261003-three-models-three-new-datasets-R1
상태: completed

사용자 모델 DSTE/J1/G0와 구조가 서로 다른 ST-GCN++, MS-G3D, 1D-CNN을 EDF, OCCU, OOPS-Fall에서 직접 평가했다. 세 비교 모델은 SAFER 공식 학습 가중치이며, 다른 학습 버전의 같은 구조를 여러 모델로 세지 않았다. 이전 Le2i·GMDCSA·CAUCA 평가는 이번 수치에 섞지 않았다.

## 낙상 사건 F1

| 외부 데이터 | 사용자 모델 | ST-GCN++ | MS-G3D | 1D-CNN |
| --- | ---: | ---: | ---: | ---: |
'''+ '\n'.join(table)+'''

## 시험 범위와 수행 조건

- EDF와 OCCU는 각각 공식 CS 시험의 긴 영상 2개를 사용했다. 두 데이터셋은 같은 연구진·피험자를 공유하며, 이 시험 분할은 각각 한 피험자와 두 시점으로 구성된다.
- OOPS는 공식 시험 CSV716행의 경로 중복을 제거한 고유572개 영상을 한 번씩 처리했다. 원본 주석의 낙상636사건을 사용했다. 이 경로 집합은 배포 시험 parquet와 동일하지만, 중복 행이 포함된 공식 구간 평가의 분모와는 다르다.
- UP-Fall은 공식 다운로드 할당량 제한으로 취득이 중단되어 OOPS로 대체했다. 성능을 본 후 대상을 교체하지 않았다. MCFD 보류 결정은 유지했다.
- 사용자 모델 행동 학습은 NTU60·SAFER·FU, 비교 모델 행동 학습은 SAFER이다. EDF·OCCU·OOPS는 네 모델의 행동 학습 및 이번 평가의 임계값 조정에 사용하지 않았다. pose 사전학습 원본 이미지 수준의 중복 여부까지 확인한 주장은 아니다.
- ST-GCN++/MS-G3D는 가중치 내부 학습 설정과 공식 코드의 일치를 확인했다. 1D-CNN은 가중치 내부 이력이 없으므로 공식 동반 설정을 근거로 학습 출처를 기재한다.

## 시간순 입력과 공통 판정

원본 영상의 프레임 시각을 따라 25Hz 입력을 만들고, 각 시점보다 미래인 원본 프레임을 선택하지 않았다. YOLOv8x와 ViTPose-B로 추출한 한 사람의 추적 pose를 네 모델이 공유했다. 여러 사람이 등장하는 영상에서도 정답 인물의 위치를 입력으로 제공하지 않았다. 비교 모델 학습에 쓰인 ViTPose-H와는 차이가 있다.

사용자 모델은 기존 MotionAGFormer 3D 변환과 DSTE→J1→G0를 유지했다. 64프레임 창을 8프레임 간격으로 평가한다. 세 비교 모델은 같은 판정 시점에서 끝나는 원래의 48프레임 연속 창을 사용하고, 각 공식 전처리를 유지한다. GCN 계열은 XY와 confidence를 사용하고, 1D-CNN은 좌표 및 motion 변환을 사용한다. 1D-CNN의 원래 고정 이미지 폭 기반 motion 정규화도 유지했다.

각 모델의 원래 다중 클래스 argmax를 fall/non-fall로 매핑했다. fall 예측의 시작점을 경보로 두고, 정답 낙상 시작0.5초 전부터 종료3초 후까지 일대일 사건 매칭을 적용했다. 품질 거부·짧은 입력 등 처리 실패는 경보 없음으로 두고 전체 정답 사건을 분모에 유지했다. 모델마다 입력 품질 요구가 다르므로 이 결과는 전체 파이프라인 비교이며 아키텍처만의 효과를 분리한 실험은 아니다.

사용자 모델의 3D lifting과 전체 시퀀스 정규화는 미래 문맥을 사용한다. 입력 프레임 순서를 지켰다는 사실을 인과적 실시간 동작이나 실시간 지연 검증으로 해석하지 않는다.

## 사건별 집계

| 데이터 | 모델 | 처리 영상/전체 | TP | FP | FN | Precision(%) | Recall(%) | F1(%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
'''+ '\n'.join(details)+f'''

[전체 지표 CSV]({summary.name}) · [영상별 모든 모델의 경보·정오 분류]({casefile.name})

## 해석 범위와 검증

사건 F1을 주 지표로 사용한다. EDF/OCCU처럼 모든 영상에 낙상이 있는 시험에서는 영상 단위 specificity가 정의되지 않으며, 영상 단위 F1·AP만으로 낙상과 비낙상의 구별 능력을 주장하지 않는다. 같은 영상 안의 반복 경보는 사건 FP에 포함된다.

각 실행의 입력 순서, 시험 분모, 고정 가중치, 실제 출력 확률, 사건 TP/FP/FN과 F1을 독립 집계기로 재검산했다. OOPS는 과거 연결 검사3개와 입력 진단 이력이 있어 최초 비공개 시험으로 부르지 않는다. 제한된 EDF/OCCU 피험자와 OOPS 영상 간 종속성 때문에 이 표를 모집단 수준의 통계적 우월성 입증으로 해석하지 않는다.

## 출처

- [SAFER 공식 구현](https://github.com/safer-activities/Safer-Activities) 및 [공식 배포 가중치](https://huggingface.co/datasets/SAFER-Activities/SAFER-Activities-Weights).
- [EDF/OCCU 원자료·원논문 정보](https://zenodo.org/records/15494102).
- [OOPS 원자료](https://oops.cs.columbia.edu/data/) 및 [OmniFall 고정 시험 목록·주석](https://huggingface.co/datasets/simplexsigil2/omnifall/tree/83572a37b9e3081df8c06a56874b1d1f2a19386c).
'''
    # Internal first, then export only research facts to the shared document.
    internal=ROOT/f'docs/internal/{TOPIC}_internal.md'
    with internal.open('a') as f:f.write('\n## 평가 및 독립 감사 완료\n\n'+json.dumps(evidence,indent=2)+'\n\n'+ '\n'.join(table)+'\n\n상태: completed. 세 데이터별 감사 PASS 및 모든 입력 계약 검증 후 공유 문서와 CSV를 생성했다.\n')
    internal.write_text(internal.read_text().replace('상태: in_progress','상태: completed',1))
    (ROOT/f'docs/shared/{TOPIC}_shared.md').write_text(text)
    for file in ['docs/README.md','docs/shared/README.md']:
        p=ROOT/file;lines=p.read_text().splitlines()
        for i,line in enumerate(lines):
            if TOPIC+'_shared.md' in line:
                lines[i]=line.replace('ST-GCN++·MS-G3D·1D-CNN과 EDF·OCCU·OOPS 외부 비교','네 모델×EDF·OCCU·OOPS 직접 추론·독립 검산').replace('`in_progress`','`completed`')
        p.write_text('\n'.join(lines)+'\n')
    io.save(OUT/'deliverable_validation.json',dict(passed=True,results_rows=len(summaries),case_rows=len(cases),audits=evidence,results_sha256=io.sha(summary),cases_sha256=io.sha(casefile)))
    print('\n'.join(table))

if __name__=='__main__':main()
