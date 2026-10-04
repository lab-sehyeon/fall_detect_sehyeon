"""Publish verified external fall comparison tables; no model execution or tuning."""
from pathlib import Path
from datetime import datetime, timezone
import csv, json, re, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fall_pipeline.external import rgb_document_io as io
OUT=ROOT/'data/fall_processed/RGB/external_comparison_expansion_20261003_r1'
OLD=ROOT/'data/fall_processed/RGB/three_by_three_20261003_r1'
TOPIC='2026-10-03_external_comparison_expansion'
SHARED=ROOT/'docs/shared'
INTERNAL=ROOT/'docs/internal'/f'{TOPIC}_internal.md'
MODELS=['own','stgcnpp','msg3d','cnn1d']
MODEL_NAMES={'own':'사용자 DSTE/J1/G0','stgcnpp':'ST-GCN++','msg3d':'MS-G3D','cnn1d':'1D-CNN'}
NAMES={'le2i38':'Le2i-CS','gmdcsa37':'GMDCSA-24-CS','cauca19':'CAUCAFall-CS','le2i127':'Le2i127 보조',
       'edf':'EDF-CS','occu':'OCCU-CS','OOPS':'OOPS-Fall test'}
ORDER=['le2i38','gmdcsa37','cauca19','edf','occu','OOPS','le2i127']


def main():
    merged={};sources=[];cases=[];all_ids=set()
    for root, scopes, origin in [(OUT,['le2i38','gmdcsa37','cauca19','le2i127'],'new_comparator_inference_reused_own'),
                                *[(OLD/d,[d],'previously_completed') for d in ['edf','occu','OOPS']]]:
        a=io.read(root/'independent_audit.json');e=io.read(root/'evaluation.json')
        assert a['passed'] and a['evaluation_sha256']==io.sha(root/'evaluation.json')
        done=io.read(root/'comparators_completed.json');assert done['passed']
        if root!=OUT:
            ref=io.read(OUT/'comparators_completed.json')
            for name in MODELS[1:]:
                assert done['models'][name]['model_before']==done['models'][name]['model_after']==ref['models'][name]['model_before']
        plan={r['id']:r for r in io.read(root/'plan.json')};all_ids.update(plan)
        truth=io.read(root/'ground_truth.json')
        for scope in scopes:
            duration=sum(plan[r['id']]['source_duration_seconds'] for r in truth if r['scope']==scope)/3600
            for model in MODELS:
                s=e['summaries'][model][scope]
                merged[(scope,model)]=dict(dataset=scope,model=model,videos=s['videos'],fall_events=s['fall_events'],processed=s['processed'],
                    **s['event'],video_f1=s['video']['f1'],video_ap=s['video']['ap'],video_specificity=s['video']['specificity'],
                    duration_hours=duration,false_alarms_per_hour=s['event']['fp']/duration,execution_origin=origin)
        with (root/'cases.csv').open() as f:cases.extend(csv.DictReader(f))
        sources.append(dict(path=str(root.relative_to(ROOT)),evaluation_sha256=io.sha(root/'evaluation.json'),audit_sha256=io.sha(root/'independent_audit.json')))
    rows=[merged[(scope,model)] for scope in ORDER for model in MODELS]
    assert len(rows)==28 and len(cases)==797 and len(all_ids)==775
    result_csv=SHARED/f'{TOPIC}_results.csv';case_csv=SHARED/f'{TOPIC}_cases.csv'
    for path,data in [(result_csv,rows),(case_csv,cases)]:
        with path.open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    def table(scopes):
        lines=['| 시험 범위 | 영상 / 낙상 사건 | 사용자 F1 | ST-GCN++ F1 | MS-G3D F1 | 1D-CNN F1 |',
               '| --- | ---: | ---: | ---: | ---: | ---: |']
        for d in scopes:
            s=merged[(d,'own')]
            vals=' | '.join(f'{100*merged[(d,m)]["f1"]:.2f}%' for m in MODELS)
            lines.append(f'| {NAMES[d]} | {s["videos"]} / {s["fall_events"]} | {vals} |')
        return '\n'.join(lines)
    detail=['| 시험 범위 | 모델 | TP / FP / FN | Precision | Recall | 처리 완료 / 전체 |',
            '| --- | --- | ---: | ---: | ---: | ---: |']
    for d in ['le2i38','gmdcsa37','cauca19','le2i127']:
        for m in MODELS:
            r=merged[(d,m)]
            detail.append(f'| {NAMES[d]} | {MODEL_NAMES[m]} | {r["tp"]} / {r["fp"]} / {r["fn"]} | {100*r["precision"]:.2f}% | {100*r["recall"]:.2f}% | {r["processed"]} / {r["videos"]} |')
    wins=[]
    for d in ORDER:
        own=merged[(d,'own')];others=[merged[(d,m)] for m in MODELS[1:]]
        best=max(r['f1'] for r in others)
        wins.append(dict(dataset=d,own_f1=own['f1'],best_comparator_f1=best,difference=own['f1']-best,
                         lower_than_own=[r['model'] for r in others if r['f1']<own['f1']],
                         higher_than_own=[r['model'] for r in others if r['f1']>own['f1']]))
    done=io.read(OUT/'comparators_completed.json')
    audit=io.read(OUT/'independent_audit.json')
    evidence=dict(passed=True,sources=sources,result_rows=len(rows),case_memberships=len(cases),unique_videos=len(all_ids),
                  new_unique_videos=199,new_windows_per_comparator=6112,new_comparator_windows=18336,
                  elapsed_inference_seconds=done['elapsed_seconds'],comparisons=wins,
                  result_csv_sha256=io.sha(result_csv),case_csv_sha256=io.sha(case_csv),
                  generated_utc=datetime.now(timezone.utc).isoformat())
    io.save(OUT/'deliverable_validation.json',evidence)
    internal=INTERNAL.read_text().replace('- 상태: `in_progress`','- 상태: `completed`',1)
    internal+='\n## 완료 및 검증\n\n'
    internal+=f'- GPU 추론: 모델당199영상/6,112창, 합계18,336창. 소요 {done["elapsed_seconds"]:.2f}초. 가중치 전후 해시 동일.\n'
    internal+='- CPU 점수 산출 후 별도 코드로 시간축, 품질 gate, logits→확률/argmax, 사건 일대일 매칭, 전체 분모, 영상 confusion/AP를 검산했다. 기존 사용자 예측·점수는 원본과 동일하다.\n'
    internal+='- 실행: `CUDA_VISIBLE_DEVICES=\'\' python scripts/run_external_comparison_expansion_20261003.py score`; `CUDA_VISIBLE_DEVICES=\'\' python scripts/audit_external_comparison_expansion_20261003.py`; 마지막으로 본 report 스크립트.\n'
    internal+='- 원본 평가·감사: `'+str(OUT.relative_to(ROOT))+'/evaluation.json`, `independent_audit.json`, `deliverable_validation.json`. 개별 확률/logits·경보 및 전체 사례 CSV 보존.\n'
    internal+='- 기존 EDF/OCCU/OOPS의 독립 감사에 묶인 원본 점수를 재확인하고 동일 세 비교 모델의 tensor hash가 이번 실행과 같음을 확인해 연결했다. EDF0.05 사후 조건은 제외했다.\n'
    internal+=f'- 전체 표28행/사례797소속은775고유 영상이다. 주 여섯 시험은670영상이며 Le2i127 보조와의22중복을 단일 분모로 합산하지 않았다. 새 audit: `{io.sha(OUT/"independent_audit.json")}`.\n'
    internal+='\n'+table(ORDER)+'\n\n'+ '\n'.join(detail)+'\n'
    internal+='\n- 숫자의 통계적 유의성이나 아키텍처 원인 분리는 검증하지 않았다. 공개 가중치의 학습 출처는 확인 가능한 config/metadata 근거이며 기반 pose 사전학습 이미지 전체의 중복 검증이 아니다.\n'
    internal+='- 공유 반영: 연구 조건·전체 점수·분모·실패 포함·신규/재사용 구분·주장 한계. 제외: GPU/명령·로컬 경로·hash·준비 오류·전체 검색 응답.\n'
    INTERNAL.write_text(internal)
    shared=f'''# 학습 미사용 외부 데이터셋의 낙상 탐지 비교

- 문서 ID: `DOC-20261003-external-comparison-expansion-R1`
- 기준일: 2026-10-03
- 상태: `completed` — 세 비교 모델 신규 추론과 별도 코드 검산 완료

## 평가의 범위

**사용자 DSTE/J1/G0와 ST-GCN++, MS-G3D, 1D-CNN을 같은 외부 시험 영상에서 직접 비교했다.** 이번에는 Le2i·GMDCSA-24·CAUCAFall의 비어 있던 세 구조 비교를 실행했다. 사용자 모델의 시간축 검증 예측을 재사용하고 세 비교기마다199개 고유 영상,6,112개 시간창을 새로 추론했다. EDF·OCCU·OOPS는 앞서 완료한 결과를 무결성 확인 후 함께 정리했다. 문헌 보고값을 직접 실행한 점수로 대신하지 않았다.

| 모델 | 행동인식 학습 데이터 | 이번 외부 시험에서 추가 학습·보정 |
| --- | --- | --- |
| 사용자 DSTE/J1/G0 | DSTE: NTU60, J1: SAFER/FU, G0: SAFER | 없음 |
| ST-GCN++ | 공식 SAFER non-wheelchair subject train | 없음 |
| MS-G3D | 공식 SAFER non-wheelchair subject train | 없음 |
| 1D-CNN | 공식 SAFER non-wheelchair subject train | 없음 |

공개 config와 배포 기록에 따르면 위 외부 데이터셋들은 네 모델의 행동인식 학습에 포함되지 않는다. 두 GCN은 checkpoint 내부 학습 설정도 확인했고, CNN은 공식 동봉 config·학습 코드가 근거다. CNN checkpoint 자체에는 학습 이력이 들어 있지 않다. [SAFER 공식 코드](https://github.com/safer-activities/SAFER-Activities), [공식 가중치](https://huggingface.co/datasets/SAFER-Activities/SAFER-Activities-Weights)

## 주 결과: 낙상 사건 F1

{table(ORDER[:-1])}

Le2i·GMDCSA·CAUCA는 이번 신규 비교이고 EDF·OCCU·OOPS는 기존 완료 비교다. 각 행은 데이터셋 전체가 아닌 고정 시험 목록이다. EDF·OCCU는 각각2개 긴 영상 속16개 낙상 사건이며, 시험 피험자는 각각1명이다. 영상 수가 많거나 적다는 이유로 행을 통합해 일반화 성능 하나로 축약하지 않았다. [고정 시험 분할·정답](https://huggingface.co/datasets/simplexsigil2/omnifall/tree/83572a37b9e3081df8c06a56874b1d1f2a19386c)

EDF의 사용자 기본값은 pelvis confidence0.30 조건이다. 사후0.05 완화 실험의64.29%는 위 기본 비교에 섞지 않았다.

## Le2i127 별도 범위

{table(['le2i127'])}

기존 원주석 기반127영상/96낙상 범위를 같은 세 모델로 추가 비교했다. 공식 논문에서 보고한130영상/99낙상과 동일한 목록이 아니다. 기존 범위에는 주석 헤더를 해석할 수 없었던3영상이 포함되지 않는다. Le2i-CS와22영상이 겹치므로 두 행을 독립 데이터셋처럼 합치면 안 된다. 전체 추가 실행199영상은 세 CS 시험94영상과 이127영상의 합집합이다.

## 이번 비교의 검출·오경보 내역

{chr(10).join(detail)}

TP는 정답 낙상과 시간상 일대일 대응된 경보, FP는 대응되지 않은 경보, FN은 놓친 정답 낙상이다. 처리 실패 영상도 전체 분모에 포함하므로 처리 완료 영상만의 성능이 아니다. F1이 높다는 사실만으로 재현율도 더 높다고 해석하지 말고 위 TP/FN과 함께 확인해야 한다.

## 관측된 차이

- Le2i-CS: 사용자 모델은22낙상 중15개를 검출했고 세 비교기는20~22개를 검출했다. 사용자 미탐7개는 모두 기존 품질 기준에서 거부된 영상이다. 품질 기준을 완화하면 이7개를 모두 검출할 수 있다는 뜻은 아니다.
- GMDCSA: 사용자와 ST-GCN++의 TP는11개로 같지만 FP가 각각3개와1개여서 사용자 F1이 낮았다. 사용자 모델은1D-CNN보다 높은 F1을 기록했다.
- CAUCA: 사용자는 TP7/FP0, ST-GCN++는 TP9/FP3, MS-G3D는 TP6/FP0,1D-CNN은 TP8/FP1이다. 사용자 F1은 두 GCN보다 높고1D-CNN보다1.39%p 낮았다. 따라서 모든 낙상을 가장 많이 검출한 모델이라는 해석은 맞지 않는다.
- 이번 세 주 시험에서 사용자 모델이 세 비교기를 모두 상회한 데이터셋은 없다. 외부 평가가 실행 가능하다는 사실과 사용자 모델의 성능 우월성이 입증됐다는 주장은 구분한다.

## 입력과 평가 규칙

1. 원영상 PTS에 맞춘25Hz 시간축으로 전체영상을 시간순서대로 입력했다. 정답 낙상 위치로 영상을 자르지 않았다.
2. 사용자 모델은64프레임/stride8, 세 비교기는 동일한 종료시각까지 최근48프레임을 사용했다. 공식 구조·가중치·모델별 공간 전처리를 유지했다.
3. 원 다중클래스 argmax를 낙상/기타로 대응했다. 사용자 fall index1, 세 비교기는9다. target별 학습·보정·임계값 선택은 하지 않았다.
4. 낙상 출력이 처음 켜지는 시각을 경보로 삼았다. 정답 시작0.5초 전부터 종료3초 후까지 일대일 대응하며 불응기간은0초다. 반복 경보는 FP가 될 수 있다.
5. 사용자 입력 품질 기준은 bbox/pose coverage0.8, pelvis median0.3으로 유지했다. 세2D 비교기에 불필요한3D 품질 기준을 적용하지 않았다. 실패는 무경보로 처리하고 정답 분모를 유지했다.
6. 보조 지표는 영상별 any-fall F1, 최대 낙상 확률 AP, 영상 specificity다. 모든 영상이 낙상 양성인 EDF·OCCU의 영상 AP는 판별 성능 근거로 사용하지 않는다. 전체 P/R/F1과 시간당 오경보는 CSV에 수록했다.

## 검증 및 해석 범위

- 가중치 불변, 전체 시험 목록·정답, 시간순서, 원 logits의 확률·argmax, 사건·영상 지표를 별도 코드로 재계산했다. 사용자 점수는 기존 검증 결과와 동일하다.
- 처리 방식이 다르다는 이유로 비교를 배제하지 않았다. 다만 이번 실행은 공통 YOLOv8x/ViTPose-B 관절에 공식 분류기를 연결한 외부 평가다. SAFER 원저자의 ViTPose-H까지 포함한 원형 시스템 전체의 재현은 아니다.
- “미학습”은 명시된 행동인식 학습 범위 기준이다. pose 사전학습 이미지 전체의 샘플 중복은 감사하지 않았고, 이전 진단에서 관찰한 데이터셋이므로 완전히 손대지 않은 blind test는 아니다.
- 사용자 모델은 전체영상 문맥의3D lifting과 정규화를 사용한다. 인과적 실시간 속도·경보 지연 성능으로 해석하지 않는다.
- 학습 데이터량·출력 클래스 수가 달라 구조 하나의 우열을 분리하는 실험은 아니다. 작은 피험자·낙상 수의 차이를 통계적으로 확정된 우월성으로 주장하지 않는다.
- 데이터셋 선정은 공개 시험 목록·정답·입력 시간 검증·가중치 실행 가능성을 기준으로 했다. 높은 결과만 골라 제시하지 않고 기존 낮은 EDF/OCCU/OOPS 결과도 포함했다.

## 산출물

- [전체28개 조건의 수치 CSV]({TOPIC}_results.csv)
- [전체797개 평가 소속의 사례 CSV]({TOPIC}_cases.csv) —775고유 영상, 중복 범위 별도 표기
- [기존 EDF·OCCU·OOPS 비교](2026-10-03_three_models_three_new_datasets_shared.md)
- [PoseC3D 두 가중치와의 기존 비교](2026-10-03_shared_unseen_external_shared.md)

데이터 출처: [GMDCSA-24 원문](https://pmc.ncbi.nlm.nih.gov/articles/PMC11416611/), [CAUCAFall v4 공식 배포](https://data.mendeley.com/datasets/7w7fccy7ky/4), [OmniFall 원문](https://arxiv.org/abs/2505.19889). 현행 안내를 확인했으며 실제 시험 목록·주석은 기존에 고정한 배포본을 유지했다.
'''
    (SHARED/f'{TOPIC}_shared.md').write_text(re.sub(r'(?<=[가-힣])(?=[0-9])', ' ', shared))
    for path in [ROOT/'docs/README.md', SHARED/'README.md']:
        lines=path.read_text().splitlines()
        lines=[line.replace('`in_progress`','`completed`') if f'{TOPIC}_shared.md' in line else line for line in lines]
        path.write_text('\n'.join(lines)+'\n')
    path=ROOT/'docs/internal/README.md'
    text=path.read_text().replace('Le2i·GMDCSA·CAUCA의 세 구조 신규 추론·기존 사용자 예측 재사용, 진행 중','199영상×세 구조 신규 추론·검산 완료, 여섯 외부 시험 및 Le2i127 보조 비교')
    path.write_text(text)
    print(json.dumps(evidence,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
