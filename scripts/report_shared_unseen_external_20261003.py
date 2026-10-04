"""Build the paired study report only from independently audited evaluations."""
from pathlib import Path
import csv
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fall_pipeline.external import rgb_document_io as io

BASE=ROOT/'data/fall_processed/RGB'
SHARED=ROOT/'docs/shared'
INTERNAL=ROOT/'docs/internal'
SCOPES=['le2i38','gmdcsa37','cauca19']
NAMES={'le2i38':'Le2i-CS','gmdcsa37':'GMDCSA-CS','cauca19':'CAUCA-CS'}
MODEL_NAMES={'own':'사용자 DSTE + J1 + G0','safer_posec3d':'PoseC3D — SAFER 학습','ntu_posec3d':'PoseC3D — NTU60 학습'}


def main():
    roots={'ntu_posec3d':BASE/'shared_unseen_external_20261003_r1',
           'safer_posec3d':BASE/'safer_posec3d_external_20261003_r1'}
    data={};audits={}
    for name,p in roots.items():
        audits[name]=io.read(p/'independent_audit.json');assert audits[name]['passed']
        assert io.sha(p/'evaluation.json')==audits[name]['evaluation_sha256']
        data[name]=io.read(p/'evaluation.json')
    assert data['ntu_posec3d']['summaries']['own']==data['safer_posec3d']['summaries']['own']
    summaries={'own':data['ntu_posec3d']['summaries']['own'],
               **{name:d['summaries']['posec3d'] for name,d in data.items()}}
    io.save(INTERNAL/'2026-10-03_shared_unseen_verified_evaluations.json',dict(summaries=summaries,audits=audits))
    rows=[]
    for scope in SCOPES:
        for model in MODEL_NAMES:
            d=summaries[model][scope]
            for unit in ['event','video']:
                m=d[unit]
                rows.append(dict(dataset=NAMES[scope],model=model,unit=unit,videos=d['videos'],positive=d['positive'],processed=d['processed'],
                    tp=m['tp'],fp=m['fp'],fn=m['fn'],tn=m.get('tn',''),precision_pct=100*m['precision'],recall_pct=100*m['recall'],
                    f1_pct=100*m['f1'],ap_pct=100*m['ap'] if 'ap' in m else ''))
    with (SHARED/'2026-10-03_shared_unseen_combined_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    case_rows=[]
    records={name:{(r['scope'],r['id']):r for r in d['rows'] if r['model']=='posec3d'} for name,d in data.items()}
    own=[r for r in data['ntu_posec3d']['rows'] if r['model']=='own']
    for r in own:
        c=dict(dataset=NAMES[r['scope']],video=r['id'],fall=int(bool(r['episodes'])))
        for model in MODEL_NAMES:
            x=r if model=='own' else records[model][(r['scope'],r['id'])]
            c.update({model+'_processed':int(x['processed']),model+'_tp':x['event']['tp'],model+'_fp':x['event']['fp'],model+'_fn':x['event']['fn'],
                      model+'_video_prediction':x['video_prediction'],model+'_alarms':';'.join(f"{p['time']:.2f}" for p in x['predictions'])})
        case_rows.append(c)
    with (SHARED/'2026-10-03_shared_unseen_combined_cases.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(case_rows[0]));writer.writeheader();writer.writerows(case_rows)
    diagnostics={}
    for scope in SCOPES:
        rs=[x for x in own if x['scope']==scope];ds={}
        for model in ['safer_posec3d','ntu_posec3d']:
            pairs=[(a,records[model][(scope,a['id'])]) for a in rs]
            positive=[(a,b) for a,b in pairs if a['episodes']]
            ds[model]=dict(both_detected=sum(bool(a['event']['tp']) and bool(b['event']['tp']) for a,b in positive),
                own_only=sum(bool(a['event']['tp']) and not b['event']['tp'] for a,b in positive),
                comparator_only=sum(not a['event']['tp'] and bool(b['event']['tp']) for a,b in positive),
                both_missed=sum(not a['event']['tp'] and not b['event']['tp'] for a,b in positive),
                own_quality_rejections_recovered=sum(not a['processed'] and bool(b['event']['tp']) for a,b in positive))
        diagnostics[scope]=ds
    report=['# 두 모델 모두 학습하지 않은 외부 데이터셋 비교', '',
        '- 문서 ID: `DOC-20261003-shared-unseen-external-R1`',
        '- 기준일: 2026-10-03',
        '- 상태: `completed` — 공식 가중치 두 종류의 새 추론 및 독립 검산 완료', '',
        '## 평가의 범위', '',
        '**Le2i-CS·GMDCSA-CS·CAUCA-CS의 동일한94개 영상에서 사용자 모델과 공식 PoseC3D 두 종류를 비교했다.** '
        '비교 모델의 예측은 직접 실행해 산출했으며, 문헌의 구간 분류 수치를 사건 점수와 섞지 않았다. '
        '사용자 모델은 직전 시간축 검증을 마친 연속 입력 예측을 무결성 확인 후 재사용했다.', '',
        '| 모델 | 행동인식 학습 데이터 | 출력 | 외부 시험에 대한 추가 학습 |',
        '| --- | --- | --- | --- |',
        '| 사용자 DSTE + J1 + G0 | DSTE: NTU60; J1: SAFER/FU; G0: SAFER | 4클래스 | 없음 |',
        '| PoseC3D — SAFER 학습 | SAFER non-wheelchair subject train | 15클래스 | 없음 |',
        '| PoseC3D — NTU60 학습 | NTU60 cross-subject train | 60클래스 | 없음 |', '',
        'SAFER 모델은 낙상과 일상행동을 학습한 직접 비교기이고, NTU 모델은 일반 행동인식에서 낙상으로의 전이를 보여주는 보조 비교기다. '
        '두 모델의 구조는 PoseC3D SlowOnly-R50 joint이며 각각의 저자 공개 가중치를 고정했다. '
        '[PoseC3D 논문](https://openaccess.thecvf.com/content/CVPR2022/html/Duan_Revisiting_Skeleton-Based_Action_Recognition_CVPR_2022_paper.html), '
        '[NTU 공식 모델](https://github.com/kennymckormick/pyskl/blob/f2bf3a6b08e2e8dec744692d64efdb187fd6719a/configs/posec3d/README.md), '
        '[SAFER 공식 추론 안내](https://github.com/safer-activities/SAFER-Activities/blob/994ed688ce9e491245ee96c1665e948c6ce6c74d/inference/README.md).', '',
        '여기서 “미사용”은 행동인식 모델의 명시된 학습 데이터에 해당 세 데이터셋이 포함되지 않았다는 뜻이다. '
        'SAFER는 자체 촬영 데이터와 ImViA 외부시험을 구분한다. 기반 모델의 모든 사전학습 이미지와의 '
        '샘플 단위 중복까지 입증한 것은 아니다. [SAFER 원문](https://arxiv.org/abs/2609.08038)', '',
        '시험은 각 데이터셋 전체가 아니라 고정된 OmniFall CS 시험 영상이다. '
        'Le2i38영상은 낙상22개/비낙상16개, GMDCSA37영상은17개/20개, CAUCA19영상은9개/10개다. '
        '이번 시험 영상들은 각각 피험자2명·1명·2명에서 나온다. '
        '[고정 시험 분할·정답](https://huggingface.co/datasets/simplexsigil2/omnifall/tree/83572a37b9e3081df8c06a56874b1d1f2a19386c).', '',
        '## 주 결과: 사건 단위 낙상 검출', '',
        '| 데이터셋 | 영상 / 낙상 사건 | 사용자 F1 | PoseC3D-SAFER F1 | PoseC3D-NTU F1 | 사용자−SAFER |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for scope in SCOPES:
        a=summaries['own'][scope];b=summaries['safer_posec3d'][scope];c=summaries['ntu_posec3d'][scope]
        report.append(f"| {NAMES[scope]} | {a['videos']} / {a['positive']} | {a['event']['f1']*100:.2f}% | {b['event']['f1']*100:.2f}% | {c['event']['f1']*100:.2f}% | {(a['event']['f1']-b['event']['f1'])*100:+.2f}%p |")
    report+=['', '| 데이터셋 | 모델 | TP / FP / FN | Precision | Recall |', '| --- | --- | ---: | ---: | ---: |']
    for scope in SCOPES:
        for model in MODEL_NAMES:
            m=summaries[model][scope]['event']
            report.append(f"| {NAMES[scope]} | {MODEL_NAMES[model]} | {m['tp']} / {m['fp']} / {m['fn']} | {100*m['precision']:.2f}% | {100*m['recall']:.2f}% |")
    report+=['', 'TP는 시간 허용 범위 안에서 일대일 대응된 낙상, FP는 대응되지 않은 경보, FN은 놓친 낙상이다. '
        '같은 낙상 영상에서 경보가 여러 번 켜지면 한 번만 TP이고 나머지는 FP가 될 수 있다.', '',
        '## 보조 결과: 영상 단위', '',
        '영상에 낙상 경보가 한 번이라도 있으면 양성이다. 사건 위치가 맞지 않아도 영상 정답만 맞을 수 있으므로 '
        '주 결과와 구분한다. AP는 영상별 최대 낙상 확률로 계산했다.', '',
        '| 데이터셋 | 모델 | 영상 F1 | AP | TP / FP / FN / TN | 처리 완료 / 전체 |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for scope in SCOPES:
        for model in MODEL_NAMES:
            d=summaries[model][scope];m=d['video']
            report.append(f"| {NAMES[scope]} | {MODEL_NAMES[model]} | {100*m['f1']:.2f}% | {100*m['ap']:.2f}% | {m['tp']} / {m['fp']} / {m['fn']} / {m['tn']} | {d['processed']} / {d['videos']} |")
    report+=['', '## 입력·판정·평가 규칙', '',
        '1. 원영상의 시각을 확인한25Hz 시간축과 동일한 YOLOv8x/ViTPose-B의2D 관절을 공유한다. '
        '정답 구간에 맞춰 영상을 잘라 주지 않는다.',
        '2. 사용자는64프레임 연속 창/stride8, PoseC3D-SAFER는 같은 경보 시각까지의 최근48프레임 연속 창을 사용한다. '
        'PoseC3D-NTU는 동일64프레임 범위 안에서 공식48프레임 표본을10회 추출한다. 각 표본 내부 시간순서는 유지된다.',
        '3. 비교기의 공식 공간 전처리·신뢰도 heatmap·좌우반전·확률 평균을 적용한다. SAFER는2개 view, NTU는20개 view다. '
        'SAFER 공개 가중치는44epoch의 표준 non-wheelchair 모델 하나를 사전에 지정했다.',
        '4. 원래 다중클래스 argmax를 낙상/기타로 대응한다. 사용자 fall index1, SAFER 모델 index9, '
        'NTU 모델 index42(A43 falling)다. 임계값·모델·정답은 target 성능에 맞춰 조정하지 않았다.',
        '5. 공통 경보 시각은 `(63+8k)/25`초다. 비낙상에서 낙상으로 변하는 시점에 경보를 내고, '
        '정답 시작0.5초 전부터 종료3초 후까지 일대일 매칭한다. 불응기간은0초다.',
        '6. 사용자 모델의 기존3D 입력 품질 거부는 무경보로 집계한다. PoseC3D에는 불필요한3D 품질 gate를 '
        '강제하지 않는다. 결측2D는 원래 zero/zero-score 입력으로 전달하며 전체 시험 분모를 유지한다.', '',
        '공식 [SAFER48프레임 설정](https://github.com/safer-activities/pyskl/blob/85525521b85a44c5df79873102192225c9565edc/configs/posec3d/safer_activity_xsub/non-wheelchair.py)과 '
        '[NTU 설정](https://github.com/kennymckormick/pyskl/blob/f2bf3a6b08e2e8dec744692d64efdb187fd6719a/configs/posec3d/slowonly_r50_ntu60_xsub/joint.py)을 '
        '이 공통 외부 평가에 연결했다. 원논문의 전체clip/전용외부추론/threshold 조건을 그대로 재현한 실험이라고 부르지 않는다.', '',
        '## 같은 낙상에서의 차이', '',
        '| 데이터셋 | 사용자·SAFER 모두 검출 | 사용자만 검출 | SAFER만 검출 | 모두 미탐 | 사용자 품질 거부 중 SAFER 검출 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for scope in SCOPES:
        x=diagnostics[scope]['safer_posec3d']
        report.append(f"| {NAMES[scope]} | {x['both_detected']} | {x['own_only']} | {x['comparator_only']} | {x['both_missed']} | {x['own_quality_rejections_recovered']} |")
    report+=['', '이 표는 관측된 오류 차이를 분해한 것이며, 차이의 원인을 adapter·활성화 함수·3D 변환 중 '
        '어느 하나로 입증하는 실험은 아니다. 영상별 경보와 처리 상태는 아래 전체 사례 표에 남겼다.', '',
        '## 결과를 해석할 수 있는 범위', '',
        '- 세 데이터셋에서 고정된 서로 다른 학습 모델의 외부 전이 성능을 직접 비교했다. '
        '같은 학습량·클래스 수의 구조 비교, 모든 외부 데이터에서의 우월성 또는 기존 논문 대비 SOTA 주장은 아니다.',
        '- 공통2D frontend를 쓰므로 원저자의 HRNet(NTU 모델) 또는 ViTPose-H(SAFER 모델)를 포함한 '
        '원형 시스템 전체의 성능과 다르다. 각 모델의 입력 문맥 길이도 다르다.',
        '- 사용자 모델은 offline243프레임3D lifting과 전체영상 정규화에 창 밖 문맥을 사용한다. '
        '같은 과거 정보만 허용한 인과적 실시간 비교나 실제 경보 지연시간 평가로 해석하지 않는다.',
        '- 세 데이터셋은 기존 프로젝트 진단에서 이미 관찰되었다. 이번 실행에서 추가 학습·target 보정은 없지만 '
        '완전히 손대지 않은 blind test라고 주장하지 않는다.',
        '- 피험자 수와 낙상 수가 작고 모두 연출 낙상이다. 외부 일반화에 대한 근거 하나이며 '
        '실제 현장·새 피험자 모집단 전체를 대표한다고 해석하지 않는다.',
        '- 본 평가는 낙상 검출이다. 낙상 후 회복 과정, 상태 전이 또는 구조 요청 시점의 성능은 평가하지 않았다.', '',
        '## 검증과 산출물', '',
        '두 비교 모델 각각94영상·2,765창을 직접 추론했다. 가중치 불변, 동일 시험 목록·정답, '
        '시간순서, 다중view 확률 평균과argmax, 사건·영상 confusion 및 AP를 독립 검산했다. '
        '사용자 모델 점수가 이전 검증 결과와 동일함도 확인했다. 네트워크 전체를 별도로 다시 구현한 '
        '독립 재현이나 실제3D 관절 정답 검증을 수행했다는 뜻은 아니다.', '',
        '- [전체 수치 CSV](2026-10-03_shared_unseen_combined_results.csv)',
        '- [94개 영상의 전체 비교 사례](2026-10-03_shared_unseen_combined_cases.csv)',
        '- [직전 사용자 모델의 연속 입력 평가](2026-10-02_temporal_continuous_evaluation_shared.md)',
        '- [참고문헌](2026-10-03_shared_unseen_references.bib)', '',
        '선정 과정에서는 Exa의 다섯 검색 방향에서 요청한28개 검색 결과(중복 제거27개 URL)를 검토하고 공식 논문·저자 코드·배포 자산으로 '
        '실행 가능성과 학습 출처를 확인했다. 시험 데이터가 학습에 포함되는 모델이나 가중치를 확보하지 못한 모델의 '
        '문헌 점수를 이번 직접 비교에 넣지 않았다.', '']
    # Internal ledger remains the detailed source of truth; write it first.
    analysis=dict(summaries=summaries,diagnostics=diagnostics,paired={k:v['paired'] for k,v in data.items()},audits=audits)
    io.save(INTERNAL/'2026-10-03_shared_unseen_analysis.json',analysis)
    internal=INTERNAL/'2026-10-03_shared_unseen_external_internal.md'
    text=internal.read_text().replace('- 상태: `in_progress`','- 상태: `completed`',1)
    text+='\n## 완료 결과\n\n두 공개 비교 가중치 모두94영상2765창 추론과 독립 감사 완료. 사용자 결과는 hash 검증 재사용.\n\n'
    for scope in SCOPES:
        text+=f"- {scope}: "+'; '.join(f"{model} eventF1={100*summaries[model][scope]['event']['f1']:.6f}%" for model in MODEL_NAMES)+'\n'
    text+='\n원본 결과: data/fall_processed/RGB/{shared_unseen_external_20261003_r1,safer_posec3d_external_20261003_r1}/{evaluation.json,independent_audit.json}.\n'
    text+='상세 수치·paired bootstrap·품질실패 회복 분해: docs/internal/2026-10-03_shared_unseen_analysis.json.\n'
    text+='모든 검증된 수치를 shared paired CSV와 최종 연구 문서에 반영했다. 실행 명령·환경·hash·내부경로는 공유용에서 제외했다.\n'
    internal.write_text(text)
    (SHARED/'2026-10-03_shared_unseen_external_shared.md').write_text('\n'.join(report))
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
