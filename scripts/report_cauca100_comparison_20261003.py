"""Publish audited CAUCA100 four-model results, without inference or tuning."""
from pathlib import Path
import csv
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fall_pipeline.external import rgb_document_io as io

OUT = ROOT / 'data/fall_processed/RGB/cauca100_comparison_20261003_r1'
TOPIC = '2026-10-03_cauca100_comparison'
NAMES = {'own': '사용자 DSTE/J1/G0', 'stgcnpp': 'ST-GCN++', 'msg3d': 'MS-G3D', 'cnn1d': '1D-CNN'}


def main():
    audit = io.read(OUT / 'independent_audit.json')
    evaluation = io.read(OUT / 'evaluation.json')
    completed = io.read(OUT / 'comparators_completed.json')
    assert audit['passed'] and completed['passed']
    assert audit['evaluation_sha256'] == io.sha(OUT / 'evaluation.json')
    assert audit['contract_sha256'] == io.sha(OUT / 'contract.json')
    assert audit['prior19_results_identical'] and audit['source_predictions_identical']
    scores = evaluation['summaries']
    rows = []
    for scope in ['cauca100', 'cauca19']:
        for model in NAMES:
            s = scores[model][scope]
            rows.append(dict(dataset=scope, model=model, videos=s['videos'], fall_events=s['fall_events'],
                             processed=s['processed'], **s['event'],
                             **{'video_' + k: v for k, v in s['video'].items()}))
    with (OUT / 'cases.csv').open() as f:
        cases = list(csv.DictReader(f))
    full_cases = [r for r in cases if r['dataset'] == 'cauca100']
    assert len(rows) == 8 and len(cases) == 119 and len(full_cases) == 100

    def event_table(scope):
        result = ['| 모델 | TP | FP | FN | 정밀도 | 재현율 | 사건 F1 | 분류 처리 / 전체 |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for m, name in NAMES.items():
            s = scores[m][scope]; e = s['event']
            result.append(f'| {name} | {e["tp"]} | {e["fp"]} | {e["fn"]} | {e["precision"]*100:.2f}% | '
                          f'{e["recall"]*100:.2f}% | {e["f1"]*100:.2f}% | {s["processed"]} / {s["videos"]} |')
        return '\n'.join(result)

    video = ['| 모델 | 영상 TP / FP / FN / TN | 정확도 | 영상 정밀도 | 영상 재현율 | 영상 F1 | 특이도 | AP |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for m, name in NAMES.items():
        s = scores[m]['cauca100']['video']
        confusion = ' / '.join(str(s[k]) for k in ['tp', 'fp', 'fn', 'tn'])
        metrics = ' | '.join(f'{s[k]*100:.2f}%' for k in ['accuracy', 'precision', 'recall', 'f1', 'specificity', 'ap'])
        video.append(f'| {name} | {confusion} | {metrics} |')
    video = '\n'.join(video)

    breakdown = []
    errors = ['| 모델 | 처리 실패에 따른 미탐 | 분류 후 무경보 미탐 | 시간 매칭 실패 | 비낙상 영상 경보 | 낙상 영상의 미매칭 추가 경보 |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for model, name in NAMES.items():
        selected = [r for r in evaluation['rows'] if r['scope'] == 'cauca100' and r['model'] == model]
        d = dict(model=model, rejected_fn=0, noalarm_fn=0, timing_fn=0, negative_fp=0, positive_fp=0)
        for r in selected:
            if r['episodes']:
                kind = 'rejected_fn' if not r['processed'] else 'noalarm_fn' if not r['predictions'] else 'timing_fn'
                d[kind] += r['event']['fn']; d['positive_fp'] += r['event']['fp']
            else:
                d['negative_fp'] += r['event']['fp']
        e = scores[model]['cauca100']['event']
        assert d['rejected_fn'] + d['noalarm_fn'] + d['timing_fn'] == e['fn']
        assert d['negative_fp'] + d['positive_fp'] == e['fp']
        breakdown.append(d)
        errors.append('| ' + name + ' | ' + ' | '.join(str(d[k]) for k in list(d)[1:]) + ' |')
    errors = '\n'.join(errors)
    best = max(NAMES, key=lambda m: scores[m]['cauca100']['event']['f1'])
    differences = [dict(model=m, own_minus_comparator_pp=100*(scores['own']['cauca100']['event']['f1']-
                    scores[m]['cauca100']['event']['f1'])) for m in NAMES if m != 'own']

    internal_path = ROOT / 'docs/internal' / f'{TOPIC}_internal.md'
    internal = internal_path.read_text().replace('상태: `in_progress`', '상태: `completed`', 1)
    internal += '\n## 실행 완료와 검산\n\n'
    internal += (f'공개 비교 모델마다100개·{audit["windows"]["stgcnpp"]:,}창을 새로 추론했다. '
                 f'세 모델 합계{sum(audit["windows"][m] for m in NAMES if m != "own"):,}창, '
                 f'비교 추론 단계 {completed["elapsed_seconds"]:.2f}초. 사용자 예측은 기존100개 그대로다.\n'
                 '기존19개 경보·판정·집계가 모두 동일하며 원시 logits 일치 여부도 감사에 기록했다.\n\n')
    internal += event_table('cauca100') + '\n\n' + video + '\n\n' + errors + '\n\n'
    internal += '기존19개 재현 확인:\n\n' + event_table('cauca19') + '\n\n'
    internal += ('실행 전 GPU0 RTX3090 free24,124MiB 및 디스크204GiB를 확인했다. '
                 '준비 단계는 CPU에서 기존 두 부모 계약·모델·입력 해시를 확인한 뒤 새 계약을 고정했다.\n'
                 '실행 명령과 단계 시각:\n\n```json\n' + json.dumps(io.read(OUT/'runner_journal.json'), ensure_ascii=False, indent=2) + '\n```\n\n'
                 '준비 명령: `CUDA_VISIBLE_DEVICES=\'\' python '
                 'scripts/run_cauca100_comparison_20261003.py prepare`. '
                 '모델 단계는 GPU0, 채점·감사는 CPU로 실행했다.\n\n'
                 '독립 검산:\n\n```json\n' + json.dumps(audit, ensure_ascii=False, indent=2) + '\n```\n\n'
                 '가중치 불변, 기존19개 입력·raw output·판정, 사용자100개 예측, PTS 순서, 품질 gate, '
                 'raw logits→확률/argmax/상승경보 및 사건·영상 confusion/AP를 별도 코드로 대조했다. '
                 '모델 추론과 감사 중 threshold 변경·재학습·실패 제외는 없었다.\n\n'
                 '평가·감사·모델별 NPZ·사례는 새 출력 폴더에 보존한다. '
                 '원본 evaluation.json의 pending_independent_audit 필드는 채점 직후 기록으로 보존되며 '
                 '동일 evaluation SHA에 연결된 independent_audit.json과 status.json의 completed가 최종 상태다.\n\n'
                 '공유 반영: 수행 범위, 검증 지표, 실패 구분, 기존19개 비교, 연구 한계. '
                 '공유 제외: 로컬 경로·명령·장비·해시·내부 로그. 기존 결과는 유지하고 후속 링크만 추가한다.\n')
    internal_path.write_text(internal)

    shared = f'''# CAUCA 전체100개에서의 낙상 탐지 비교

문서 ID: DOC-20261003-cauca100-comparison-R1  
갱신일: 2026-10-03  
상태: `completed` — 세 공개 비교 모델 추론 및 독립 검산 완료

## 평가 범위

**동일한 CAUCAFall 전체100개(낙상50/비낙상50)에 사용자 DSTE/J1/G0와 ST-GCN++, MS-G3D, 1D-CNN을 비교했다.**
검증된 관절 입력과 사용자 예측을 재사용하고, 세 비교 모델은 각각100개 영상·2,367개 시간창을 새로 추론했다.
문헌 수치를 옮긴 표가 아니라 공개된 SAFER 학습 가중치를 실제 실행한 결과다.

원영상은 [CAUCAFall 공식 v4 배포본](https://data.mendeley.com/datasets/7w7fccy7ky/4),
구간 정답은 기존 평가와 같은 고정 OmniFall 주석이다. 전체100개에는 OmniFall train69/val9/test19와
분할 목록 밖3개가 포함된다. 이 분할명은 배포 메타데이터이며 이번 실행에서 해당 영상으로 학습했다는 뜻이 아니다.
전체100개 결과와 공식19개 시험 부분집합 결과를 구분하고 두 분모를 합산하지 않았다.

## 전체100개 사건 탐지

{event_table('cauca100')}

TP는 정답 사건에 대응한 탐지, FN은 놓친 낙상 사건, FP는 정답과 일대일 대응하지 않는 경보다.
같은 낙상에 여러 번 경보하면 추가 경보는 FP에 포함될 수 있다.
이번 조건에서 사건 F1이 가장 높은 모델은 **{NAMES[best]}({scores[best]['cauca100']['event']['f1']*100:.2f}%)**다.

## 전체100개 영상 단위 분류

영상 안에서 한 번이라도 낙상으로 판정하면 양성으로 집계한다. 사건 시각의 정확성이나 반복 경보는
이 영상 단위 지표에서 구분되지 않으므로 사건 F1과 별도로 제시한다.

{video}

## 실패와 오탐의 구성

{errors}

사용자 모델은 전체 영상의 사람·관절 검출률 및 골반 confidence를 확인하는 품질 기준을 유지했다.
분류한93개와 품질 거부7개를 모두100개 분모에 포함했으며, 거부된 낙상5개는 FN으로 계산했다.
세 비교 모델은 기존2D 입력 조건을 유지하여100개 모두 분류했다.
따라서 이 표는 모델 고유 입력 처리까지 포함한 파이프라인 성능 비교다.
입력 차원·학습 데이터·품질 거부 차이가 함께 있으므로 차이 전체를 adapter나 분류기 구조 하나의 효과로 해석할 수 없다.

## 기존19개 부분집합의 재확인

{event_table('cauca19')}

기존19개의 영상별 경보·판정·지표는 앞선 평가와 동일했다. 전체100개 결과는 이19개를 포함한 확장 평가다.

## 유지한 실행·평가 조건과 주장 범위

- 전체 영상의 시간 순서를 유지한25Hz 입력,64프레임 결정창과8프레임 간격을 사용했다.
  비교 모델의 고유48프레임 입력은 같은 결정창의 마지막48프레임으로 구성했다.
- 사용자 모델은 고정 DSTE→J1→G0, 비교기는 SAFER 공식 ST-GCN++·MS-G3D·1D-CNN 가중치와
  해당 모델의 정규화·입력 변환을 사용했다. 모델별 클래스에서 낙상 argmax를 판정했다.
- 낙상 판정으로 전환되는 상승 경보와 정답 시작−0.5초~종료+3초 범위를 일대일 매칭했다.
  타깃 추가 학습·임계값 조정 없이 기존 평가 조건을 유지했다.
- 사용자 행동 학습 출처는 NTU60·SAFER·FU-Kinect-Fall, 비교기는 공식 SAFER 학습 배포본이다.
  CAUCA 행동 학습 미사용 근거는 공개 설정·기록 범위이며, 기반 pose 사전학습 이미지의 모든 중복을 검증한 것은 아니다.
- CAUCA는 이전 개발 진단과 평가에서 이미 관찰했다. 이번 확장을 완전히 처음 보는 blind test로 부르지 않는다.
- 공개 모델을 **공통 외부 평가 절차**에서 실행한 결과다. 저자의 전체 RGB 전처리·후처리와 원래 평가프로그램을
  그대로 재현한 논문 보고 점수와 구분한다. 실시간 성능·ADL 다중 분류·회복 과정 성능은 이번 평가 대상이 아니다.

[전체 지표 CSV]({TOPIC}_results.csv) · [100개 영상별 비교 CSV]({TOPIC}_cases.csv)
'''
    shared_path = ROOT / 'docs/shared' / f'{TOPIC}_shared.md'
    shared_path.write_text(shared)
    csv_paths = []
    for suffix, values in [('results', rows), ('cases', full_cases)]:
        path = ROOT / 'docs/shared' / f'{TOPIC}_{suffix}.csv'
        with path.open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
        csv_paths.append(path)
    assert all(x not in shared for x in ['/home/', 'CUDA_VISIBLE_DEVICES', 'checkpoint/', 'sha256', 'docs/internal'])
    assert f'{scores["own"]["cauca100"]["event"]["f1"]*100:.2f}%' in shared
    io.save(OUT/'report_validation.json', dict(passed=True, evaluation_sha256=audit['evaluation_sha256'],
            audit_sha256=io.sha(OUT/'independent_audit.json'), table_rows=len(rows), case_rows=len(full_cases),
            docs_sha256={str(p.relative_to(ROOT)): io.sha(p) for p in [internal_path, shared_path, *csv_paths]},
            breakdown=breakdown, differences=differences, best_model=best))
    print(event_table('cauca100'))
    print(video)
    print(errors)
    print('REPORT PASS')


if __name__ == '__main__':
    main()
