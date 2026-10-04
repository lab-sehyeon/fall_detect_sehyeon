"""Read-only aggregation of completed external runs; no model inference or tuning."""
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RGB = ROOT / 'data/fall_processed/RGB'
STEM = '2026-10-04_external_comparison_summary'
DOC_ID = 'DOC-20261004-external-comparison-summary-R4'
SOURCES = [
    'le2i130_20261003_r1', 'cauca100_comparison_20261003_r1',
    'original_baselines_20261004_r2', 'hfd_reproduction_20261004_r1',
    'privacy_x3d_external_20261004_r1', 'usdrl_external_20261004_r1',
    'flash_external_20261004_r1', 'external_comparison_expansion_20261003_r1',
    'three_by_three_20261003_r1/edf', 'three_by_three_20261003_r1/occu',
    'three_by_three_20261003_r1/OOPS', 'shared_unseen_external_20261003_r1',
    'safer_posec3d_external_20261003_r1',
]
LABELS = {
    'own': '사용자 DSTE/J1/G0', 'stgcnpp': 'ST-GCN++ (SAFER)',
    'msg3d': 'MS-G3D (SAFER)', 'cnn1d': '1D-CNN (SAFER)',
    'rthfd': 'RTHFD', 'hfd_reproduction': 'HFD 3D-CNN+SVM 재학습',
    'privacy_x3d_uda_rgb': 'Privacy X3D-UDA (RGB 입력)',
    'usdrl_ntu60': 'USDRL/DSTE + NTU60 분류기', 'flash': 'FLASH 재학습',
    'posec3d_ntu60': 'PoseC3D (NTU60)', 'posec3d_safer': 'PoseC3D (SAFER)',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(c):
    tp, fp, fn = (c[x] for x in ('tp', 'fp', 'fn'))
    out = dict(c, precision=tp / (tp + fp) if tp + fp else 0.,
               recall=tp / (tp + fn) if tp + fn else 0.,
               f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.)
    if 'tn' in c:
        out['accuracy'] = (tp + c['tn']) / sum(c.values())
    return out


def event_counts(predictions, episodes):
    remaining = set(range(len(episodes)))
    hits = 0
    for pred in sorted(predictions, key=lambda p: p['time']):
        candidates = [i for i in remaining if episodes[i]['fall_start'] - .5 <= pred['time']
                      <= max(episodes[i]['fall_start'], episodes[i]['fall_end']) + 3.]
        if candidates:
            remaining.remove(min(candidates, key=lambda i: (episodes[i]['fall_start'], i)))
            hits += 1
    return dict(tp=hits, fp=len(predictions) - hits, fn=len(remaining))


def same(actual, stored):
    for k, v in actual.items():
        if k in stored:
            assert abs(v - stored[k]) < 1e-12, (k, v, stored[k])


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, r)) + ' |' for r in rows])


def main():
    loaded = {s: json.loads((RGB / s / 'evaluation.json').read_text()) for s in SOURCES}
    full_truth = {}
    for source in SOURCES[:2]:
        for row in loaded[source]['rows']:
            if row['model'] == 'own' and row['scope'] in ('le2i130', 'cauca100'):
                full_truth[row['scope'], row['id']] = row['episodes']
    assert len(full_truth) == 230
    verified, rows, unique = [], [], {}
    for source, data in loaded.items():
        evaluation_path = RGB / source / 'evaluation.json'
        audit_path = RGB / source / 'independent_audit.json'
        audit = json.loads(audit_path.read_text())
        assert audit['passed'] is True, source
        digest = sha(evaluation_path)
        if 'evaluation_sha256' in audit:
            assert audit['evaluation_sha256'] == digest, source
        verified.append(dict(source=source, evaluation_sha256=digest,
                             previous_audit_sha256=sha(audit_path),
                             previous_audit_passed=True,
                             previous_audit_has_evaluation_digest='evaluation_sha256' in audit))
        groups = defaultdict(list)
        for r in data['rows']:
            groups[r['model'], r['scope']].append(r)
        for (model, scope), group in groups.items():
            assert len({r['id'] for r in group}) == len(group)
            if scope in ('le2i130', 'cauca100'):
                assert {r['id'] for r in group} == {i for s, i in full_truth if s == scope}
            vc = dict(tp=0, fp=0, fn=0, tn=0)
            ec = dict(tp=0, fp=0, fn=0)
            has_event = 'event' in group[0]
            positives = events = 0
            for r in group:
                gt = full_truth.get((scope, r['id']), r.get('episodes'))
                assert gt is not None
                if 'episodes' in r and (scope, r['id']) in full_truth:
                    assert [(g['fall_start'], g['fall_end']) for g in gt] == [
                        (g['fall_start'], g['fall_end']) for g in r['episodes']]
                if 'ground_truth' in r:
                    assert r['ground_truth'] == int(bool(gt))
                truth, pred = bool(gt), bool(r['video_prediction'])
                vc['tp' if truth and pred else 'fp' if pred else 'fn' if truth else 'tn'] += 1
                positives += int(truth)
                events += len(gt)
                if not r['processed']:
                    assert not pred
                if has_event:
                    assert pred == bool(r['predictions'])
                    c = event_counts(r['predictions'], gt)
                    same(c, r['event'])
                    for k in ec:
                        ec[k] += c[k]
            vm, em = metrics(vc), metrics(ec) if has_event else None
            published = data['summaries'][model][scope] if has_event else data['summaries'][scope]
            same(vm, published['video'] if has_event else published)
            if has_event:
                same(em, published['event'])
            assert published['videos'] == len(group)
            assert published['processed'] == sum(bool(r['processed']) for r in group)
            label_key = model
            if model == 'posec3d':
                label_key += '_safer' if source.startswith('safer_') else '_ntu60'
            result = dict(source=source, model=label_key, label=LABELS[label_key], scope=scope,
                          videos=len(group), processed=published['processed'],
                          positive_videos=positives, fall_events=events, video=vm, event=em)
            key = label_key, scope
            if key in unique:
                old = unique[key]
                for k in ('videos', 'processed', 'positive_videos', 'fall_events', 'video', 'event'):
                    assert old[k] == result[k], (source, key, k)
            else:
                unique[key] = result
            rows.append(result)
    def pct(model, scope, unit, metric='f1'):
        value = unique[model, scope][unit]
        return f'{100 * value[metric]:.2f}%' if value is not None else '—'
    all_models = ['own', 'stgcnpp', 'msg3d', 'cnn1d', 'usdrl_ntu60',
                   'hfd_reproduction', 'privacy_x3d_uda_rgb', 'rthfd', 'flash']
    external_models = ['hfd_reproduction', 'privacy_x3d_uda_rgb', 'flash']
    main_models = ['own', 'usdrl_ntu60', *external_models]
    roles = {'own':'사용자 모델', 'usdrl_ntu60':'원본 기반 기준모델',
             **{m:'선정 외부 비교 모델' for m in external_models}}
    historical_models = ['own', 'stgcnpp', 'msg3d', 'cnn1d']
    main_table = table(['모델', 'Le2i Precision', 'Le2i Recall', 'Le2i F1',
                        'CAUCA Precision', 'CAUCA Recall', 'CAUCA F1'],
                       [[LABELS[m]] + [pct(m, s, 'video', k) for s in ('le2i130', 'cauca100')
                                      for k in ('precision', 'recall', 'f1')]
                        for m in main_models])
    coverage = table(['모델', 'Le2i 처리/전체', 'CAUCA 처리/전체'],
                     [[LABELS[m]] + [f"{unique[m,s]['processed']}/{unique[m,s]['videos']}"
                                     for s in ('le2i130', 'cauca100')] for m in main_models])
    other = table(['데이터셋', '영상/낙상 사건', '사용자', 'ST-GCN++', 'MS-G3D', '1D-CNN'],
                  [[label, f"{unique['own',s]['videos']}/{unique['own',s]['fall_events']}"] +
                   [pct(m, s, 'event') for m in historical_models]
                   for s, label in [('gmdcsa37','GMDCSA'), ('edf','EDF-CS'),
                                    ('occu','OCCU-CS'), ('OOPS','OOPS-Fall 시험 분할')]])
    partial = table(['모델', 'Le2i 38개 사건 F1', 'GMDCSA 37개 사건 F1', 'CAUCA 19개 사건 F1'],
                    [[LABELS[m]] + [pct(m,s,'event') for s in ('le2i38','gmdcsa37','cauca19')]
                     for m in ['own','posec3d_safer','posec3d_ntu60','stgcnpp','msg3d','cnn1d']])
    older = table(['모델', 'Le2i 127개 사건 F1'],
                  [[LABELS[m], pct(m,'le2i127','event')] for m in historical_models])
    counts = table(['데이터', '모델', '사건 TP/FP/FN', '영상 TP/FP/FN/TN'],
                   [[s, LABELS[m], '/'.join(str(unique[m,s]['event'][k]) for k in ('tp','fp','fn'))
                     if unique[m,s]['event'] else '미측정',
                     '/'.join(str(unique[m,s]['video'][k]) for k in ('tp','fp','fn','tn'))]
                    for s in ('le2i130','cauca100') for m in main_models])
    supplementary = table(['모델', 'Le2i 영상 F1', 'Le2i 사건 F1', 'CAUCA 영상 F1', 'CAUCA 사건 F1'],
                          [[LABELS[m]] + [pct(m,s,u) for s in ('le2i130','cauca100')
                                          for u in ('video','event')]
                           for m in all_models if m not in main_models])
    body = f'''# 사용자 모델·원본 기준모델·선정 세 모델의 외부 낙상 비교

문서 ID: {DOC_ID}  
기준일: 2026-10-04  
상태: `completed` — 기존 실행 결과 집계·검산 완료

## 범위와 평가 기준

종합표는 사용자 DSTE→J1→G0, 원본 기반 DSTE+NTU60 분류기, 선정한 HFD 3D-CNN+SVM·Privacy X3D-UDA RGB·FLASH의 다섯 행으로 구성한다.
선정한 외부 비교 모델은 세 개로 유지하며, 원본 기반 기준모델은 J1/G0 적용 전 구성과의 비교용으로 구분한다.
메인 지표는 낙상을 양성 클래스로 한 영상 단위 Precision·Recall·F1 세 가지다. 모든 값은 백분율로 표시한다.
Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1 = 2TP/(2TP+FP+FN)으로 계산하며 macro/weighted 평균이 아니다.
다섯 구성에 공통으로 측정된 영상 단위를 사용하고, 사건 단위 지표를 같은 표에 섞지 않는다.
이전 다른 모델의 실험은 부록에 보존하며 메인 비교 모델에 포함하지 않는다.
이 표에 논문 보고값을 섞지 않았다. 새 학습이나 추론은 수행하지 않고 기존 예측과 정답으로 집계를 재검산했다.
NTU60의 ADL 다중 클래스 성능이나 회복 과정 인식 성능을 평가한 표는 아니다.

영상 F1은 영상 단위 낙상/비낙상 판정을 평가한다. 사건 F1은 정답 시작 0.5초 전부터 종료 3초 후까지
경보를 일대일 매칭하고, 미탐과 추가·반복 경보를 반영한다. 처리 실패는 무경보로 두고 전체 분모에 포함한다.
X3D는 저자 방식대로 5개 clip 점수를 평균한 영상 판정이며, 시각 출력이 없어 사건 F1은 미측정이다.
다른 모델들의 영상 판정은 경보가 한 번이라도 있는지를 사용한다. 학습 데이터·입력 변환·집계 방식이 다른 전체 시스템 비교다.

## 전체 Le2i 130개·CAUCAFall 100개

Le2i는 낙상 99개·비낙상 31개, CAUCAFall은 낙상 50개·비낙상 50개다.
각 데이터셋 안에서 모든 모델에 동일한 영상 목록과 정답을 적용했다.
전체 CAUCA 100개는 배포 시험 부분집합 19개보다 넓은 외부 평가 범위다.

{main_table}

{coverage}

처리 성공은 모든 프레임의 관절 추정이 정확하다는 뜻이 아니다. 사용자 모델의 품질 거부도 성능에 포함된다.
FLASH는 CAUCA의 낙상 영상 1개에서 관절을 추출하지 못했으며 해당 영상도 FN으로 포함했다.

## 모델·학습 출처

| 모델 | 실제 사용한 자산과 학습 조건 | 평가 성격 |
| --- | --- | --- |
| 사용자 DSTE/J1/G0 | NTU60 인코더, SAFER+FU에서 학습한 J1과 SAFER 분류 경로 | 기존 고정 모델 |
| 원본 기반 DSTE+NTU60 분류기 | 저자 공개 DSTE 인코더 + 프로젝트에서 NTU60으로 학습한 60클래스 선형 분류기; A43를 낙상으로 판정 | J1/G0 미사용; 저자 배포 완성형 낙상 분류기 아님 |
| HFD 3D-CNN+SVM | 저자 C3D 가중치·특징 추출, GMDCSA 32개로 SVM 새 학습 | 공식 코드 기반 재현; 원저자 최종 SVM 가중치 아님 |
| Privacy X3D-UDA | 저자 RGB+depth 공동 학습 가중치, RGB 입력; 공개 설정은 Kinetics-700, backbone은 Kinetics400 | 저자 원래 추론·평가 실행; 순수 RGB source-only 가중치 아님 |
| FLASH | 공식 HyperMamba 모델을 공개 UP-Fall 82시퀀스로 학습 | 변경점을 명시한 재현; 원저자 배포 가중치 아님 |

이 실행에서는 타깃 학습·무라벨 적응·임계값 조정을 하지 않았다. 다만 Le2i·CAUCA는 이전 진단에서
이미 관찰했으므로 최초 blind test라고 부르지 않는다. 전체 pose/기반 모델 사전학습 자료의 중복까지 확인한 것은 아니다.
X3D 선행 가중치의 전체 이력은 독립 인증하지 못했다. HFD를 학습한 GMDCSA를 해당 모델의 외부 시험으로 사용하지 않는다.

## 종합 결과 해석

- 사용자 모델은 이 다섯 구성 중 두 데이터셋 모두 영상 F1이 가장 높다. Precision·Recall 모두 최고라는 의미는 아니다.
  Le2i Precision은 원본 기반 기준모델이 98.63%로 가장 높고, Recall은 FLASH가 Le2i 100.00%·CAUCA 98.00%로 가장 높다.
  선정 범위의 관측 결과이며 모든 기존 모델보다 우수하다는 주장이 아니다.
- 원본 기반 기준모델 대비 사용자 모델의 영상 F1 차이는 Le2i +7.28%p, CAUCA +31.11%p다.
  CAUCA Recall은 40.00%에서 78.00%로 높아졌다. 낙상 학습 데이터와 분류 경로도 달라 adapter 구조만의 효과를 분리한 실험은 아니다.
- FLASH의 Le2i 영상 F1 86.46%는 비낙상 구분이 좋다는 뜻이 아니다. Le2i 31개와 CAUCA 50개의 비낙상 영상 모두에서
  경보를 냈고 반복 경보도 많아 사건 F1은 낮다. 공개 source는 낙상 영상 내 impact/non-impact 과제이며 독립 ADL 분류와 다르다.
- HFD는 source clip 단위 검증에서 같은 원영상의 조각이 양쪽 분할에 들어갈 수 있다. FLASH의 source 분할도
  피험자 독립 평가가 아니며, loader 수정·전처리 차이를 명시한 재현이다. 낮은 외부 수치를 원논문의 성능으로 일반화하지 않는다.
- 학습 데이터·입력 품질 기준·표현·시간 집계가 함께 다르므로 adapter 구조 하나의 효과를 입증한 비교가 아니다.
  오프라인 미래 문맥을 쓰는 경로가 있어 이 표로 실시간 지연 성능을 주장하지 않는다.

## 혼동행렬

{counts}

## 부록: 이전 비교 모델의 결과

다음 결과는 완료된 과거 실험 기록이며 현재 선정한 세 비교 모델에 포함하지 않는다.

{supplementary}

ST-GCN++·MS-G3D·1D-CNN은 SAFER 공개 가중치를 공통 외부 평가 절차에서 실행했다.
RTHFD는 MoveNet과 저자의 고정 규칙을 사용했다.
전체 과거 비교에서는 ST-GCN++·MS-G3D가 사용자보다 높은 사건 F1을 보였으므로 메인 세 모델의 결과를 모든 모델에 일반화하지 않는다.

### 이전 추가 데이터셋의 사건 F1

{other}

EDF/OCCU는 각각 한 피험자·두 시점의 긴 영상 2개이고 모두 낙상을 포함한다. 두 자료는 연구진·피험자 풀을 공유한다.
영상 단위 비낙상 특이도를 평가하기에는 부족하다. OOPS는 공식 시험 목록에서 경로 중복을 제거한 572개 영상과
636개 낙상 사건이며, 원래 중복 구간 단위 평가의 분모와 다르다.

### 이전 부분집합 비교

{partial}

PoseC3D는 SAFER/NTU60 공개 가중치를 사용했으며 전체 130개/100개 확장 결과는 없다.
과거 사용자 78.95%·87.50%는 각각 38개·19개 결과다. 부분집합과 전체 평가를 합산하지 않는다.
기존 Le2i 127개 집계는 다음과 같으며 현재 메인 비교에서는 정답을 확보한 130개 결과를 사용한다.

{older}

### 별도 보존한 실험

URFD 70개 사용자 단독 평가, EDF confidence 사후 완화 진단, OmniFall 구간 단위 평가와 문헌 보고값 비교는
동일 모델·동일 영상의 위 비교표와 성격이 달라 원래 보고서에 분리 보존했다. confidence 완화 수치로 기본 EDF 결과를 대체하지 않았다.
SDFA·DistillH-Mamba·STGCN-GRU-BiLSTM의 외부 성능은 아직 확보하지 않았으며, MCFD 평가는 보류 상태다.

[다섯 구성 종합 CSV]({STEM}_selected_results.csv) · [과거 실험 포함 전체 CSV]({STEM}_results.csv) ·
[원본 기반 기준모델 상세](2026-10-04_usdrl_external_evaluation_shared.md) ·
[FLASH 상세](2026-10-04_flash_reproduction_shared.md) ·
[HFD·X3D 상세](2026-10-04_hfd_x3d_external_comparison_shared.md) ·
[추가 세 데이터셋](2026-10-03_three_models_three_new_datasets_shared.md) ·
[PoseC3D 부분집합](2026-10-03_shared_unseen_external_shared.md)
'''
    evidence_dir = ROOT / 'docs/internal' / (STEM + '_evidence')
    evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence = dict(passed=True, checked_at_utc=datetime.now(timezone.utc).isoformat(),
                    source_files=verified, source_summary_groups=len(rows),
                    unique_model_scope_groups=len(unique),
                    full_comparison_models=all_models, selected_comparison_models=main_models,
                    external_comparison_models=external_models,
                    base_reference_model='usdrl_ntu60',
                    primary_metric_unit='video', positive_class='fall',
                    primary_metrics=['precision','recall','f1'],
                    common_target_videos=230,
                    all_video_counts_recomputed=True, all_event_matches_recomputed=True,
                    all_duplicates_agree=True, gpu_or_training_executed=False)
    (evidence_dir / 'consolidation_audit.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    provenance = table(['실행', '평가 파일 SHA256', '기존 감사의 평가 digest'],
                       [[v['source'],v['evaluation_sha256'],
                         '일치' if v['previous_audit_has_evaluation_digest'] else '필드 없음; 집계·기존 감사 수치 대조']
                        for v in verified])
    internal = f'''# 외부 비교 통합 내부 원장

문서 ID: {DOC_ID}
상태: `completed`

요청: 지금까지 실제 수행한 사용자/다른 모델 비교를 표로 통합한다.
R2 정정: 사용자는 이미 HFD·Privacy X3D·FLASH 세 비교 모델을 정했다. R1에서 과거9모델을 메인으로 확대한 것은 범위 오류다.
메인은 사용자 포함4행으로 수정하고 나머지는 부록에 보존한다. 결과를 보고 모델을 새로 고른 변경이 아니며 기존 합의를 반영했다.
R1 내부·공유 문서와 감사 JSON은 동반 evidence/revision_r1에 보존했다. 실험·지표는 바꾸지 않는다.
R2 생성 스크립트의 목록 분리 중 닫는 괄호 1개가 남아 최초 실행이 SyntaxError로 중단되었다.
괄호를 수정한 뒤 생성·검산을 다시 완료했다. 원본 평가 파일이나 모델 실행에는 영향이 없다.
R3: 사용자가 평가 요소를 Precision·Recall·F1 세 가지로 지정했다. 메인 표와 선정 CSV를 영상 단위 PR/F1으로 통일했다.
선정 CSV의 *_pct 열은 0~100 범위이며 원본 전체 CSV의 0~1 비율과 구분한다. 양성 클래스는 fall이며 micro/macro 평균이 아니다.
X3D에 사건 시각 출력이 없으므로 네 모델 공통 비교는 영상 단위로 유지한다. 기존 영상 정답·예측·threshold를 변경하지 않았다.
R2 내부·공유 문서, 선정 CSV, 감사 JSON을 evidence/revision_r2에 보존했고, 이전 사건 지표는 전체 CSV와 원 보고서에 유지한다.
R4: 원본 DSTE 기준모델 평가를 확인한 후 사용자가 종합을 요청했다. 사용자+원본 기반 기준모델+기존 외부3종의 5구성으로 통합했다.
선정 외부 비교 모델3종은 유지한다. 원본 기반 기준모델은 official DSTE와 프로젝트 NTU60 분류기의 결합이며 저자 최종 낙상 checkpoint로 표현하지 않는다.
CSV는 역할 열로 원본 기준/외부 비교/사용자를 구분한10행이며 PR/F1 및 평가 단위는 유지한다. R3 문서·CSV·감사는 evidence/revision_r3에 보존했다.
원본 실험·가중치·예측은 변경하지 않고 기존 완료 결과 13개와 독립 감사 기록을 읽었다.
새 학습·추론·다운로드·시스템 변경은 하지 않았다. Python 표준 라이브러리만 사용했다.
집계 실행: `python scripts/summarize_external_comparisons_20261004.py`.
시작/검산 시각은 동반 `consolidation_audit.json` UTC 필드에 기록한다.
원본 근거 기준 디렉터리: `data/fall_processed/RGB/`; 아래 실행마다 `evaluation.json`과 `independent_audit.json`.

모든 영상별 판정에서 혼동행렬을 다시 계산하고, 모든 경보를 정답 시작−0.5초~종료+3초로 재매칭했다.
모든 source summary와 대조했고, 중복 모델·scope 수치도 일치한다. 전체 평가 9모델 모두 같은230개 ID 및 정답을 확인했다.
기존 감사 13개는 passed다. HFD 감사에는 evaluation_sha256 필드가 원래 없으며 hash 불일치가 아니다.
나머지12개는 digest가 일치한다. HFD도 이번 집계에서 영상 판정·정답·경보로 지표를 다시 계산했다.
이 기록은 과거 감사의 존재와 현재 집계를 확인하며, 이번에 신경망 전수 재추론을 다시 했다는 뜻이 아니다.
FLASH 과거 source logit 허용오차 실패는 보존되어 있고 source 판정·집계는 일치한다. target 재추론은 원 감사에서 일치했다.
RTHFD는 observer 수정 후 R2만 채택하며 R1은 통합하지 않는다. PoseC3D 두 학습본은 별도 ID로 분리한다.

{provenance}

내부 먼저 기록하고 공유용을 이어 생성한다. 공유에는 결과·평가 조건·출처 성격·제약만 반영하고 경로·digest·명령은 제외한다.
생성물: 본 내부 문서, 공유 문서, 공유 CSV, 동반 집계 검산 JSON, 집계 스크립트. 문서 인덱스 세 곳에 새 링크 추가.
권한·라이선스·접근 조건 변경 없음; 새 영상·개인정보·비밀 자료를 문서에 복사하지 않음.
남은 집계 작업 없음. 원 모델 간 학습·입력 조건 차이와 X3D 전체 학습 이력의 불확실성은 공유 본문에 유지한다.

---

{body}
'''
    (ROOT / 'docs/internal' / (STEM + '_internal.md')).write_text(internal)
    csv_path = ROOT / 'docs/shared' / (STEM + '_results.csv')
    flat = []
    for r in unique.values():
        item = {k:r[k] for k in ('model','label','scope','videos','processed','positive_videos','fall_events')}
        for unit in ('video','event'):
            for k in ('tp','fp','fn','tn','precision','recall','f1','accuracy'):
                item[unit+'_'+k] = r[unit].get(k,'') if r[unit] else ''
        flat.append(item)
    with csv_path.open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(flat[0]))
        writer.writeheader();writer.writerows(flat)
    selected = [r for r in flat if r['model'] in main_models and r['scope'] in ('le2i130','cauca100')]
    assert len(selected) == 10
    selected_prf = [dict(model=r['model'],label=r['label'],scope=r['scope'],videos=r['videos'],
                         role=roles[r['model']],
                         metric_unit='video',positive_class='fall',
                         **{k+'_pct':100*r['video_'+k] for k in ('precision','recall','f1')})
                    for r in selected]
    with (ROOT / 'docs/shared' / (STEM + '_selected_results.csv')).open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(selected_prf[0]))
        writer.writeheader();writer.writerows(selected_prf)
    (ROOT / 'docs/shared' / (STEM + '_shared.md')).write_text(body)
    print(json.dumps(evidence,ensure_ascii=False,indent=2))
    print(main_table)


if __name__ == '__main__':
    main()
