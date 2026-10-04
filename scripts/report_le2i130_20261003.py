"""Publish only independently audited Le2i130 results to paired research records."""
from pathlib import Path
import csv
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/fall_processed/RGB/le2i130_20261003_r1'
STEM = '2026-10-03_le2i130_reevaluation'
NAMES = {'own': '사용자 DSTE/J1/G0', 'stgcnpp': 'ST-GCN++', 'msg3d': 'MS-G3D', 'cnn1d': '1D-CNN'}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(headers, rows):
    lines = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    return '\n'.join(lines + ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def pct(value):
    return f'{100 * value:.2f}%'


def main():
    audit = read(OUT / 'independent_audit.json')
    assert audit['passed'] and audit['evaluation_sha256'] == sha(OUT / 'evaluation.json')
    assert audit['contract_sha256'] == sha(OUT / 'contract.json')
    journal = read(OUT / 'runner_journal.json')
    assert [r['stage'] for r in journal] == ['prepare', 'frontend', 'lift', 'infer', 'comparisons', 'score', 'audit']
    assert all(r['returncode'] == 0 for r in journal)
    evaluation = read(OUT / 'evaluation.json')
    delta = read(OUT / 'comparison_127_to_130.json')
    plan = read(OUT / 'plan.json')
    scores = {m: evaluation['summaries'][m]['le2i130'] for m in NAMES}
    config = read(ROOT / 'configs/le2i130_20261003_r1.json')
    parent = read(ROOT / 'configs/external_comparison_expansion_20261003_r1.json')
    same = {k: config[k] == parent[k] for k in ['detector', 'pose_config', 'quality', 'lifting', 'inference', 'model', 'comparator_protocol', 'source_training']}
    assert all(same.values())
    rows = evaluation['rows']
    failures = [r for r in rows if r['model'] == 'own' and not r['processed']]
    missed = [r for r in rows if r['model'] == 'own' and r['processed'] and r['event']['fn']]
    assert len(failures) == 11 and sum(r['event']['fn'] for r in failures) == 11
    assert len(missed) == 2 and all(not r['predictions'] for r in missed)
    added_ids = {r['id'] for r in plan if r['added_video']}
    added_rows = {(r['id'], r['model']): r for r in rows if r['id'] in added_ids}
    assert delta['own']['added3']['tp'] == 3 and delta['own']['added3']['fp'] == 0
    event_table = table(['모델', '탐지 TP', '오탐 FP', '미탐 FN', '정밀도', '재현율', '낙상 사건 F1'],
        [[NAMES[m], *[scores[m]['event'][k] for k in ['tp', 'fp', 'fn']], *[pct(scores[m]['event'][k]) for k in ['precision', 'recall', 'f1']]] for m in NAMES])
    delta_table = table(['모델', '기존 127개 F1', '전체 130개 F1', '변화', '추가 3개 TP/FP/FN'],
        [[NAMES[m], pct(delta[m]['previous127']['f1']), pct(delta[m]['full130']['f1']), f"{delta[m]['f1_delta_pp']:+.2f}%p", '/'.join(str(delta[m]['added3'][k]) for k in ['tp', 'fp', 'fn'])] for m in NAMES])
    secondary_table = table(['모델', '영상 단위 정확도', '영상 단위 F1', '특이도'],
        [[NAMES[m], *[pct(scores[m]['video'][k]) for k in ['accuracy', 'f1', 'specificity']]] for m in NAMES])
    added_table = table(['추가 영상', '원본 낙상 구간(초)', '사용자 경보 시각(초)', '사용자 TP/FP/FN', 'ST-GCN++ TP/FP/FN', 'MS-G3D TP/FP/FN', '1D-CNN TP/FP/FN'],
        [[added_rows[(sid, 'own')]['path'], f"{added_rows[(sid, 'own')]['episodes'][0]['fall_start']:.2f}–{added_rows[(sid, 'own')]['episodes'][0]['fall_end']:.2f}",
          '; '.join(f"{p['time']:.2f}" for p in added_rows[(sid, 'own')]['predictions']),
          *['/'.join(str(added_rows[(sid, m)]['event'][k]) for k in ['tp', 'fp', 'fn']) for m in NAMES]] for sid in sorted(added_ids)])
    internal = ROOT / 'docs/internal' / (STEM + '_internal.md')
    completion = f'''

## 재평가 완료 및 독립 검산

상태: `completed`. 전체 130개 영상, 낙상 99건, 비낙상 31개를 네 모델로 집계했다.
기존 128개 입력·예측을 검증하여 복사하고 2개(2,169프레임, 모델별 256창)를 새 추론했다.
현재 active 경로는 DSTE/J1/G0이며 G2·ADL·회복 분기를 추가하지 않았다.

실행 명령:
```bash
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python scripts/run_le2i130_20261003.py all
```

CPU 준비·채점·감사 단계는 runner가 CUDA_VISIBLE_DEVICES를 빈 문자열로 설정한다.
처음 sandbox 내 GPU 조회는 드라이버 접근 제한으로 실패했고, 승인된 GPU 0 범위에서 실행했다.
시스템·드라이버·공용 환경을 변경하지 않았다. 사전 준비 실패와 수정은 앞 절과 prepare 로그에 보존했다.
수정 후 단 한 번의 고정 모델 추론을 수행했으며 시험 결과에 맞춘 임계값·설정 변경은 없다.

{event_table}

{delta_table}

변화량은 반올림 전 수치로 계산했다.

{added_table}

{secondary_table}

사용자 모델 미탐 13건 중 11건은 기존 품질 gate로 분류기를 실행하지 않은 낙상 영상이다.
나머지 2건은 분류를 실행했으나 낙상 경보가 없었다. 추가 세 영상은 모두 gate 통과·탐지 성공이다.
Coffee_room_02/50에서 ST-GCN++의 71.96초, MS-G3D의 71.64초 경보는
허용 구간 시작 72.14초보다 빨라 오탐으로 집계되었다. 이후 73.56초 경보는 두 모델 모두 TP다.

독립 감사는 원주석 130개, ffprobe FPS, PTS 순서, detector 선택, 품질 gate, 3D fuse와 정규화,
CPU 함수식 J1/G0 출력, 네 모델의 원시 logits와 모든 지표를 검산했다.
기존 127개 영상의 경보·처리 여부·점수·창 수가 parent와 모두 동일하고,
추가 3개의 TP/FP/FN 합이 전체 변경량과 일치했다.

```json
{json.dumps(audit, indent=2, ensure_ascii=False)}
```

가중치·입력 관련 config 동일성: `{same}`.

```json
{json.dumps(journal, indent=2, ensure_ascii=False)}
```

산출물은 `data/fall_processed/RGB/le2i130_20261003_r1/`의 `evaluation.json`,
`independent_audit.json`, `comparison_127_to_130.json`, `cases.csv`, `contract.json`,
`cache_reuse.json`, `source_media.json`(각 영상 폴더), `logs/`에 보존했다.
`evaluation.json`의 pending_independent_audit 필드는 채점 당시 상태이며,
그 파일의 해시를 바인딩한 별도 `independent_audit.json` 및 `status.json`이 완료를 증명한다.
원본 평가 JSON을 사후 수정하지 않았다.

공유본에는 전체 성능, 추가 세 영상, 품질 미탐의 영향, 재사용 범위와 해석 한계를 반영했다.
명령·장비·실패 로그·해시는 공유본에 포함하지 않았다.
이 실험은 기존 공통 프로토콜의 Le2i130 확장이며 저자의 원본 추론/15프레임 채점 재현은 아니다.
원논문 성능과 동일 조건 우열을 주장하지 않는다. 이전 결과 파일은 보존했다.
'''
    text = internal.read_text().split('\n## 재평가 완료 및 독립 검산')[0]
    internal.write_text(text.replace('상태: `in_progress`', '상태: `completed`', 1) + completion)
    shared = ROOT / 'docs/shared' / (STEM + '_shared.md')
    shared.write_text(f'''# Le2i 전체 130개 영상 낙상 탐지 재평가

문서 ID: DOC-20261003-le2i130-reevaluation-R1  
갱신일: 2026-10-03  
상태: `completed`

**원본 정답에 따른 전체 130개 영상에서 사용자 모델의 낙상 사건 F1은 {pct(scores['own']['event']['f1'])}다.**
이전 평가에서 제외된 세 낙상을 모두 탐지하여 기존 127개 평가의 90.71%보다 {delta['own']['f1_delta_pp']:.2f}%p 높아졌다.
모델·입력 처리·결정 기준을 유지한 상태에서 평가 목록만 확장했다.

## 평가 범위와 조건

원본 Le2i 영상 130개, 낙상 99건, 비낙상 31개를 사용했다.
추가된 세 영상의 시작·종료 정답은 원본 주석 파일 중간에서 확인했고 공식 배포본과 일치했다.
[원본 정답 검증](2026-10-03_le2i_missing_annotation_search_shared.md)에 출처를 정리했다.
기존 127개와 추가 1개의 동일 조건 예측을 검증해 재사용하고 나머지 2개는 새로 추론했다.
전체 130개는 원본 정답으로 다시 채점했으며 품질 거부 영상도 분모에 포함했다.

사용자 모델은 고정 DSTE → J1 adapter → G0 낙상 분류 경로다.
비교 모델은 공식 SAFER 공개 가중치를 사용한 ST-GCN++, MS-G3D, 1D-CNN이다.
시험 데이터에서 추가 학습이나 임계값 조정을 하지 않았다.
모든 모델은 같은 영상에서 시간순 입력을 사용하며, 사용자 모델 64프레임과 비교 모델 48프레임의
종료 시점을 맞춰 8프레임 간격으로 분류한다. 시간축은 25Hz다.

주 지표는 낙상 클래스로 전환되는 경보를 정답 시작 0.5초 전부터 종료 3초 후까지의 구간에
일대일로 대응한 **낙상 사건 정밀도·재현율·F1**이다. 추가 경보와 허용 구간 밖 경보는 오탐이다.
이 규칙은 기존 공통 평가와 같다.

## 전체 130개 주 결과

{event_table}

TP는 탐지한 낙상 수, FP는 대응되는 정답이 없는 경보 수, FN은 놓친 낙상 수다.
사용자 모델은 99건 중 86건을 탐지했고, 13건을 놓쳤으며, 오탐 경보는 4건이었다.
세 비교 모델의 사건 F1이 사용자 모델보다 높았다.

## 추가 세 영상과 기존 결과의 변화

{delta_table}

변화량은 반올림 전 수치로 계산했다.

{added_table}

추가 세 영상의 낙상은 네 모델 모두 탐지했다.
ST-GCN++와 MS-G3D는 50번 영상에서 허용 시간보다 이른 경보를 각각 한 번 추가로 출력해
오탐이 1건씩 늘었다. 기존 127개 영상의 예측과 채점 결과는 모든 모델에서 그대로 유지되었다.

## 보조 지표: 영상별 낙상/비낙상 분류

{secondary_table}

이 보조 평가는 영상 안에서 낙상 예측이 한 번이라도 나오면 낙상 영상으로 판정한다.
경보 시각과 반복 오탐을 반영하는 주 지표와 계산 단위가 다르므로 두 F1을 혼용하지 않는다.

## 품질 기준과 해석 범위

사용자 모델은 130개 중 119개에서 분류를 수행했다. 나머지 11개는 기존 관절 품질 기준에
걸린 낙상 영상이며, 평가에서 제외하지 않고 모두 미탐에 포함했다.
전체 미탐 13건은 품질 거부 11건과 분류 실행 후 경보가 없었던 2건으로 구성된다.
추가된 세 영상은 모두 품질 기준을 통과했다.

원본 정답·시간축·입력 품질·3D 변환·사용자 분류 연산·원시 예측·전체 지표를 별도 코드로 검산했다.
이 결과는 고정된 외부 평가 조건에서의 결과다. 입력 표현과 품질 조건이 모델마다 다르므로
점수 차이를 특정 아키텍처나 adapter만의 효과로 해석하지 않는다.
또한 3D 변환은 영상 전체 문맥을 쓰는 오프라인 처리이므로 실시간 탐지 성능을 입증하지 않는다.

**이번 수치는 기존 공통 평가 프로그램의 결과다.** 저자의 원래 전체 추론 프로그램과
15프레임 허용폭 채점 프로그램을 그대로 실행한 수치가 아니므로 논문 보고 성능과 같은 조건의
재현 결과로 표기하지 않는다. 다른 외부 데이터셋의 기존 결과는 이번 작업에서 변경하지 않았다.

[모델별 결과 CSV]({STEM}_results.csv) · [130개 영상별 결과 CSV]({STEM}_cases.csv)
''')
    report_rows = []
    for m in NAMES:
        s = scores[m]
        row = dict(model=m, model_name=NAMES[m], dataset='Le2i130_original', videos=130, fall_events=99, processed=s['processed'])
        row.update({'event_' + k: v for k, v in s['event'].items()})
        row.update({'video_' + k: v for k, v in s['video'].items()})
        row.update(previous127_event_f1=delta[m]['previous127']['f1'], f1_delta_pp=delta[m]['f1_delta_pp'])
        report_rows.append(row)
    with (shared.parent / (STEM + '_results.csv')).open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(report_rows[0])); writer.writeheader(); writer.writerows(report_rows)
    shutil.copy2(OUT / 'cases.csv', shared.parent / (STEM + '_cases.csv'))
    if (OUT / 'runner_failure.json').exists():
        failed = read(OUT / 'runner_failure.json')
        assert failed['stage'] == 'prepare' and failed['started'] == '2026-10-03T12:32:46.640400+00:00'
        (OUT / 'runner_failure.json').rename(OUT / 'prepare_attempt1_failure.json')
    print(event_table)
    print(delta_table)


if __name__ == '__main__':
    main()
