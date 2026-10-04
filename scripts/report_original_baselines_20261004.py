"""Write paired reports only from independently verified original-baseline results."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/fall_processed/RGB/original_baselines_20261004_r2'
INTERNAL=ROOT/'docs/internal/2026-10-04_original_baseline_evaluation_internal.md'
SHARED=ROOT/'docs/shared/2026-10-04_original_baseline_evaluation_shared.md'
EVIDENCE=ROOT/'docs/internal/2026-10-04_original_baseline_evaluation_evidence'


def read(p): return json.loads(p.read_text())


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,r))+' |' for r in rows])


def main():
    ev=read(OUT/'evaluation.json');audit=read(OUT/'independent_audit.json')
    assert ev['passed'] and not ev['pending_independent_audit'] and audit['passed']
    assert audit['evaluation_sha256']==sha(OUT/'evaluation.json')
    names={'own':'사용자 DSTE/J1/G0','rthfd':'RTHFD 원본 코드'}
    ds={'le2i130':'Le2i 130개','cauca100':'CAUCA 100개'}
    event_rows=[];video_rows=[];breakdown=[]
    for scope in ds:
        for model in names:
            s=ev['summaries'][model][scope];e=s['event'];v=s['video'];b=audit['error_breakdown'][model][scope]
            event_rows.append([ds[scope],names[model],e['tp'],e['fp'],e['fn'],
                               *[f'{100*e[k]:.2f}%' for k in ('precision','recall','f1')],f"{s['processed']} / {s['videos']}"])
            video_rows.append([ds[scope],names[model],'/'.join(str(v[k]) for k in ('tp','fp','fn','tn')),
                               *[f'{100*v[k]:.2f}%' for k in ('accuracy','precision','recall','f1')]])
            breakdown.append([ds[scope],names[model],b['failed_positive_videos'],b['processed_positive_no_alarm'],
                              b['positive_alarm_without_match'],b['negative_alarm_videos'],b['positive_unmatched_events']])
    et=table(['데이터셋','모델','TP','FP','FN','정밀도','재현율','사건 F1','처리 / 전체'],event_rows)
    vt=table(['데이터셋','모델','영상 TP/FP/FN/TN','정확도','정밀도','재현율','영상 F1'],video_rows)
    bt=table(['데이터셋','모델','처리 실패 낙상 영상','처리 후 무경보','경보 있으나 시간 불일치','오경보 비낙상 영상','낙상 영상 미매칭 경보'],breakdown)
    run_lines=(EVIDENCE/'rthfd_run.log').read_text().splitlines()
    terminal=[json.loads(x) for x in run_lines if x.startswith('{') and 'inference_complete' in x][-1]
    replay=read(OUT/'alarm_replay.json')
    completion=f'''

## 실제 실행 완료·독립 검산 결과

RTHFD 230개 완료. 원본 pose 추론 시간 {terminal['elapsed_seconds']:.2f}초, 처리 실패 {terminal['failures']}개.
R1의 누락 observer를 수정한 R2 전체 경보 재생성 {replay['seconds']:.2f}초.
원본의68개 true-fall 분기를 모두 연결했다. R1 대비 {replay['changed_alarm_frames']}프레임의
경보 누락을 수정했으며 모든 원본 관절·counter·flag·판정 조건·threshold는 동일하다.
CPU만 사용했으며 학습·calibration·rule 변경은 없었다. 사용자 예측230개는 부모 결과와 동일하다.
독립 감사: 원영상 {audit['decoded_frames_verified']:,}프레임 픽셀·PTS 재확인,
동일 수의 원본 URFD rule 재생성, {audit['pose_model_replayed_frames']}프레임 모델 재추론,
최대 pose 절대 차이 {audit['max_pose_abs_error']:.9g}, 모든 사건/영상 혼동행렬 일치.
감사 시간 {audit['seconds']:.2f}초. 460개 모델별 사례와 비교 CSV를 저장했다.

{et}

{vt}

오류 분해:

{bt}

실행·감사 명령:

```bash
python scripts/evaluate_original_rthfd_20261004_r2.py
CUDA_VISIBLE_DEVICES='' python scripts/audit_original_rthfd_20261004.py
```

근거 파일은 output의 `evaluation.json`, `independent_audit.json`, `cases.csv`, `comparison.csv`,
각 영상 `rthfd.npz`와 처리 receipt다. `report_validation.json`은 문서와 감사 파일 해시를 묶는다.
유효 출력은 `data/fall_processed/RGB/original_baselines_20261004_r2`다. R1은 잘못된 observer의
기록으로 보존하며 논문 결과에서 제외한다. R2 config와 contract는 원본 pose 재사용 전에 고정했다.
원본 자산 취득/HFD·SDFA 미확보 근거는 evidence의 `checkpoint_availability.json`에 별도로 남겼다.

상태를 구분한다: RTHFD 실제 외부 평가·검산 `completed`; HFD/SDFA 원본 모델 평가 `paused`.
후자는 점수0이 아니라 실행 불가/미평가다. 재개에는 저자 학습 완료 SVM/checkpoint 및 학습 출처가
필요하다. 저자에게 연락하거나 학습을 대신 수행하지 않았다. 존재하지 않는 결과를 채우지 않았다.
논문용 claim은 현재 확인된 원본 RTHFD 대비 결과에 한정하며 세 비교 모델 모두 실행했다고 하지 않는다.

공유본에는 위 지표·성능 해석·원본 미확보와 동일 조건의 한계를 반영했다. 로컬 경로·환경 설치·해시·
실행 로그·취득 진단 실패는 내부 문서에만 유지했다. 기존 네 모델 평가 산출물과 checkpoint는 변경하지 않았다.
'''
    text=INTERNAL.read_text()
    marker='\n\n## 실제 실행 완료·독립 검산 결과'
    if marker in text:text=text.split(marker)[0]
    text=text.replace('상태: `in_progress`','상태: RTHFD 평가 `completed`, HFD·SDFA 원본 미확보 `paused`',1)
    text=text.replace('문서 ID: DOC-20261004-original-baseline-evaluation-R1','문서 ID: DOC-20261004-original-baseline-evaluation-R2',1)
    INTERNAL.write_text(text+completion)
    shared=f'''# 저자 원본 모델의 외부 낙상 탐지 비교

문서 ID: DOC-20261004-original-baseline-evaluation-R2  
기준일: 2026-10-04  
상태: RTHFD 평가 `completed`, HFD·SDFA 원본 미확보 `paused`

## 완료 범위

RTHFD의 원본 MoveNet Thunder v3와 저자 공개 낙상 판정 코드를 사용해,
사용자 DSTE/J1/G0와 동일한 Le2i 130개·CAUCA 100개에서 평가했다.
RTHFD는 이번에 230개 영상을 실제 추론했으며 처리 실패는 {terminal['failures']}개였다.
사용자 모델은 해당 영상들에서 검산을 마친 고정 예측을 재사용했다.
추가 학습·fine-tuning·타깃 threshold 조정은 수행하지 않았다.

| 비교 후보 | 실제 확보한 내용 | 평가 상태 |
| --- | --- | --- |
| HFD 3D-CNN+SVM | 공식 코드와 일반 C3D 사전학습 정보; 학습 완료 낙상 SVM 미확보 | 미평가 |
| SDFA | 공식 모델·평가 설정; 학습 완료 낙상 checkpoint 미확보 | 미평가 |
| RTHFD | 원본 Thunder v3와 저자 고정 판정 규칙 | 230개 평가 완료 |

HFD의 일반 특징 추출기만으로 학습된 fall/ADL SVM을 대신할 수 없다.
SDFA의 공개 설정에 적힌 가중치 경로도 실제 가중치 배포 파일과 구분한다.
두 모델의 미평가를 성능 0으로 간주하지 않는다. 공식 저장소 이력·배포 페이지·저자 페이지·
모델 허브에서 필요한 파일을 확보하지 못했으며, 공개되지 않은 원본의 존재 여부는 판단하지 않는다.
([HFD 공식 코드](https://github.com/ekramalam/HFD_3DCNN), [SDFA 공식 코드](https://github.com/saniazahan/SDFA))

## 같은 조건으로 비교한 범위

- Le2i는 130영상·낙상99건·비낙상31영상, CAUCA는 100영상·낙상50건·비낙상50영상이다.
- 같은 영상 파일과 구간 정답을 사용했다. 순서와 실제 프레임 시각을 확인했다.
- 경보가 꺼짐에서 켜짐으로 바뀔 때 사건 경보를 생성한다. 정답 시작 0.5초 전부터 종료 3초 후까지
  일대일 매칭하며, 추가 경보는 FP로 집계한다. 별도 refractory/smoothing은 적용하지 않았다.
- 영상 단위 예측은 영상 중 경보가 하나라도 있으면 fall이다. 사건 시간 일치 여부는 별도 지표다.
- 사용자 모델의 품질 조건에 따른 처리 실패도 제외하지 않고 전체 분모에 포함했다.

RTHFD의 입력 색상 순서·크기 변환·기하 조건·counter 처리는 저자 코드를 유지했다.
GMDCSA용과 URFD용 공개 코드의 낙상 판정 로직이 동일함을 확인했다.
원본 코드의 모든 true-fall 선언을 포착해 공통 평가기에 연결했다.
([RTHFD 공식 코드](https://github.com/ekramalam/RTHFD))

## 사건 단위 낙상 탐지

{et}

## 영상 단위 fall/non-fall 분류

{vt}

영상 F1은 낙상 영상에서 경보를 한 번이라도 냈는지 평가한다. 사건 F1은 경보 시각과 정답 구간을
비교하고 반복 경보도 오경보로 세므로, 같은 모델의 두 F1이 다를 수 있다.

## 검증과 해석 범위

모든 처리 프레임의 시간순서·픽셀을 대조하고, 원본 규칙으로 경보를 다시 계산했다.
영상마다 사전 정의한 처음·중간·마지막 프레임의 모델 출력을 재실행했다.
이 검산과 별도 사건·영상 집계에서 저장된 결과와 일치함을 확인했다.

{bt}

위 결과는 저자 모델과 판정 규칙을 공통 외부 평가에 연결한 결과다. 원 논문의 데이터 분할이나
평가 프로그램 전체를 재현한 점수로 제시하지 않는다. 원본 코드의 조건식을 논문 설명에 맞춰
임의로 고치거나 성능을 보고 변경하지 않았다.
RTHFD의 원 논문 source 데이터셋 성능 재현은 이번 실행에서 확인하지 않았다.
따라서 공개 코드·가중치의 해당 외부 입력에 대한 결과로 해석하며, 논문 보고 성능 자체의
재현 성공까지 입증한 것으로 표현하지 않는다.

RTHFD는 모든 원프레임마다 판정하며 사용자 모델은 기존 시간창 간격으로 판정한다.
출력 빈도·pose 추정기·입력 표현·후처리 특성이 다르므로 차이를 특정 구성요소의 효과로 단정하지 않는다.
반복 경보의 영향도 영상 지표와 함께 해석해야 한다.

기존 타깃 관찰 이력이 있으므로 완전히 관측하지 않은 blind test라고 주장하지 않는다.
이번 평가에 맞춘 추가 학습·보정은 없었다. 기반 pose 모델의 사전학습 이미지까지
중복이 없음을 검증한 것은 아니다. 결과는 두 외부 데이터셋과 고정된 현 실행 조건의 범위다.
HFD·SDFA까지 포함한 세 원본 비교 모델 전체 평가가 완료되었다고 쓰지 않는다.

## 다음 실행에 필요한 자료

HFD는 Le2i·CAUCA를 사용하지 않고 학습한 저자 SVM, SDFA는 해당 타깃을 사용하지 않은
저자 낙상 checkpoint와 그 학습·입력 조건이 확보되면 같은 목록으로 평가를 이어갈 수 있다.
현재 미확보를 해소하기 위한 새 학습은 이번 비교 범위에 포함하지 않는다.
'''
    SHARED.write_text(shared)
    availability=read(EVIDENCE/'checkpoint_availability.json')
    availability['RTHFD']['status']='completed_and_independently_audited'
    availability['RTHFD']['videos']=230
    (EVIDENCE/'checkpoint_availability.json').write_text(json.dumps(availability,ensure_ascii=False,indent=2)+'\n')
    summary='RTHFD 원본230개 실제 평가·독립 검산; HFD SVM·SDFA checkpoint 미확보'
    for rel,entry in [
        ('docs/README.md',f'| [저자 원본 RTHFD 외부 평가](shared/{SHARED.name}) | {summary} | RTHFD `completed`, 두 모델 `paused` |'),
        ('docs/shared/README.md',f'| [저자 원본 RTHFD 외부 평가]({SHARED.name}) | {summary} | RTHFD `completed`, 두 모델 `paused` |'),
        ('docs/internal/README.md',f'- [저자 원본 RTHFD 외부 평가]({INTERNAL.name}): {summary}')]:
        path=ROOT/rel;t=path.read_text()
        if '2026-10-04_original_baseline_evaluation' not in t:
            anchor='| [HFD·SDFA·RTHFD 후보 검토]' if rel!='docs/internal/README.md' else '- [HFD·SDFA·RTHFD 후보 검토]'
            assert anchor in t;t=t.replace(anchor,entry+'\n'+anchor,1)
        if rel=='docs/README.md':t=t.replace('- 기준일: 2026-10-03','- 기준일: 2026-10-04',1)
        path.write_text(t)
    for kind in ('internal','shared'):
        path=ROOT/f'docs/{kind}/2026-10-03_hfd_sdfa_rthfd_feasibility_{kind}.md';t=path.read_text()
        link=f'\n> 후속 확인: [2026-10-04 원본 모델 평가](2026-10-04_original_baseline_evaluation_{kind}.md)에서 RTHFD 230개 평가·검산을 완료했다. HFD/SDFA는 원본 학습 파일 미확보로 미평가다.\n'
        if '2026-10-04_original_baseline_evaluation' not in t:
            first,rest=t.split('\n',1);path.write_text(first+'\n'+link+rest)
    # Numeric correspondence and shared-document hygiene.
    assert len(list(csv.DictReader((OUT/'cases.csv').open())))==460
    assert '/home/' not in shared and 'sha256' not in shared and 'pip ' not in shared and 'scripts/' not in shared
    for r in event_rows:
        assert '| '+' | '.join(map(str,r))+' |' in shared
    validation=dict(passed=True,internal_sha256=sha(INTERNAL),shared_sha256=sha(SHARED),
        evaluation_sha256=sha(OUT/'evaluation.json'),audit_sha256=sha(OUT/'independent_audit.json'),
        shared_numeric_rows_verified=len(event_rows)+len(video_rows),case_rows=460,
        missing_models_have_no_scores=True,shared_hygiene_checked=True)
    (OUT/'report_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    print(json.dumps(validation))


if __name__=='__main__':main()
