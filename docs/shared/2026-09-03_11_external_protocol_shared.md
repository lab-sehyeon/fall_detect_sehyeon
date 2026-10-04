# E11 — 외부 데이터 적격성·봉인 평가 프로토콜

- 문서 ID: `DOC-20260903-exp11-external-protocol-R1`
- 기준일: 2026-09-03
- 연구 상태: `planned`

## 목적

외부 영상을 실행했다는 사실과 독립 일반화 검증을 구분한다. 모델 추론 전에 데이터 source,
annotation, 시간축, decoder와 event scorability를 고정한다.

## 추론 전 적격성 기준

1. 공식 배포처, version과 license를 확인한다.
2. fall onset/end, recovery와 negative annotation 의미를 확정한다.
3. container, frame count, FPS와 annotation 시간축의 일관성을 검사한다.
4. 실제 평가 decoder로 전체 영상을 완독한다.
5. 모델 입력 window를 구성할 수 있는지 확인한다.
6. 사전 정의한 tolerance로 GT·예측 event를 평가할 수 있는지 확인한다.

`window_capable`과 `event_scorable`은 별도로 판정한다. 영상이 충분히 길어도 GT 시간축이 불명확하면
event metric에 포함하지 않는다.

## 데이터셋 역할

| Dataset | 연구 역할 | 제한 |
| --- | --- | --- |
| AIHub 71641 | sealed final candidate | 사전 등록 후 단일 확증 평가 |
| HQFSD | development external | 모델 선택·오류분석 가능, final claim과 분리 |
| MCFD | cross-view external | view별 protocol 사전 고정 |
| CAUCAFall | 반복 diagnostic | pristine test 아님 |
| OOPS/OmniFall | front-end·event failure audit | 수정 전후 matched 결과 보존 |
| Le2i/URFD | engineering external | 작은 표본·분포 편향 명시 |

## 봉인 원칙

- 제외 기준, checkpoint, threshold, window/stride와 event decoder를 결과 전에 고정한다.
- 결과 확인 뒤 수정하면 해당 데이터는 development로 재분류한다.
- 최종 보고에는 전체·제외 수, decode failure, window/event 적격성을 함께 제시한다.
- 역사적 결과와 복구 후 재현 결과를 별도 namespace로 관리한다.

## 다음 연구

Core skeleton 및 RGB 경로 재현 후 development external로 오류 유형을 고정하고, 개선안 확정 뒤 sealed
dataset을 한 번만 평가한다.
