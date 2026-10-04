# E12 — LLAVIDAL/VLM 해석 계층과 TrackMemory

- 문서 ID: `DOC-20260903-exp12-vlm-trackmemory-R1`
- 기준일: 2026-09-03
- 연구 상태: 설계 `completed`, end-to-end 통합 `planned`

## 연구 목표와 권한 구조

VLM은 낙상을 새로 판정하거나 detector 결정을 취소하는 모델이 아니라 이미 검출된 사건의 상황과
불확실성을 설명하는 interpreter로 사용한다.

```text
authoritative detector → event trigger → Pass A blind visual observation
                                    └→ Pass B detector-aware reconciliation
                                                 ↓
                                    deterministic output composer
```

- Detector가 alert와 event time을 결정한다.
- Pass A는 detector label·score 없이 영상의 관찰 사실을 구조화한다.
- Pass B는 detector context와 Pass A를 대조해 맥락과 불확실성을 작성한다.
- VLM은 alert를 suppress할 수 없다.
- VLM 실패·timeout·schema 오류 시 detector-only output으로 fallback한다.

## 기존 기록 기준 진단과 한계

LLAVIDAL standalone 진단에서는 160개 중 159개를 `FALL`로 예측했고 specificity는 0%였다. 따라서
VLM을 단독 fall classifier로 사용하지 않는다. Detector→VLM bridge, Pass A/B, composer와 통합
평가는 아직 완료되지 않았다. strict realtime과 multi-person 성능도 검증 전이다.

## TrackMemory

사람별 기록은 무제한 자연어 history가 아니라 서로 다른 수명의 bounded memory로 분리한다.

- short-term: 최근 pose, tracking continuity와 event evidence
- medium-term: 최근 낙상·회복·비활동 episode
- long-term: 명시적으로 허용된 축약 통계만 제한 보존

낙상, 배회와 장시간 비활동은 독립 detector/state machine이 계산하고 VLM에는 구조화된 event만
제공한다.

## 비식별 설계

- RGB와 audio는 가능한 한 edge에서 분석하고 원본을 서버로 보내지 않는다.
- 서버 payload는 event type, confidence/quality, 축약 시간과 비가역적으로 정규화된 표현으로 제한한다.
- Skeleton도 체형·보행 패턴으로 재식별될 수 있으므로 person-centered normalization, session-scoped
  random ID, 짧은 보존기간, 암호화와 접근통제를 함께 적용한다.
- YAMNet은 충격음·환경음의 보조 증거로만 사용하고 단독 낙상 판정 권한을 주지 않는다.
- 음성 원본이나 장기 audio embedding 보존은 별도 개인정보 영향평가 전에는 허용하지 않는다.

## 다음 연구

Core G0/G2/D1 detector와 event schema를 먼저 재현한다. 이후 detector-only fallback, Pass A/B의
confirmation-bias 감소, malformed output과 timeout을 고정 fixture로 검증하고 edge-only 입력 정책
아래 VLM/YAMNet의 추가가치를 평가한다.
