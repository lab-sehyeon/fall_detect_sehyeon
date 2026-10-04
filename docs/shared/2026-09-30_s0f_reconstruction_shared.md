# S0-F: posterior 기반 전이 보조 모델

- 문서 ID: `DOC-20260930-s0f-reconstruction-R1`
- 기준일: 2026-09-30
- 상태: `in_progress`

## 연구 질문과 보존 조건

고정 상태 분류기의 posterior만으로 낙상 후 눕기, 착석 완료, 누운 상태에서 일어나기를
구분할 수 있는지 검사한다. 기존 S0-C의16상태 출력은 변경하지 않는다.
입력은 posterior16차원, 시간차분16차원, entropy/최댓값/margin의 총35차원이며,
출력은 none 및10→12,6→4,12→7의4class다. 작은 causal TCN만학습한다.
Background와세event window를균형추출하고 sqrt inverse-frequency cap20/50을비교한다.

## 별도 재구현의 사전 조건

원본 기록에 없는 조건은 이번 실험의 명시적 설정이며 원본과의 동일성을 주장하지 않는다.
TCN은hidden64, dilation1/2/4, kernel3,왼쪽padding,dropout0.1을 사용한다.
두후보를동일초기화하고12epoch, batch512, AdamW학습률.001/weight decay1e-4로학습한다.
학습window는64frames/stride8, positive target은직접전이boundary±2frames다.
중복target은가까운boundary,동률이면이른boundary와class순으로정한다.
학습label은offline annotation이며모델입력에는포함하지않는다.

선택은validation의최저camera-view event Macro-F1→overall Macro-F1순이다.
동률은이른epoch→cap20으로정하며없는class도F1=0으로세class평균에포함한다.
예측event는nonzero argmax run의시작,matching은±12frames내일대일대응이다.
전체Macro-F1≥40%,최저view≥20%,세전이recall각≥40%,false events/min≤6을
모두충족해야통합가능으로판정한다. 결과를보고조건을완화하지않는다.

## 검증 순서와 한계

입력·동결·순차성검사후학습하고선택을고정한뒤test/OOD를평가한다.
이 holdout은선행모델에서이미평가했으므로새로운독립외부평가로주장하지않는다.
선택후validation AUPRC는표현력진단에만사용하며threshold를추가조정하지않는다.
TCN 자체는과거·현재만참조하지만상류상태분류timeline에는window미래문맥이있으므로
전체pipeline의실시간인과성을주장하지않는다. 직접전이3종은낙상후회복episode전체를대변하지않는다.
제한검증에서동결조건·인과성·학습재개동일성을확인하고전체실험을시작했다.
Train directevent는56/1336/96개,val은39/196/32개로기존기록의support와일치한다.
완료성능은아직없으며후속은문맥회복모델과고정외부평가다.

[선행 규칙 비교](2026-09-30_s0e_reconstruction_shared.md) ·
[관련 연구·참고문헌](2026-09-03_09_state_recovery_decoder_shared.md).
