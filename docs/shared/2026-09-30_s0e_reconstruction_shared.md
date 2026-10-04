# S0-E 전이 민감 상태 규칙 비교

- 문서 ID: `DOC-20260930-s0e-reconstruction-R2`
- 기준일: 2026-09-30
- 상태: `not_selected`

## 연구 질문과 방법

고정된 S0-C 상태 출력에서 fall→lying_down, sitting_down→sit,
lying_down→getting_up을 명시적 순차 규칙으로 보정할 때 전이 검출과 전체 상태 품질을 함께
보존할 수 있는지 평가한다. 새로운 신경망을 학습하지 않으며 별도 낙상 경보 모델은 변경하지 않는다.

Fall settle8/12/16, recovery confirmation3/5/8, recovery hold5/8,
sit confirmation0/5/8/12의72조건을 validation에서 비교한다.
현재 선행 C는 A+EMA0.5/연속3프레임/margin0.05다. 과거 선택 EMA0.70과 구분한다.
입력fall은그대로보존하고,fall후지속ground를lying_down으로확인한뒤non-ground가연속관측되면
getting_up을제한된프레임동안출력한다. 규칙은sequence별reset하고과거출력을수정하지않는다.

## 별도 재구현에서 고정한 조건

기존 기록에 없는 reset·non-ground 범위·tie와8개gate의수치조건은이번실험에서사전명시했다.
Non-ground는class1~9,13~15이며no_label은확인증거에서제외한다. 새fall은즉시규칙상태를reset한다.

안전 조건은Macro-F1 하락≤0.5%p,fall mask정확보존,상태전환비증가,lying_down F1하락≤2%p다.
전이 조건은fall→lying_down 및lying_down→getting_up의successor recall각각+10%p,
sit premature -5%p, sit successor recall하락≤2%p다. 모두통과해야채택한다.
실패시가장많은gate를통과한diagnostic후보를고정하며,성능을보고조건을완화하지않는다.
동률은Macro-F1→SegmentF1@50→Edit→선언순서로해결한다.

선택후고정test/OOD평가를수행한다. 해당자료는이미앞선모델에서평가됐으므로새로운독립외부평가가아니다.
상류window의미래문맥때문에전체실시간인과성은주장하지않는다.
과거negative result를강제로재현하는것이아니며,원본과수치적동일성도주장하지않는다.

## 완료 결과

72조건중채택가능후보는0개다. 가장많은기준을통과한진단후보는settle16/recovery3/hold5/sit5이며
8개중4개기준을충족했다. Validation Macro-F1은60.813%,lying_down F1은7.925%다.
Test Macro-F1은59.290%,OOD는34.326%다. 낙상출력은정확히보존했다.
Fall→lying_down successor recall은크게증가했지만전체분류·lying_down품질·전환횟수·회복전이기준을
함께충족하지못했다. 따라서규칙을채택하지않고기존 S0-C를유지한다.
순차성·후보전수재생·선택고정·평가검증은완료했다. 다음은[S0-F 전이보조모델](2026-09-30_s0f_reconstruction_shared.md)이다.

[선행 S0-B/C 결과](2026-09-29_s0bc_results_analysis_shared.md) ·
[기존 연구와 참고문헌](2026-09-03_09_state_recovery_decoder_shared.md).
