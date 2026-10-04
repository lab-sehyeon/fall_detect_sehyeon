# J0/J1 공동학습·고정 평가·nested 검증 재구현

- 문서 ID: `DOC-20260922-joint-reconstruction-R4`
- 기준일: 2026-09-22
- 상태: `completed` — V1/V2 전체 학습·평가·독립 검산 완료

## 연구 방법

V1/V2 입력을 분리해 frozen DSTE-TS2048D에 SAFER4-class와 FU binary head를 학습한다.
J0는 두 dataset-specific linear heads, J1은zero-init2048→256→2048 shared residual을 추가한다.
한 step에서각dataset batch를같은가중치로사용하되FU clip label을frame에복제하지않는다.
J1의epoch0가J0와동일한지확인하고개선이없으면epoch0로돌아갈수있도록한다.

문서로확인되는조건은유지하고유실된세부는별도재구현으로실행전에명시한다.
AdamW와train-only inverse-sqrt class weighting,LayerNorm/GELU residual을사용한다.
선택은SAFER validation macro-F1과지정FU validation F1의평균을우선하며AP로동률을구분한다.

## 평가 단계

1. Development:5-fold FU validation을사용한J0/J1비교. 개발OOF이며최종일반화수치가아니다.
2. Locked SAFER:모든fold모델고정후test/OOD평가,raw-logitmean ensemble.
3. Nested FU:outerk는평가전용,inner(k+1)%5는선택용,나머지3fold로학습한다.
4. Final fit:이번nested inner-best epoch중앙값으로SAFERtrain과FU전체에고정횟수학습한다.

원본과동일한수치복원을주장하지않는다. Nested도기존설계를본후의post-design control이며,
final fit의학습데이터재평가를일반화성능으로사용하지않는다.
V1의 전체 개발·고정 평가·nested·최종 학습·독립 검산을 완료했다.
V2도 별도 계보에서 전체 검증을 완료했다. 아래는 새단일seed 결과이며 원본 성능이 아니다.

| V1 평가 | J0 | J1 |
| --- | ---: | ---: |
| FU development F1 | 94.895% | 95.441% |
| FU nested F1 | 92.771% | 93.373% |
| FU nested precision | 92.216% | 92.814% |
| FU nested recall | 93.333% | 93.939% |
| FU nested lying FP/168 | 13 | 12 |
| SAFER locked test fall F1 | 72.235% | 76.976% |
| SAFER locked OOD fall F1 | 39.872% | 48.167% |

993개outer OOF가각1회포함되는지와모델·출력·지표·선택을독립검산했다.
이번nested중앙값에따라최종J0/J1을23/14epochs학습했다. 위locked결과는개발5foldensemble이며
최종단일모델의외부성능과동일하지않다. 단일seed의차이를보편적개선으로주장하지않는다.

| V2 평가 | J0 | J1 |
| --- | ---: | ---: |
| FU development F1 | 94.955% | 95.441% |
| FU nested F1 | 93.093% | 92.683% |
| FU nested precision | 92.262% | 93.252% |
| FU nested recall | 93.939% | 92.121% |
| FU nested lying FP/168 | 13 | 11 |
| SAFER locked test fall F1 | 76.743% | 79.281% |
| SAFER locked OOD fall F1 | 50.526% | 53.255% |

V2의nestedJ1은J0보다오탐이2개줄었지만정탐도3개줄어F1이낮았다.
오탐억제와recall손실을함께보고한다. 이번nested중앙값으로최종J0/J1을14/12epochs학습했다.
모든fold·선택·고정예측·지표를독립검산했으며, 후속V3는별도비교로진행한다.

[전체 진행](2026-09-22_recovery_execution_shared.md) · [연구 맥락·참고문헌](2026-09-03_project_complete_summary_shared.md)
