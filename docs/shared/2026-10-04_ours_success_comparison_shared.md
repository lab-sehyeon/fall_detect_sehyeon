# 동일 영상에서 Ours가 검출하고 비교 모델이 놓친 낙상 사례

문서 ID: DOC-20261004-ours-success-comparison-R1

기준일: 2026-10-04. 상태: `completed`.

Ours는 낙상을 검출했지만 다른 모델이 적어도 하나는 놓친 동일 영상 사례가 Le2i 83개, CAUCAFall 22개 확인됐다. 다른 네 모델이 모두 놓친 낙상 영상은 없다. 논문 본문의 영상 단위 비교에는 Le2i `Coffee_room_01_014`와 CAUCAFall `FallForwardS9`를 우선 후보로 제안한다.

전체 평가는 이미 동일한 Le2i 130개와 CAUCAFall 100개 영상 및 정답을 사용했다. 이전의 모델별 보충그림은 각 모델에서 성공·실패 영상을 독립적으로 선정했기 때문에 그림마다 영상이 달랐다. 이번 분석은 공통 영상 ID로 다섯 모델의 판정을 연결한 것이며, 새 추론이나 판정 기준 변경은 하지 않았다.

[낙상 후보 전체 105개](2026-10-04_ours_success_comparison_candidates.csv) · [공통 영상 전체 230개 판정](2026-10-04_ours_success_comparison_all_cases.csv) · [전체 성능과 모델 출처](2026-10-04_external_comparison_summary_shared.md)

## 같은 영상의 판정 비교

TP는 실제 낙상 영상을 낙상으로 분류한 결과, FN은 낙상 영상을 놓친 결과다. 아래 표는 모두 **영상 단위** 판정이다.

| 데이터셋 | 영상 | Ours | USDRL + NTU60 | HFD | X3D-UDA | FLASH |
| --- | --- | --- | --- | --- | --- | --- |
| Le2i | Coffee_room_01_014 | TP | FN | FN | FN | TP |
| CAUCAFall | forward/FallForwardS9 | TP | FN | TP | FN | TP |
| CAUCAFall | side/FallLeftS2 | TP | TP | FN | TP | TP |
| CAUCAFall | forward/FallForwardS8 | TP | FN | TP | TP | FN 입력 실패 |
| Le2i | Coffee_room_01_003 | TP | TP | FN | FN | TP |

첫 두 후보는 다섯 모델 모두 입력 처리를 완료한 영상이다. Ours는 두 영상 모두 사건 TP/FP/FN=1/0/0으로 정답 사건을 검출했고 추가 경보는 없었다. `Coffee_room_01_014`에서 FLASH는 영상 TP이며 사건 매칭도 성공했지만 추가 경보가 2개다. `FallForwardS9`에서 FLASH는 영상 TP이고 추가 경보는 4개다. 따라서 이 두 그림에서 FLASH를 단순히 실패로 표시해서는 안 된다.

Le2i에서는 `Coffee_room_01_029`, `Home_01_007`, `Home_02_031`도 Ours TP이면서 USDRL·HFD·X3D가 모두 FN인 대체 후보다. CAUCAFall의 `side/FallRightS6`도 Ours TP, USDRL·X3D FN으로 첫 CAUCA 후보와 같은 영상 판정 조합을 보인다.

CAUCAFall에서 HFD와의 검출 차이를 보여주려면 `side/FallLeftS2`가 적합하다. 반면 `FallForwardS8`의 FLASH는 모든 프레임에서 포즈를 얻지 못해 분류기가 실행되지 않았다. 이는 전체 시스템의 입력 실패에 따른 FN이지, 정상 입력을 받은 분류기의 오판 사례는 아니다.

## 후보 수와 반대 방향 사례

아래 건수는 Ours TP인 같은 낙상 영상에서 비교 모델이 FN인 경우다. 모델별 영상이 중복될 수 있으므로 열의 합을 후보 총수로 사용하지 않는다.

| 데이터셋 | USDRL + NTU60 | HFD | X3D-UDA | FLASH | 적어도 한 모델 FN |
| --- | --- | --- | --- | --- | --- |
| Le2i | 14 | 66 | 41 | 0 | 83 |
| CAUCAFall | 19 | 2 | 3 | 1 | 22 |

Le2i 후보 83개는 모두 다섯 모델의 입력 처리가 완료됐다. CAUCAFall 후보 22개 중 21개는 모두 처리됐고, 나머지 하나가 FLASH 입력 실패 영상이다. 전체 공통 영상 중 모든 모델의 입력 처리가 완료된 영상은 Le2i 119개, CAUCAFall 92개다. 처리 실패 영상을 전체 성능의 분모에서 제거하지 않았다.

반대로 Ours는 FN이고 상대는 TP인 낙상 영상도 있다. USDRL/HFD/X3D/FLASH 순서로 Le2i는 0/5/10/13개, CAUCAFall은 0/11/11/11개다. 성공 사례의 선택만으로 전체 성능 우월성을 주장할 수 없는 이유다.

## FLASH의 영상 검출과 사건 검출 구분

FLASH는 Le2i에서 낙상 영상 단위 FN이 없으므로 Ours TP 대 FLASH 영상 FN 사례를 만들 수 없다. 그러나 Ours는 정답 낙상 사건을 검출하고 FLASH는 시점 매칭에 실패한 영상은 Le2i 9개, CAUCAFall 5개다. CAUCAFall의 5개에는 앞서 설명한 입력 실패 1개가 포함된다. X3D에는 사건 시각 출력이 없으므로 사건 검출 비교는 미측정으로 둔다.

예를 들어 Le2i `Coffee_room_01_003`의 GT 낙상 구간은 8.92–10.44초다. Ours의 경보는 10.20초이고 사건 TP/FP/FN=1/0/0이다. FLASH는 경보가 존재하므로 영상 TP지만, 경보가 0.56, 0.84, 4.16, 8.28초에 발생해 사건 TP/FP/FN=0/4/1이다. 기존 매칭 허용 범위인 GT 시작 0.5초 전부터 종료 3초 후까지에도 맞는 경보가 없다. 이 사례는 **영상 미탐이 아니라 낙상 시점 검출 실패**로 설명해야 한다. 오프라인 시각 정렬이며 실시간 처리 지연 측정은 아니다.

## 논문에 사용할 때의 해석 범위

주 후보는 모든 모델이 입력 처리를 완료하고 Ours가 추가 경보 없이 사건을 검출한 사례 중, 타 모델의 영상 FN 수가 많은 순서와 영상 ID 순서로 제안했다. 결과 확인 이후의 사후 선정이며 무작위 표본이 아니다. 기존 전체 정량 평가와 실패 사례도 함께 제시해야 한다.

동일 원영상과 정답을 비교하지만 학습 자료, 전처리, 입력 표현과 출력 집계는 모델마다 다르다. USDRL은 공식 backbone과 프로젝트 학습 NTU60 분류기의 조합이고, HFD와 FLASH는 변경 사항을 명시한 프로젝트 재학습 구성이다. 이 비교를 원논문의 보편적 성능이나 adapter 구조만의 효과로 일반화하지 않는다. 이번 작업은 후보 탐색이며 기존 논문 그림을 교체하지 않았다.
