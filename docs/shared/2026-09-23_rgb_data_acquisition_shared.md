# RGB·외부 평가 데이터 준비

- 문서 ID: `DOC-20260923-rgb-data-acquisition-R2`
- 기준일: 2026-09-28
- 상태: `completed` — OOPS818·SAFER OOD30 자료 확보 범위

후반 RGB·외부 진단을 위해 기존 연구와 대응하는 OOPS-Fall 818개 영상과 공식 시간 구간 라벨,
SAFER OOD 30개 원영상을 확보했다. 영상의 해상도·FPS·인코딩은 변경하지 않았다.
SAFER normal 467개 영상은 아직 준비되지 않았으므로 전체 RGB 입력 확보 완료는 아니다.
자료 수집과 모델 학습·외부 평가 완료는 구분한다.

OOPS의 공식 대응표818행과 라벨4,022행 및 영상 집합의 일치를 확인했다.
SAFER OOD30개는 공식 체크섬이 모두 일치했다. 영상 첫·마지막 프레임 검사를 기본으로 했으며
6개에는 전체 프레임 검사를 적용했다. 후속 OOPS 전체276,539프레임 검사도 완료했다.
[입력 검증 결과와 남은 정합](2026-09-28_rgb_data_preparation_shared.md)을 별도로 정리했다.
자료 확보 완료는 모델 학습·평가 완료를 뜻하지 않는다.

OOPS 영상은 [원 데이터셋](https://oops.cs.columbia.edu/data/)에서,
대상 영상의 대응표와 annotation은 [OmniFall 공식 배포](https://huggingface.co/datasets/simplexsigil2/omnifall)에서 가져온다.
기존 OOPS 진단 이력이 있으므로 새 평가를 pristine untouched test로 표현하지 않는다.

## 선행 스켈레톤 결과

별도 사후 평가 정의를 명시한 V3 R3의 전체 전처리·동일 조건 기준 모델·J0/J1·FU 고정 비교가 완료됐다.
전체 497개/8,091,357프레임의 전처리 검산이 통과했고 경계 이상 비율은 97.155% 감소했다.
V3 기준 모델의 test/OOD fall F1은 80.551/56.109%, J1의 FU nested F1은 93.333%였다.
J1의 SAFER 고정 5-fold ensemble test/OOD fall F1은 79.709/52.915%였다.
전처리 개선이 모든 분류 결과의 개선을 보장하지 않으며, 원본과 수치적으로 동일한 복원도 아니다.

[R3 평가 정의와 한계](2026-09-22_v3_valid_support_shared.md) ·
[연구 전체와 참고문헌](2026-09-03_project_complete_summary_shared.md)
