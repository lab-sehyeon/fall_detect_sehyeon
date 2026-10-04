# Git 협업 가이드

이 저장소는 Git-flow를 단순화한 `main` → `develop` → 작업 브랜치 구조를 사용합니다.

## 브랜치 역할

| 브랜치 | 역할 | 직접 push |
| --- | --- | --- |
| `main` | 검증이 끝난 release만 보관 | 금지 |
| `develop` | 다음 release를 준비하는 통합 브랜치 | 금지 |
| `feature/#<issue>` | 새로운 기능 개발 | 해당 작업자만 |
| `fix/#<issue>` | 버그 수정 | 해당 작업자만 |
| `refactor/#<issue>` | 기능 변경 없는 리팩터링 | 해당 작업자만 |
| `design/#<issue>` | UI·스타일 변경 | 해당 작업자만 |
| `test/#<issue>` | 테스트 추가·수정 | 해당 작업자만 |
| `docs/#<issue>` | 문서 추가·수정 | 해당 작업자만 |

`#`가 포함된 브랜치 이름은 shell에서 항상 따옴표로 감싸 사용합니다.

## Commit Convention

커밋 제목은 `<Type>: <summary>` 형식을 사용합니다.

| Type | 용도 | 예시 |
| --- | --- | --- |
| `Feat` | 기능 개발 | `Feat: add SAFER label alignment` |
| `Fix` | 버그 수정 | `Fix: correct temporal window indices` |
| `Docs` | 문서 수정 | `Docs: document dataset provenance` |
| `Refactor` | 코드 리팩터링 | `Refactor: separate feature builders` |
| `Design` | UI·스타일 변경 | `Design: update dashboard colors` |
| `Test` | 로직 및 코드 테스트 | `Test: add split integrity checks` |

- 제목은 명령형으로 간결하게 작성합니다.
- 한 커밋에는 하나의 논리적 변경만 포함합니다.
- 데이터, checkpoint, token, 개인정보와 로컬 경로는 커밋하지 않습니다.
- merge commit, revert와 Git의 `fixup!`·`squash!` 커밋은 검사에서 예외로 처리합니다.

로컬 commit message 검사를 활성화하려면 clone마다 한 번 실행합니다.

```bash
git config core.hooksPath .githooks
```

## PR Convention

| PR 종류 | 제목 표시 | shortcode |
| --- | --- | --- |
| Design | `🎨 Design` | `:art:` |
| Feature | `✨ Feature` | `:sparkles:` |
| Fix | `🔥 Fix` | `:fire:` |
| Test | `✅ Test` | `:white_check_mark:` |
| Refactoring | `♻️ Refactoring` | `:recycle:` |
| Docs | `📘 Docs` | `:blue_book:` |
| Release | `🚀 Release` | `:rocket:` |

예: `✨ Feature: add J1 fall specialist trainer`

작업 PR은 `develop`을 대상으로 하고, release PR만 `main`을 대상으로 합니다. 작업 PR은 검토 후
**Squash and merge**하고 작업 브랜치를 삭제합니다.

## 개발 시작

1. 프로젝트 저장소에서 Issue를 생성합니다.
2. `develop`을 최신 상태로 갱신합니다.
3. Issue 번호에 맞는 작업 브랜치를 생성합니다.

프로젝트 저장소를 직접 관리하는 경우:

```bash
git fetch origin
git switch develop
git pull --ff-only origin develop
git switch -c 'feature/#123'
```

개인 fork에서 작업하는 경우 `origin`은 개인 fork, `upstream`은 프로젝트 저장소를 가리킵니다.

```bash
git remote add upstream https://github.com/senior-action-recognition/fall_detection.git
git fetch upstream
git switch develop
git merge --ff-only upstream/develop
git push origin develop
git switch -c 'feature/#123'
```

## 개발 종료

1. 관련 테스트와 문서를 갱신합니다.
2. Commit Convention에 맞춰 커밋합니다.
3. 작업 브랜치를 자신의 `origin`에 push합니다.
4. 작업 브랜치에서 프로젝트의 `develop`으로 PR을 생성합니다.
5. 검토와 필수 검사를 통과한 뒤 Squash and merge합니다.
6. merge된 작업 브랜치를 로컬과 원격에서 삭제합니다.

```bash
git push -u origin 'feature/#123'
git switch develop
git pull --ff-only origin develop
git branch -d 'feature/#123'
```

fork 사용자는 merge 후 `upstream/develop`을 받아 자신의 `origin/develop`을 갱신합니다.

## Release

release 준비가 끝났을 때만 `develop` → `main` PR을 생성합니다.

1. 제목을 `🚀 Release: vX.Y.Z`로 작성합니다.
2. 전체 테스트, 데이터·checkpoint 제외 여부와 문서 링크를 재확인합니다.
3. 참여자가 변경 사항을 확인한 뒤 merge commit으로 병합합니다.
4. 병합된 commit에 `vX.Y.Z` annotated tag와 GitHub Release를 생성합니다.
5. release merge commit을 `develop`에 다시 반영합니다.

`main`에는 기능 브랜치나 개인 fork에서 직접 PR을 보내지 않습니다. 긴급 수정도 `fix/#<issue>` →
`develop` → `main` release 순서를 따릅니다.

## GitHub 보호 규칙

GitHub repository settings에서 `main`과 `develop`에 다음 branch protection 또는 ruleset을 적용합니다.

- pull request 없이 merge 금지
- `Branch policy` status check 통과 필수
- conversation resolution 필수
- force push와 branch deletion 금지
- `main` PR의 source branch는 `develop`만 허용
- `develop` PR의 source branch와 PR 제목은 이 문서의 convention을 따름

코드 리뷰 승인은 팀 규모에 따라 생략할 수 있지만, release PR은 참여자가 변경을 확인한 후
병합합니다.
