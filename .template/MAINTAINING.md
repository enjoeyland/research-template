# template을 고칠 때의 규칙

이 폴더(`.template/`)는 **template repo 전용**이다. 새 프로젝트를 만들 때 `make rename NAME=...`이 이 폴더를 삭제하므로 프로젝트에는 영향이 없다.
(template repo 자신에서는 `make rename`이 거부한다: `.template-version`의 `url`과 `origin`이 같으면 멈춘다. 일부러 하려면 `FORCE=1`.)

## 어디에 쓰나

- **template을 관리하는 기록은 `.template/`에 쓰고 `docs/`에는 쓰지 않는다**: 변경 이력과 버전(`CHANGELOG.md`), 다운스트림 현황(`DOWNSTREAM.md`), 이 규칙.
  `docs/`는 각 프로젝트의 문서라서, template의 이력이 들어가면 새 프로젝트로 그대로 딸려 간다.
- 예외: 구조나 규칙 **자체**가 바뀌면 그것은 새 프로젝트에도 쓰이는 참조 문서라서 현재 상태에 맞게 고친다: `docs/adr/adr-template-structure.md`의 본문(구조, 규칙, 사용법)과 해당 폴더의
  README. 이력("무엇이 언제 바뀌었나")은 여기가 아니라 `CHANGELOG.md`.
- `CLAUDE.md`는 사용자가 관리한다. 고칠 점이 보이면 제안만 한다(수정은 사용자가 허락한 것만).

## 무엇이 template에 속하나

- 연구 프로젝트에 **공통**인 것만 넣는다. 특정 프로젝트의 데이터셋, 모델 이름, 실험 이름, 날짜가 든 사례는 넣지 않는다(주석에도).
- 프로젝트에서 먼저 만든 개선을 가져올 때는 일반화해서 가져온다(`.template/DOWNSTREAM.md`의 "새 소식"을 먼저 확인).
- 클러스터 고유 정보(파티션, 불량 노드 목록)는 `scripts/sbatch/profiles/`에 근거와 함께 둔다.

## 버전을 올리는 기준과 절차

다운스트림이 **영향받는 변경 묶음마다 한 번** 버전을 올린다(변경마다 올리지 않는다). 올릴 때:

1. `CHANGELOG.md` 맨 위에 `## vN (YYYY-MM-DD) — 제목`을 추가한다. 항목: 무엇이 바뀌었나, **다운스트림에서 할 일**, **영향 경로**.
2. `.template-version`의 `version:`을 N으로 올린다(둘이 어긋나면 `tests/test_template_version.py`가 실패한다).
3. 코드, 설정, 테스트, 문서와 이 두 파일(`CHANGELOG.md`, `.template-version`)을 **같은 커밋**에 넣는다.
4. 커밋 뒤에 태그 `template-vN`을 단다(`git tag template-vN`). 태그를 원격에 올리는 `git push --tags`는 사용자에게 확인하고 push할 때 같이 한다.
5. `DOWNSTREAM.md`에서 프로젝트별로 영향이 있는지 본다.

## 검증 (template 변경은 "내 환경에서만 돌아간다"가 되기 쉽다)

- **깨끗한 clone에서 사용 체크리스트를 그대로 따라 한다**: `git clone`, `cp .env.example .env`, `make test`, toy smoke(`debug=smoke logger=csv`).
- sbatch 인프라를 건드렸으면 실제 제출로 확인한다(`PROFILE=cpu LOGGER=csv ./scripts/sbatch/template.sh`, 필요하면 `EXTRA_ARGS="debug=smoke"`). 로컬 실행으로 sbatch를 대신 검증하지 않는다.
- 공유 설정(defaults 리스트 등)을 바꾸면 모든 실험 config가 train/eval/analyze에서 조립되는지 테스트로 확인한다(`tests/test_configs.py`).
- 같은 실험 이름을 동시에 두 번 제출해서 검증하지 않는다(같은 seed가 같은 경로에 동시에 쓴다).
