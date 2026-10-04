# third_party/

이 프로젝트가 참고하거나 가져다 쓰는 **외부 코드**를 git submodule로 둔다. 연구가 바뀌어도 계속 두는 폴더다.

## 규칙

- 원본을 직접 수정하지 않는다. 고쳐야 하면 GitHub에서 fork한 뒤 submodule URL을 fork로 바꾼다.
- 우리 코드(`src/`)는 `third_party/<repo>`를 import하거나 필요한 부분만 `src/`로 옮겨 쓴다. 옮겨 쓸 때는 출처
  경로와 commit을 주석으로 남긴다.
- 이 폴더의 파일은 우리 테스트 대상이 아니다.

## 환경: 기본적으로 repo마다 venv를 따로 만들어 쓴다

외부 repo는 우리 프로젝트와 의존성(torch, numpy 등 버전)이 다르고 서로도 충돌하므로, **기본적으로 각 third_party repo는
자기 venv를 만들어서 실행한다.** 우리 프로젝트 venv에 그 repo의 requirements를 섞어 설치하지 않는다.

```bash
python -m venv /scratch2/$USER/venvs/<repo>      # 이름은 third_party 폴더 이름과 같게
source /scratch2/$USER/venvs/<repo>/bin/activate
pip install -U pip
pip install -r third_party/<repo>/requirements.txt   # 또는 그 repo의 README 설치법
```

- venv는 repo 안이 아니라 `/scratch2/$USER/venvs/`에 둔다 (공유 스토리지, git 대상 아님).
- 우리 코드에서 그 repo를 import해야 하면 먼저 같은 venv로 돌릴 수 있는지 확인하고, 안 되면 필요한 부분만 `src/`로 옮겨 쓴다
  (아래 import 예시는 의존성이 우리 venv와 충돌하지 않는 경우에만).
- sbatch로 돌릴 때는 프로필 대신 `VENV=/scratch2/$USER/venvs/<repo>`를 지정한다 (`scripts/sbatch/common/env.sh`).
- 새 venv를 만들면 아래 "현재 목록" 표에 venv 이름과 설치 방법을 적는다.

## 추가 / 갱신

```bash
git submodule add https://github.com/<owner>/<repo>.git third_party/<repo>
git submodule update --init --recursive           # clone 직후
git clone --recurse-submodules <this-repo-url>    # 처음부터 받을 때
git -C third_party/<repo> checkout <commit> && git add third_party/<repo>   # 버전 고정
```

## import 예시

```python
import sys
sys.path.insert(0, "third_party/<repo>")   # 패키지로 설치되지 않는 코드
import <module>
```

## 현재 목록

| 폴더 | 출처 | 용도 | venv |
|---|---|---|---|
