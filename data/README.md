# data/

무거운 데이터셋은 repo 안에 복사하지 않고 **이 폴더에 심볼릭 링크**를 만들어 쓴다. `data/*`는 gitignore이므로
링크는 커밋되지 않고(이 README와 `.gitkeep`만 추적), 코드는 어느 머신에서든 같은 경로 `data/<dataset>`로 읽는다.

```bash
# 실제 데이터가 있는 곳(공유 스토리지 등)을 가리키는 링크 만들기
ln -sfn /lustre/<user>/datasets/<dataset> data/<dataset>
ls -l data/            # 링크 대상 확인
```

## datamodule에서 쓰는 법

```python
from src.utils import require_data_path

root = require_data_path(Path(data_dir) / "<dataset>")   # 링크가 없거나 끊어졌으면 해결 방법이 적힌 오류
```

- `data_dir`은 config에서 `data_dir: ${paths.data_dir}`로 받는다 (= 항상 `<repo>/data/`). 데이터 위치를 바꾸고 싶으면 환경변수가 아니라 **링크 대상**을 바꾼다.
- 컴퓨트 노드에서도 같은 경로가 보여야 하므로, 링크 대상은 로그인 노드 전용 디스크(`/tmp`, `/home` 일부)가 아니라
  클러스터 공유 경로로 잡는다.
- 데이터 분할 파일(split csv 등)처럼 **작고 재현에 필요한 것은 링크가 아니라 repo에 커밋**한다 (예: `split/`).
- 새 데이터셋을 추가하면 아래 표에 링크 이름과 실제 위치(어디서 받았는지)를 적는다.

## 현재 목록

| 링크 | 실제 위치 | 출처 / 메모 |
|---|---|---|
