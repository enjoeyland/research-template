# src/data/

`LightningDataModule`을 둔다(데이터 로딩 부품은 `components/`). 참고 구현: `toy_datamodule.py`(다운로드 없는 합성 데이터, 새 프로젝트에서 교체).
설정은 `configs/data/<name>.yaml`이고 `train.yaml`/`eval.yaml`/`analyze.yaml`이 `data: <name>`으로 고른다.

## 규칙

- **데이터 경로는 config로 받는다**: `data_dir: ${paths.data_dir}`(= 항상 `<repo>/data/`). 무거운 데이터셋은 repo에 복사하지 않고 `data/<dataset>`에 심볼릭 링크로
  두며([data/README.md](../../data/README.md)), datamodule은 `require_data_path(Path(data_dir) / "<dataset>")`로 열어 링크가 없거나 끊어졌을 때
  해결 방법이 적힌 오류를 받는다.
- **분할은 재현 가능하게**: 데이터 자체는 seed와 무관하게 고정하고, 학습/검증/테스트 분할만 `fold`(config 기본값 `${seed}`)로 고른다. 스윕의 축이 fold이기
  때문이다(`scripts/sbatch/template.sh`). 즉석 랜덤 분할을 새 실행에 쓰지 않는다. 공유 split 파일이 있으면 repo에 커밋하고 그것을 기준으로 한다.
- `num_workers`, `pin_memory`, `batch_size`는 config 키로 둔다(테스트가 `num_workers=0`으로 덮어쓴다).
- `train_dataloader` / `val_dataloader` / `test_dataloader`를 구현한다(`eval.py`가 `test_dataloader()`를 쓴다). 배치 형식은 모델의 `forward(batch)`와
  맞춘다(toy는 `(x, y)`). 정답은 모델이 `ModelOutput.target`에 담아 loss/metric이 읽는다([src/models/README.md](../models/README.md)).

## 새 데이터셋 추가

1. `src/data/<name>_datamodule.py`와 `configs/data/<name>.yaml`을 만든다.
2. 실험 config에서 `override /data: <name>`로 고른다.
3. `tests/test_configs.py`가 모든 실험 config를 조립해 보므로 새 config도 자동으로 검증된다.
