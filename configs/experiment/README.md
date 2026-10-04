# configs/experiment/

실험 1개 = yaml 1개. 두 그룹이 있고 각각 Hydra config group이다.

```
experiment/
├── train/<topic>/<YYMMDD_round>/<YYMMDD-name>.yaml   # train.py / eval.py / analyze.py 공용 (experiment/train=...)
└── compare/<YYMMDD-name>.yaml                        # 여러 실험을 묶어 비교 (configs/compare.yaml, experiment/compare=...)
```

## train/ 계층

- `<topic>` — 모델 계열 또는 데이터셋(예: `toy`, `milk10k_scm_cascade`). 같은 `configs/model/*.yaml`을 쓰는 실험끼리
  묶는다. 스윕 스크립트 하나(`scripts/<topic>_train.sh`)가 한 topic을 담당한다 (CLAUDE.md §1).
- `<YYMMDD_round>` — 실험 라운드(한 질문을 확인하려고 묶어 돌린 실험 모음). 폴더 이름의 날짜는 라운드 시작일.
- `<YYMMDD-name>.yaml` — 파일 이름(확장자 제외)이 곧 `experiment_name`이다. 체크포인트는
  `logs/train/runs/<experiment_name>/checkpoints/`에 쌓이므로 **config 안에서 `experiment_name: <파일 이름>`으로 맞춘다.**
- 헤더 주석에 의도(어떤 질문), 비교군(어떤 config와 무엇이 다른지), 실행 명령을 쓴다.

## 선택 방법

```bash
python src/train.py experiment/train=toy/261004_example/261004-toy-example
```

## 새 실험 만들기

비교군 config를 복사해서 **바꾸는 키를 최소로**(한 arm에 변경 하나, CLAUDE.md §5.2) 하고, 헤더에 어느 config 대비 무엇이
다른지 적는다. 새 라운드는 `<YYMMDD_round>/` 폴더를 새로 만든다.
