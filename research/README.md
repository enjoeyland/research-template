# research/

다른 사람과 논의하거나 논문을 쓰기 위해 **정리한 글**을 두는 곳이다. `docs/`는 실험·수정 과정의 기록이고, 여기는 그 근거 위에 세운 서사다.

## 구조: 주제마다 폴더 하나

```
research/
├── README.md                              이 문서 + 아래 주제 목록
├── _TEMPLATE/YYMMDD_topic.md              새 주제의 글 템플릿 (폴더째 복사해서 시작)
├── YYMMDD_<주제>/
│   ├── YYMMDD_<주제>.md                   글로 정리하는 문서 (주장 ↔ 근거, 논의할 것, 결정)
│   ├── YYMMDD_<주제>.pptx                 그것을 시각적으로 만든 발표 자료
│   └── figures/                           .md에 쓰는 그림
└── paper/                                 논문 원고 (별도 repo, gitignore)
```

- **파일 이름에 주제를 넣는다**: `.md`와 `.pptx`는 따로 내려받아 보내기도 하므로, 이름만 봐도 무슨 글인지 알아야 한다. 날짜는 그 주제를 시작한 날이다(`<YYMMDD_topic>` 규칙).
- **그림은 원본을 복사해서 고정**한다(`docs/figures/`나 `logs/runs/<exp>/analyze/`에서). 논의한 시점의 결과가 나중에 바뀌어도 이 글의 그림은 그대로여야 한다.
  그림 아래에 어느 코드/실험에서 나온 것인지 한 줄 출처를 적는다. 파일 이름은 `fig<번호>_<이름>.png`.
- **글은 단독으로 읽혀야 한다**: repo 안 링크(`docs/...`)는 파일만 내려받으면 깨진다. 근거는 **값과 출처 경로를 본문에 직접 적고**, 링크는 덧붙이는 용도로만 쓴다.
- `.pptx`는 이진 파일이라 diff는 안 되지만 논의용 산출물이므로 git에 올린다(크면 PDF로 내보내 함께 둔다).
- `paper/`는 Overleaf 같은 별도 repo라서 이 repo에서는 gitignore한다(`.gitignore`의 `research/paper/`).

## 새 주제 시작

```bash
cp -r research/_TEMPLATE research/261006_my-topic
mv research/261006_my-topic/YYMMDD_topic.md research/261006_my-topic/261006_my-topic.md
# 발표 자료는 같은 이름으로: research/261006_my-topic/261006_my-topic.pptx
```

## 주제 목록

| 주제 | 상태 | 한 줄 |
|---|---|---|
