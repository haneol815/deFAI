# defFAI — Federated Learning 과제

[Flower](https://flower.ai) 기반 연합학습(Federated Learning) 과제 템플릿입니다.
Flower Simulation Engine으로 한 머신에서 여러 클라이언트(SuperNode)를 시뮬레이션하며,
같은 코드를 Deployment Engine으로 실제 분산 환경에 배포할 수도 있습니다.

- 집계 전략: FedAvg, FedProx (Flower 내장 strategy)
- 데이터: MNIST, CIFAR-10 ([Flower Datasets](https://flower.ai/docs/datasets/)로 다운로드/분할)
- 분할: IID, Dirichlet non-IID
- 평가: 클라이언트 로컬 평가(federated) + 서버 전체 테스트셋 평가(centralized)

## 구조
```
.
├── pyproject.toml          # 의존성 + Flower 앱 설정 (run config)
├── deffai/
│   ├── server_app.py       # ServerApp: 전략 선택, 라운드 진행, 중앙 평가, 결과 저장
│   ├── client_app.py       # ClientApp: 로컬 학습 / 평가
│   └── task.py             # 모델, 데이터 분할/로딩, train/test 함수
├── scripts/plot.py         # 정확도 곡선 시각화
├── docs/report.md          # 보고서 템플릿
└── results/                # 실험 결과 (git 제외)
```

## 설치
Python 3.11 이상이 필요합니다.
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## 실행
```bash
# 기본 실행 (MNIST, IID, FedAvg, 10 rounds, SuperNode 10개)
flwr run . --stream --federation-config "num-supernodes=10"

# Non-IID (Dirichlet α=0.5) + FedProx
flwr run . --stream --federation-config "num-supernodes=10" \
  --run-config "partitioner='dirichlet' strategy='fedprox' proximal-mu=0.01"

# 라운드 수, 학습률 등 변경
flwr run . --stream --federation-config "num-supernodes=10" \
  --run-config "num-server-rounds=30 learning-rate=0.05"
```

- 설정 가능한 값은 `pyproject.toml`의 `[tool.flwr.app.config]`에 모두 있습니다.
- Flower 1.32부터 기본 SuperNode 수가 2개이므로 `--federation-config`로 지정합니다.
  매번 쓰기 번거로우면 기본값을 영구적으로 바꿀 수 있습니다.
  `flwr federation simulation-config --num-supernodes 10`
- 결과는 `results/<dataset>_<strategy>_<partitioner>_<timestamp>/history.json`에 저장됩니다.

```bash
pip install -e ".[plot]"
python scripts/plot.py results/*/history.json -o results/acc.png
```

## 확장 포인트
- **새 집계 알고리즘**: `flwr.serverapp.strategy.FedAvg`를 상속해 `aggregate_train()` 등을 오버라이드하고 `server_app.py`의 `build_strategy()`에 등록합니다.
  Flower에는 FedAdam, FedYogi, FedMedian, Krum 같은 전략도 내장돼 있습니다.
- **새 로컬 학습 방식**: `task.py`의 `train()`, `client_app.py`의 `train()`
- **새 데이터셋/모델**: `task.py`의 `DATASETS`, `build_model()`
- **실제 분산 배포**: [Deployment Engine 가이드](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html)

## License
[MIT](LICENSE)
