# defFAI — Federated Learning 과제

PyTorch 기반의 간단한 연합학습(Federated Learning) 시뮬레이션 템플릿입니다.
단일 머신에서 여러 클라이언트를 시뮬레이션하며 FedAvg / FedProx를 지원합니다.

## 구조
```
.
├── main.py              # 실험 진입점
├── configs/default.yaml # 실험 설정
├── fl/
│   ├── data.py          # 데이터셋 로드 및 IID / Dirichlet non-IID 분할
│   ├── models.py        # CNN 모델
│   ├── client.py        # 로컬 학습 (FedAvg / FedProx)
│   ├── server.py        # 클라이언트 샘플링, 가중 평균 집계, 평가
│   └── utils.py         # seed, device, config 유틸
├── scripts/plot.py      # 정확도 곡선 시각화
├── docs/report.md       # 보고서 템플릿
└── results/             # 실험 결과 (git 제외)
```

## 설치
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 실행
```bash
# 기본 (MNIST, IID, FedAvg, 20 rounds)
python main.py

# Non-IID (Dirichlet α=0.5)
python main.py --partition dirichlet

# FedProx
python main.py --partition dirichlet --algorithm fedprox

# 다른 설정 파일 사용
python main.py --config configs/my_exp.yaml
```

결과는 `results/<dataset>_<algo>_<partition>_<timestamp>/history.json`에 저장됩니다.

```bash
pip install matplotlib
python scripts/plot.py results/*/history.json -o results/acc.png
```

## 확장 포인트
- 새 집계 알고리즘: `fl/server.py`의 `fedavg()` 옆에 추가 후 `Server.run_round()`에서 분기
- 새 로컬 학습 방식: `fl/client.py`의 `Client.train()`
- 새 데이터셋/모델: `fl/data.py`의 `_DATASETS`, `fl/models.py`의 `build_model()`

## License
[MIT](LICENSE)
