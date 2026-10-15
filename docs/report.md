# 연합학습 과제 보고서

## 1. 개요
- 과제 목표:
- 사용 데이터셋 / 모델:
- 프레임워크: Flower

## 2. 실험 설정
| 항목 | 값 |
|---|---|
| SuperNodes (전체 / 라운드당 비율) | 10 / 0.5 |
| Rounds | 10 |
| Local epochs | 1 |
| Batch size / LR | 32 / 0.01 |
| Strategy | FedAvg / FedProx (μ) |
| Partition | IID / Dirichlet(α) |

## 3. 결과
<!-- python scripts/plot.py results/*/history.json -o docs/acc.png -->

## 4. 분석 및 고찰

## 5. 참고문헌
- McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data*, AISTATS 2017.
- Li et al., *Federated Optimization in Heterogeneous Networks* (FedProx), MLSys 2020.
- Beutel et al., *Flower: A Friendly Federated Learning Research Framework*, arXiv:2007.14390, 2020.
