# 연합학습 과제 보고서

## 1. 개요
- 과제 목표:
- 사용 데이터셋 / 모델:

## 2. 실험 설정
| 항목 | 값 |
|---|---|
| Clients (전체 / 라운드당) | 10 / 5 |
| Rounds | 20 |
| Local epochs | 1 |
| Batch size / LR | 32 / 0.01 |
| Partition | IID / Dirichlet(α) |

## 3. 결과
<!-- python scripts/plot.py results/*/history.json -o docs/acc.png -->

## 4. 분석 및 고찰

## 5. 참고문헌
- McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data*, AISTATS 2017.
- Li et al., *Federated Optimization in Heterogeneous Networks* (FedProx), MLSys 2020.
