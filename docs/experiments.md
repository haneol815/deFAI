# Experiments

Results of runs made with this repository's code, to show what the implemented algorithms do on real data and how
much of it is noise. Every table is produced by `scripts/run_grid.py` (which calls `scripts/compare.py`); the exact
commands are under each section. Read the caveats first.

## Setup and caveats

| | |
|---|---|
| Machine | 4 CPU cores (Intel Xeon 2.1 GHz), no GPU; Python 3.13, torch 2.14, Flower 1.42 |
| Model | the FedAvg-paper CNN of `defai/task.py` (1.66 M parameters) |
| Data | Fashion-MNIST (`dataset='fashion-mnist-mirror'`, see the README), 60 000 training images split over the clients; 10 000 test images for the `test_acc` of the global model |
| Federation | 10 clients, 5 sampled per round, 1 local epoch, batch size 32, client learning rate 0.01 |
| Seeds | 1 and 2. The seed changes the model initialisation, the data split, the client sampling and the data order |

- **Two seeds are not enough to rank algorithms.** The `±` is the standard deviation over two runs, not a confidence
  interval. Only differences that are large compared to it (and consistent in both seeds) are worth reading.
- **Nothing was tuned.** FedProx μ, MOON μ and q-FedAvg q are the defaults (or the values listed); the learning rate
  is the template's 0.01 for every algorithm. A tuned comparison could rank them differently.
- **The runs are short** (12 rounds) and not converged. The final-round accuracy of a non-IID run jumps by several
  points from round to round, so the tables also show the mean over the last three rounds.
- The accuracy columns are the *global* model on the central test set. `best acc` is the maximum over the rounds,
  i.e. selected on the test set: it is optimistic and only listed for reference.
- `eval_acc_min` is the accuracy of the worst client among the (five) clients evaluated in the last round; it varies
  with which clients were sampled.

## E1: non-IID data (Dirichlet α = 0.1)

Labels are split with `DirichletPartitioner(alpha=0.1)`: a client mostly holds two or three classes
(`python scripts/partition_stats.py --dataset fashion-mnist-mirror --partitioner dirichlet --alpha 0.1 --num-partitions 10 --seed 1`
gives a mean largest-class share of 0.61, uniform would be 0.10) and the clients hold between 755 and 11 442 examples.

```bash
python scripts/run_grid.py --output-dir results/fmnist_noniid --seeds 1 2 --num-supernodes 10 \
  --base "dataset='fashion-mnist-mirror' partitioner='dirichlet' dirichlet-alpha=0.1 num-server-rounds=12" \
  --vary "strategy='fedavg'" "strategy='fedprox' proximal-mu=0.01" "strategy='fedprox' proximal-mu=0.1" \
         "strategy='scaffold'" "strategy='moon'" "strategy='fednova'" "strategy='qfedavg'"
python scripts/compare.py results/fmnist_noniid/*/history.json --tail 3 --extra eval_acc_min comm-mb-total --target 0.6 --markdown
```

| run                                                     | runs | final acc     | last-3 mean   | best acc      | rounds to 0.6 | eval_acc_min  | comm-mb-total    |
| ------------------------------------------------------- | ---- | ------------- | ------------- | ------------- | ------------- | ------------- | ---------------- |
| fedavg                   | 2    | 0.618 ± 0.028 | 0.618 ± 0.004 | 0.648 ± 0.001 | 10.0 (2/2)    | 0.288 ± 0.188 | 761.548 ± 0.000  |
| fednova                  | 2    | 0.587 ± 0.032 | 0.608 ± 0.017 | 0.653 ± 0.002 | 4.5 (2/2)     | 0.171 ± 0.049 | 761.548 ± 0.000  |
| fedprox proximal-mu=0.01 | 2    | 0.619 ± 0.027 | 0.619 ± 0.004 | 0.648 ± 0.000 | 9.5 (2/2)     | 0.288 ± 0.187 | 761.548 ± 0.000  |
| fedprox proximal-mu=0.1  | 2    | 0.621 ± 0.020 | 0.619 ± 0.003 | 0.648 ± 0.008 | 9.5 (2/2)     | 0.293 ± 0.191 | 761.548 ± 0.000  |
| moon                     | 2    | 0.568 ± 0.055 | 0.596 ± 0.026 | 0.631 ± 0.012 | 6.5 (2/2)     | 0.223 ± 0.093 | 761.548 ± 0.000  |
| qfedavg                  | 2    | 0.490 ± 0.054 | 0.499 ± 0.033 | 0.514 ± 0.021 | -             | 0.052 ± 0.065 | 761.548 ± 0.000  |
| scaffold                 | 2    | 0.648 ± 0.034 | 0.626 ± 0.060 | 0.680 ± 0.007 | 6.5 (2/2)     | 0.258 ± 0.091 | 1523.095 ± 0.000 |

(`rounds to 0.6` is the first round whose accuracy reached 0.6, averaged over the runs that did; `comm-mb-total` is the
training traffic of the whole run in MiB.)

What can and cannot be concluded:

- **FedAvg, FedProx, FedNova, MOON and SCAFFOLD are not separated by these runs.** Their last-3 means lie between
  0.60 and 0.63 and the spread over the two seeds is up to 0.06. SCAFFOLD has the highest `best acc` (0.68), and in
  seed 1 it ends clearly ahead (0.67 against 0.60 for FedAvg), but in seed 2 it ends behind (0.62 against 0.64).
- **FedProx is indistinguishable from FedAvg** for μ = 0.01 and μ = 0.1: the accuracy curves differ by at most 0.004
  (μ = 0.01) and 0.03 (μ = 0.1) in any round, less than the round-to-round jumps. With a learning rate of 0.01 and a
  model that moves little within one epoch, the proximal pull is tiny at these values; larger μ was not tried.
- **q-FedAvg (q = 0.1) is clearly worse** here: 0.50 against 0.62 for FedAvg, in both seeds, and its curve is still
  climbing at round 12. Its update divides by `q·F^(q-1)·‖Δ‖² + L·F^q` with `L = 1/lr`; with lr = 0.01 that
  probably shrinks the step a lot (not verified). It was meant to help the *worst* clients, and it does not here.
- **Traffic is identical for all algorithms except SCAFFOLD**, which sends and receives a control variate next to the
  model: 1523 MiB against 762 MiB.

Caveat for this table: it was produced before commit `6b49a2e` ("Shuffle the mirrored training split before
partitioning"). That fix changes which images a client holds, not how many of each class (checked with
`partition_stats.py`: the class counts are identical), and it was not re-run. Seed 3 of the grid was not run, to save time.
