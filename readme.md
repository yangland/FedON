# FedON

Code for the ICASSP 2026 paper:

**FedON: Black-Box Untargeted Model Poisoning via Multi-Objective Reinforcement Learning**

## Setup

```bash
conda create -n fedon python=3.10
conda activate fedon
pip install -r requirements.txt
```

## Run

FedON and baseline attacks on CIFAR-10 with FedAvg:

```bash
python main_fl.py --params yamls/paper/cifar10/cifar10_series_fedavg.yaml
```

FedON and baseline attacks on SVHN with FLAME:

```bash
python main_fl.py --params yamls/paper/svhn/svhn_series_flame.yaml
```

FedON and baseline attacks on GTSRB with RFA:

```bash
python main_fl.py --params yamls/paper/gtsrb/gtsrb_series_rfa.yaml
```

Run all paper configs:

```bash
GPU_ID=0 bash scripts/run_paper_experiments.sh all
```

Run one dataset:

```bash
GPU_ID=0 bash scripts/run_paper_experiments.sh cifar10
```

Dataset configs: `cifar10`, `svhn`, `gtsrb`.

Aggregation configs: `fedavg`, `median`, `krum`, `flame`, `fltrust`, `nb`, `rfa`.

Logs and plots are written to `logs_fl/`.

## Citation

Citation information will be added after the proceedings entry is available.
