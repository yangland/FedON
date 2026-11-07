# FedON

Code for the paper published at **ICASSP 2026**:

**FedON: Black-Box Untargeted Model Poisoning via Multi-Objective Reinforcement Learning**

[IEEE Xplore](https://doi.org/10.1109/ICASSP55912.2026.11463752)


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

If you find this work useful, please cite:

```bibtex
@inproceedings{li2026fedon,
  title     = {FedON: Black-Box Untargeted Model Poisoning via Multi-Objective Reinforcement Learning},
  author    = {Li, Yang and Kong, Adams Wai Kin},
  booktitle = {2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2026},
  publisher = {IEEE},
  doi       = {10.1109/ICASSP55912.2026.11463752}
}
```
