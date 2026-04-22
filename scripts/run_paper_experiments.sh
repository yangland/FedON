#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-all}"
GPU_ID="${GPU_ID:-0}"

DATASETS=(cifar10 svhn gtsrb)
AGGRS=(fedavg median krum flame fltrust nb rfa)

run_config() {
  local cfg="$1"
  local tmp_cfg
  tmp_cfg="$(mktemp /tmp/fedon_config_XXXXXX.yaml)"
  sed "s/^cuda_device: .*/cuda_device: ${GPU_ID}/" "$cfg" > "$tmp_cfg"
  echo "[FedON] Running $cfg on cuda:${GPU_ID}"
  python main_fl.py --params "$tmp_cfg"
  rm -f "$tmp_cfg"
}

case "$TARGET" in
  all)
    for dataset in "${DATASETS[@]}"; do
      for aggr in "${AGGRS[@]}"; do
        run_config "yamls/paper/${dataset}/${dataset}_series_${aggr}.yaml"
      done
    done
    ;;
  cifar10|svhn|gtsrb)
    for aggr in "${AGGRS[@]}"; do
      run_config "yamls/paper/${TARGET}/${TARGET}_series_${aggr}.yaml"
    done
    ;;
  *)
    echo "Usage: GPU_ID=0 bash scripts/run_paper_experiments.sh [all|cifar10|svhn|gtsrb]" >&2
    exit 1
    ;;
esac
