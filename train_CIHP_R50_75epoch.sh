#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

export NGPUS="${NGPUS:-1}"
python tools/check_environment.py
CUDA_VISIBLE_DEVICES="${GPU_DEVICES:-0}" python -m torch.distributed.launch --nproc_per_node="$NGPUS" tools/train_net.py \
		--config-file "configs/CIHP/CIHP_AIParsing_R50_75epoch_serialGE_parsingiou_iouloss.yaml"
