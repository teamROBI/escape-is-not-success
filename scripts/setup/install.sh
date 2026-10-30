#!/usr/bin/env bash
# Installation only; this script does not launch simulation or training.
set -euo pipefail
release_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${release_root}"
: "${ISAACSIM_PATH:?Set ISAACSIM_PATH to your Isaac Sim 5.1 installation}"
python3.11 -m venv venvs/sim
python3.11 -m venv venvs/act
venvs/sim/bin/python -m pip install --upgrade pip
venvs/act/bin/python -m pip install --upgrade pip
ln -sfn "${ISAACSIM_PATH}" third_party/IsaacLab/_isaac_sim
for escape_ext in isaaclab isaaclab_assets; do
    venvs/sim/bin/python -m pip install -e "third_party/IsaacLab/source/${escape_ext}"
done
venvs/sim/bin/python -m pip install -r requirements-sim.txt
venvs/act/bin/python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
venvs/act/bin/python -m pip install -r requirements-act.txt
venvs/act/bin/python -m pip install -e third_party/act/detr
bash scripts/setup/patch_act_state_dim.sh
