# Installation

Run setup commands from the repository root.

Use Linux, an NVIDIA GPU and an installed Isaac Sim **5.1** distribution. Initialize the two pinned dependencies:

```bash
git submodule update --init third_party/IsaacLab third_party/act
export ISAACSIM_PATH=/path/to/isaac-sim-5.1
bash scripts/setup/install.sh
source scripts/setup/activate_sim.sh
```

Replace the installation path above with your own. The setup creates separate Python 3.11 environments, `venvs/sim` for Isaac Lab, collection, conversion and reporting, and `venvs/act` for ACT training and serving. Isaac Lab is installed from its pinned submodule. ACT's dimension patch is applied by [patch_act_state_dim.sh](../scripts/setup/patch_act_state_dim.sh); it supports the 8-dimensional measured joint state and either 7- or 8-dimensional actions. The environment variables `ESCAPE_STATE_DIM` and `ESCAPE_ACTION_DIM` must agree with each checkpoint; the recipe driver sets them. Install FFmpeg if you want H.264 rollout videos. ACT downloads ImageNet backbone weights on its first training run.

Activate the ACT environment in training and policy-server terminals:

```bash
source venvs/act/bin/activate
```

Set `CUDA_VISIBLE_DEVICES` in each terminal before GPU work; select devices available on your machine.

## Wrist-camera mounts

Collection A supplies the interface and anchoring comparisons and head-only zeroed-state control. Its wrist lens is inside the hand and produces a near-uniform dark image; set `ESCAPE_WRIST_LEGACY=1` for both collection and evaluation. Collection B supplies all camera comparisons and the wrist-only zeroed-state control, with the wrist camera beside the fingers; set `ESCAPE_WRIST_LEGACY=0`. The reproduction driver selects the matching mount. Head images are 672×376 and wrist images 424×240. Images are halved; when both are used, the wrist image is first resized to match the head image.

## Verification

Run `python3 scripts/run_checks.py` for the standard-library tests, and `venvs/sim/bin/python scripts/run_checks.py --numpy` to add numerical/archive tests. `scripts/verify_source_parity.py --source /path/to/reference` compares grid, splits, envelope distances and deterministic reset seeds with a read-only reference checkout without importing Isaac. `python3 scripts/bench/check_self_contained.py` checks the tracked release files for private paths and credentials.

Continue with [the reproduction guide](REPRODUCE.md).
