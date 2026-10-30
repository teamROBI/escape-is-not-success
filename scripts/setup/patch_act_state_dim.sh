#!/usr/bin/env bash
# Apply the state/action dimension changes used by the paper, preserving all other ACT code.
set -euo pipefail
release_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
python3 - "${release_root}/third_party/act/detr/models/detr_vae.py" "${1:-}" <<'PY'
from pathlib import Path
import ast
import sys

path = Path(sys.argv[1])
s = path.read_text()
if 'ESCAPE_STATE_DIM = ' in s:
    ast.parse(s)
    print('ACT dimension patch already applied')
    raise SystemExit(0)
if sys.argv[2] == '--check':
    print('ACT dimension patch required')
    raise SystemExit(1)
changes = {
    'import torch': 'import os\nimport torch',
    'self.action_head = nn.Linear(hidden_dim, state_dim)': 'self.action_head = nn.Linear(hidden_dim, ESCAPE_ACTION_DIM)',
    'self.action_head = nn.Linear(1000, state_dim)': 'self.action_head = nn.Linear(1000, ESCAPE_ACTION_DIM)',
    'self.input_proj_robot_state = nn.Linear(14, hidden_dim)': 'self.input_proj_robot_state = nn.Linear(ESCAPE_STATE_DIM, hidden_dim)',
    'self.encoder_action_proj = nn.Linear(14, hidden_dim)': 'self.encoder_action_proj = nn.Linear(ESCAPE_ACTION_DIM, hidden_dim)',
    'self.encoder_joint_proj = nn.Linear(14, hidden_dim)': 'self.encoder_joint_proj = nn.Linear(ESCAPE_STATE_DIM, hidden_dim)',
    'output_dim=14': 'output_dim=ESCAPE_ACTION_DIM',
    'state_dim = 14 # TODO hardcode': 'state_dim = ESCAPE_STATE_DIM',
}
for before, after in changes.items():
    if before not in s:
        raise SystemExit(f'Unexpected ACT source: missing {before!r}; use the pinned submodule')
    s = s.replace(before, after)
banner = '''\n# Configurable measured-state and predicted-action widths.
ESCAPE_STATE_DIM = int(os.environ.get("ESCAPE_STATE_DIM", 8))
ESCAPE_ACTION_DIM = int(os.environ.get("ESCAPE_ACTION_DIM", ESCAPE_STATE_DIM))
'''
s = s.replace('import IPython\n', 'import IPython\n' + banner)
ast.parse(s)
path.write_text(s)
print('Applied independent state/action widths (default 8; EEF residual action 7)')
PY
