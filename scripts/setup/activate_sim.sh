#!/usr/bin/env bash
# Source after installing Isaac Sim 5.1 and this release's sim environment.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo 'Source this file: source scripts/setup/activate_sim.sh' >&2
    exit 1
fi
escape_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${ISAACSIM_PATH:?Set ISAACSIM_PATH to your Isaac Sim 5.1 installation}"
source "${escape_root}/venvs/sim/bin/activate"
source "${ISAACSIM_PATH}/setup_conda_env.sh"
export PYTHONPATH="${escape_root}${PYTHONPATH:+:${PYTHONPATH}}"
# The USD extension is also needed by the procedural asset builders.
for escape_ext in "${ISAACSIM_PATH}"/extscache/omni.usd.libs-*; do
    [[ -d "${escape_ext}/pxr" ]] || continue
    export PYTHONPATH="${escape_ext}:${PYTHONPATH}"
    export LD_LIBRARY_PATH="${escape_ext}/bin:${escape_ext}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    break
done
unset escape_ext escape_root
