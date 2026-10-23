#!/usr/bin/env bash
set -eo pipefail
source /etc/profile.d/x2.sh
set -u

if [[ $# != 1 ]]; then
    echo "Usage: x2-component sim|mc" >&2
    exit 2
fi

# The official MC launcher selects REAL mode when this variable is present.
unset AGIBOT_SOFTWARE_HOME

case "$1" in
    sim)
        cd /workspace/sim_mujoco/bin
        # Replace this shell so the official script's pgrep cannot match it.
        exec ./start_sim.sh -s
        ;;
    mc)
        cd /workspace/mc/bin
        model_dir=mc_param/src/robot/lx2501_3_t2d5
        dump="$model_dir/mc_sim.yaml.dump"
        if [[ ! -f "$model_dir/mc_sim.yaml" || -L "$dump" ]]; then
            echo "Expected the official T2.5 simulation config and a regular generated dump." >&2
            exit 1
        fi
        # On the tested Mac bind mount, MC created this generated dump with
        # mode 0200 and then failed to copy into it. Leave the source YAML alone.
        touch -- "$dump"
        chmod u+rw -- "$dump"
        exec ./em_run.sh
        ;;
    *)
        echo "Usage: x2-component sim|mc" >&2
        exit 2
        ;;
esac
