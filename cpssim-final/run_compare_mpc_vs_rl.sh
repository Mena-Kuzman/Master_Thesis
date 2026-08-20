#!/bin/bash
# Run MPC vs MPC+RL comparison with the same runtime library setup as run_simulation.sh

export LD_LIBRARY_PATH="/home/mena/miniconda3/envs/pympc/lib:$LD_LIBRARY_PATH"

cd /home/mena/cpssim3/cpssim/IsaacLabFromTemu

# Default to a known completed parallel trial checkpoint unless user provides --checkpoint.
DEFAULT_TRIAL_ID="1777120627"  #1777120627    1778755326
DEFAULT_CHECKPOINT="/home/mena/cpssim3/cpssim/checkpoints/sac_step_height_residual_parallel_seed1_${DEFAULT_TRIAL_ID}_final.pt"

has_checkpoint_arg=false
for arg in "$@"; do
	if [[ "$arg" == "--checkpoint" || "$arg" == --checkpoint=* ]]; then
		has_checkpoint_arg=true
		break
	fi
done

if [[ "$has_checkpoint_arg" == true ]]; then
	python3 compare_mpc_vs_mpc_rl.py "$@"
else
	if [[ ! -f "$DEFAULT_CHECKPOINT" ]]; then
		echo "[compare-launcher] Default checkpoint not found: $DEFAULT_CHECKPOINT" >&2
		echo "[compare-launcher] Pass --checkpoint <path> explicitly." >&2
		exit 1
	fi
	echo "[compare-launcher] Using default trial checkpoint: $DEFAULT_CHECKPOINT"
	python3 compare_mpc_vs_mpc_rl.py --checkpoint "$DEFAULT_CHECKPOINT" "$@"
fi
