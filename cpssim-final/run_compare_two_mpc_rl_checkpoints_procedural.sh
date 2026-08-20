#!/bin/bash
# Run the procedural two-checkpoint MPC+RL comparison with the same runtime library setup as run_simulation.sh
export LD_LIBRARY_PATH="/home/mena/miniconda3/envs/pympc/lib:$LD_LIBRARY_PATH"

cd /home/mena/cpssim3/cpssim/IsaacLabFromTemu

DEFAULT_CHECKPOINT_A="/home/mena/cpssim3/cpssim/checkpoints/sac_step_height_residual_parallel_seed1_1778755326_final.pt"
DEFAULT_CHECKPOINT_B="/home/mena/cpssim3/cpssim/checkpoints/sac_step_height_residual_parallel_seed1_1777120627_final.pt"

has_checkpoint_a_arg=false
has_checkpoint_b_arg=false
for arg in "$@"; do
	if [[ "$arg" == "--checkpoint-a" || "$arg" == --checkpoint-a=* ]]; then
		has_checkpoint_a_arg=true
	fi
	if [[ "$arg" == "--checkpoint-b" || "$arg" == --checkpoint-b=* ]]; then
		has_checkpoint_b_arg=true
	fi
done

if [[ "$has_checkpoint_a_arg" == true && "$has_checkpoint_b_arg" == true ]]; then
	python3 compare_two_mpc_rl_checkpoints_procedural.py "$@"
elif [[ "$has_checkpoint_a_arg" == true ]]; then
	echo "[compare-launcher] Using default checkpoint B: $DEFAULT_CHECKPOINT_B"
	python3 compare_two_mpc_rl_checkpoints_procedural.py --checkpoint-b "$DEFAULT_CHECKPOINT_B" "$@"
elif [[ "$has_checkpoint_b_arg" == true ]]; then
	echo "[compare-launcher] Using default checkpoint A: $DEFAULT_CHECKPOINT_A"
	python3 compare_two_mpc_rl_checkpoints_procedural.py --checkpoint-a "$DEFAULT_CHECKPOINT_A" "$@"
else
	echo "[compare-launcher] Using default checkpoint A: $DEFAULT_CHECKPOINT_A"
	echo "[compare-launcher] Using default checkpoint B: $DEFAULT_CHECKPOINT_B"
	python3 compare_two_mpc_rl_checkpoints_procedural.py --checkpoint-a "$DEFAULT_CHECKPOINT_A" --checkpoint-b "$DEFAULT_CHECKPOINT_B" "$@"
fi

