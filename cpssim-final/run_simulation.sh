#!/bin/bash
# Run Go2 MPC simulation with correct library paths

# Set library path to use conda's libstdc++ (fixes casadi CXXABI issue)
export LD_LIBRARY_PATH="/home/mena/miniconda3/envs/pympc/lib:$LD_LIBRARY_PATH"

# Change to Isaac Lab directory and run the simulation
cd /home/mena/cpssim2/IsaacLabFromTemu
python3 go2_pympc.py "$@"
