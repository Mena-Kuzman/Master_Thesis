# RL-Augmented Model Predictive Control for Quadruped Locomotion

**Master's Thesis (Grade: 1.0) — Technical University of Munich (TUM)**  
*Chair of Cyber-Physical Systems in Production Engineering*  
**Author:** [Mena Kuzman](https://www.google.com/search?q=https://www.linkedin.com/in/mena-kuzman/) | **Platform:** Unitree Go2 | **Simulator:** NVIDIA Isaac Sim 4.5.0

This repository contains the control architecture, reinforcement learning training pipeline, and high-fidelity simulation environment for my Master's thesis at the Technical University of Munich. It implements a hierarchical locomotion control framework for the **Unitree Go2** quadruped robot, coupling a high-level **Soft Actor-Critic (SAC)** reinforcement learning policy in PyTorch with a low-level **Nonlinear Model Predictive Controller (NMPC)** and **Whole-Body Controller (WBC)** in CasADi / `acados` to achieve robust locomotion over complex stair terrains and under external payload disturbances.

## Performance Comparison: Standard NMPC vs. RL-Augmented NMPC

Adding the high-level RL policy enables the quadruped to dynamically adapt its step height and swing-phase timing in real time, overcoming stair obstacles where a standard fixed-gait model-based controller fails.

| Baseline (Standard SRBD NMPC) | RL-Augmented NMPC (Proposed Approach) |
| :---: | :---: |
|  |  |
| *Robot struggles with fixed gait timings and rigid foot-swing trajectories on stairs.* | *Robot dynamically adapts step height and swing duration via the PyTorch SAC policy.* |

## System Architecture

The hierarchical control framework separates terrain-adaptive decision-making from high-frequency dynamic execution:



### 1. High-Level Control: Soft Actor-Critic (SAC) Policy (`PyTorch`)
* **State Inputs:** Current robot base states (orientation, linear/angular velocity), foot contact states, and user velocity/yaw commands.
* **Adaptive Residual Outputs:** Continuously modulates the nominal **Step Height** and **Swing-Phase Timing** sent to the low-level gait and swing controllers to accommodate uneven and stepped terrain.

### 2. Low-Level Control: Model-Based Optimal Execution (`CasADi` / `acados`)
* **Adaptive Gait Sequencer:** Computes the contact schedule, phase variables, and nominal foothold targets from user velocity commands and RL timing adjustments.
* **Nonlinear MPC (Single Rigid Body Dynamics — SRBD):** Formulated in **CasADi** and compiled to fast C code via **`acados`**, optimizing Center-of-Mass (CoM) trajectories and computing optimal **Ground Reaction Forces (GRFs)** for stance legs under friction-cone constraints.
* **Swing Trajectory Controller:** Generates collision-free foot swing trajectories parameterized by the nominal footholds and the RL agent's adaptive step height.
* **Whole-Body Controller (WBC):** Combines the optimal stance GRFs and desired swing-leg accelerations into physical **Joint Torques** executed on the Unitree Go2 in **NVIDIA Isaac Sim 4.5.0**.

## Robustness & Payload Generalization Benchmarks

The closed-loop architecture was systematically evaluated in **NVIDIA Isaac Sim** across both **procedural** and **randomized stair geometries**, testing zero-shot robustness under external payload disturbances:

* **Stair Climb Validation:** Evaluated standard fixed-height NMPC vs. RL-augmented NMPC across varying riser heights (`cpssim-final/fixed_step_height_stairs_results/`, `cpssim-final/random_stairs/`).
* **Payload Stress Testing (3 kg, 5 kg, and 7 kg):** Validated trajectory tracking, CoM stability, and obstacle clearance on procedural and randomized stairs with additional **3 kg, 5 kg, and 7 kg** payloads (`cpssim-final/procedural_3kg/`, `cpssim-final/procedural_5Kg/`, `cpssim-final/procedural_7kg/`, `cpssim-final/random_stair_3kg/`, `cpssim-final/random_stairs_5kg/`, `cpssim-final/random_stairs_7kg/`).

## Repository Structure

```text
Master_Thesis/
├── BaselineMPC.gif                  # Standard NMPC stair-climbing baseline demo
├── MPC+RL.gif                       # RL-Augmented NMPC stair-climbing demo
├── Hybrid_arch.png                  # Hierarchical SAC + SRBD NMPC + WBC block diagram
├── README.md                        # Project documentation
└── cpssim-final/
    ├── rl_sac_planner/              # High-level PyTorch SAC policy & Isaac Sim Go2 interface
    │   ├── go2_pympc.py             # Closed-loop Unitree Go2 Isaac Sim + PyMPC integration
    │   ├── sac_actor.py             # Soft Actor-Critic policy network architecture
    │   ├── train_sac_step_height_residual_parallel.py  # Parallel residual SAC training
    │   ├── compare_mpc_vs_mpc_rl.py # Automated benchmark runner (NMPC vs. RL+NMPC)
    │   ├── checkpoints/             # Trained SAC policy weights
    │   └── analysis/                # Quantitative trajectory & torque evaluation scripts
    ├── pympc/                       # Low-level SRBD NMPC, Gait Sequencer, Swing & WBC stack
    │   ├── quadruped_pympc/         # Core NMPC (CasADi / acados) & WBC implementation
    │   ├── simulation/              # Simulation configuration and terrain wrappers
    │   └── ros2/                    # ROS 2 interface nodes and message definitions
    ├── run_simulation.sh            # Launch script for closed-loop Isaac Sim evaluation
    └── run_compare_mpc_vs_rl.sh     # Launch script for comparative stair/payload benchmarks
```

## Tech Stack & Dependencies

* **Simulation Environment:** NVIDIA Isaac Sim 4.5.0
* **Reinforcement Learning:** Python 3.10+, PyTorch (Soft Actor-Critic)
* **Optimal Control & Codegen:** CasADi, `acados` (C solver generation), NumPy, SciPy
* **Operating System:** Ubuntu 22.04 LTS (Linux)

## Installation & Quick-Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/Mena-Kuzman/Master_Thesis.git
cd Master_Thesis/cpssim-final
```

### 2. Install `pympc` & `acados` Solver Dependencies
Follow the setup scripts inside `pympc/installation/` to build `acados` and install the Python environment required for the SRBD NMPC and Whole-Body Controller:
```bash
cd pympc
pip install -e .
cd ..
```

### 3. Run Closed-Loop Simulation in NVIDIA Isaac Sim
Ensure NVIDIA Isaac Sim 4.5.0 is installed and configured in your environment, then launch the closed-loop Unitree Go2 simulation:
```bash
./run_simulation.sh
```

### 4. Run Comparative Benchmarks (Standard NMPC vs. RL-Augmented NMPC)
To reproduce the stair-climbing and payload robustness comparisons (3 kg, 5 kg, 7 kg):
```bash
./run_compare_mpc_vs_rl.sh
```

### 5. Train the High-Level SAC Residual Planner
To train the Soft Actor-Critic policy from scratch in parallel simulation environments:
```bash
python3 rl_sac_planner/train_sac_step_height_residual_parallel.py
```
