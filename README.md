# RL-Augmented Model Predictive Control for Quadruped Locomotion

This repository contains the code and simulation environment for my Master's thesis at the Technical University of Munich (TUM). It implements a hierarchical control architecture for a quadruped robot, combining a high-level **Soft Actor-Critic (SAC)** reinforcement learning agent with a low-level **Nonlinear Model Predictive Controller (NMPC)** to achieve robust locomotion over complex terrain.

## 🚀 Performance Comparison: Standard MPC vs. RL-Augmented MPC

The addition of the RL agent allows the robot to dynamically adapt its step height and swing time to overcome obstacles that a standard model-based controller fails to navigate.

| Baseline (Standard NMPC) | RL-Augmented MPC (Our Approach) |
| :---: | :---: |
| <img src="BaselineMPC.gif" width="400" alt="Baseline Standard NMPC" /> | <img src="MPC+RL.gif" width="400" alt="RL Augmented MPC" /> |
| *Robot struggles with fixed gait timings and rigid trajectory planning on stairs.* | *Robot successfully adapts step height and swing time dynamically via SAC policy.* |

---

## 🧠 System Architecture

The control framework is split into two distinct loops: a **High-Level RL Agent** running at a lower frequency for adaptive planning, and a **Low-Level Model-Based Controller** running at high frequency for precise dynamic execution. The entire system is validated using **NVIDIA Isaac Sim**.

<img src="Hybrid_arch.png" width="800" alt="Control Architecture Diagram" />

### 1. High-Level Control (Reinforcement Learning)
The high-level policy is trained using a **Soft Actor-Critic (SAC)** algorithm via PyTorch. 
* **Inputs:** Current robot states, foot contacts, and user velocity/yaw commands.
* **Outputs:** It continuously outputs an **Adaptive Step Height** and **Swing Time** adjustments to handle uneven terrain.

### 2. Low-Level Control (Model-Based Execution)
The low-level controller processes the RL agent's high-level adjustments and translates them into physical joint torques:
* **Adaptive Gait Sequencer:** Generates the baseline contact schedule, phase timing, and nominal footholds based on user commands.
* **Nonlinear MPC (SRBD):** Utilizes Single Rigid Body Dynamics to compute the **Optimal Ground Reaction Forces (GRFs)** for the stance legs, optimizing the Center of Mass (CoM) trajectory.
* **Swing Trajectory Controller:** Calculates the desired swing trajectories based on the nominal footholds and the RL agent's adaptive step height.
* **Whole-Body Controller (WBC):** Fuses the optimal GRFs and swing trajectories to compute the final **Joint Torques**, which are sent to the robot in NVIDIA Isaac Sim.

---

## 🛠️ Tech Stack & Dependencies

* **Simulation:** NVIDIA Isaac Sim
* **Machine Learning:** Python, PyTorch (SAC Implementation)
* **Optimal Control:** CasADi (Nonlinear MPC formulation)
* **OS:** Ubuntu 22.04 LTS (Linux)

---

## 💻 Installation & Quick Start

### 1. Clone the repository
```bash
git clone [https://github.com/Mena-Kuzman/rl-augmented-mpc-quadruped.git](https://github.com/Mena-Kuzman/rl-augmented-mpc-quadruped.git)
cd rl-augmented-mpc-quadruped