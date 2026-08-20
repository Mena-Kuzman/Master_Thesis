# SAC + MPC Inputs/Outputs (Isaac-only path)

This document describes exactly what the current implementation exchanges between RL and MPC, and how to run training yourself.

Important: in terminal, paste raw commands only. Do not paste Markdown links like `[python.sh](...)` or `[train_sac_step_height_residual.py](...)`.

## 1) Run training yourself (non-headless)

Use Isaac's Python launcher and clear `PYTHONPATH` to avoid package conflicts.

```bash
env -u PYTHONPATH /home/mena/isaacsim/python.sh -m pip install gymnasium torch tensorboard absl-py pygame
```

If `acados_template` is not installed in Isaac Python:

```bash
env -u PYTHONPATH /home/mena/isaacsim/python.sh -m pip install casadi
env -u PYTHONPATH /home/mena/isaacsim/python.sh -m pip install -e /home/mena/cpssim2/pympc/quadruped_pympc/acados/interfaces/acados_template
```

Run training **with rendering** (non-headless):

```bash
env -u PYTHONPATH /home/mena/isaacsim/python.sh /home/mena/cpssim2/IsaacLabFromTemu/train_sac_step_height_residual.py --render --total-timesteps 200000
```

Straight-walk speed curriculum + randomized stairs:

```bash
env -u PYTHONPATH /home/mena/isaacsim/python.sh /home/mena/cpssim2/IsaacLabFromTemu/train_sac_step_height_residual.py --render --total-timesteps 200000 --speed-curriculum-min 0.3 --speed-curriculum-max 0.8 --speed-curriculum-warmup-episodes 20 --speed-curriculum-full-episodes 300 --randomize-stairs --stair-start-x-min 1.5 --stair-start-x-max 3.0 --stair-step-height-min 0.08 --stair-step-height-max 0.14 --stair-step-width-min 0.24 --stair-step-width-max 0.36 --stair-num-steps-up-min 2 --stair-num-steps-up-max 5 --stair-num-steps-down-min 2 --stair-num-steps-down-max 5
```

Option B: just switch to a free TensorBoard port

```bash
env -u PYTHONPATH -u LD_LIBRARY_PATH -u LD_PRELOAD /tmp/tbvenv/bin/python -m tensorboard.main --logdir /home/mena/cpssim2/IsaacLabFromTemu/runs --port 6010 --host 127.0.0.1
```

Optional TensorBoard (separate terminal):

```bash
tensorboard --logdir /home/mena/cpssim2/IsaacLabFromTemu/runs
```

Stop training with `Ctrl+C` in the training terminal.

---

## 2) RL side (what SAC consumes/produces)

### Observation space
From `IsaacMPCController.get_rl_observation_vector()`: fixed `Box(shape=(25,), dtype=float32)`.

Indices:
- `0:3` -> commanded base linear velocity (`ref_base_lin_vel`)
- `3:6` -> actual base linear velocity (`linear_velocity`)
- `6:9` -> base angular velocity in body frame (`base_ang_vel_B`)
- `9:11` -> base roll/pitch (`base_ori_euler_xyz[0:2]`)
- `11` -> effective step height currently applied
- `12` -> current residual step-height value
- `13:17` -> current contact flags (FL, FR, RL, RR)
- `17:25` -> per-leg heightmap stats: `(valid_ratio, z_range)` for FL, FR, RL, RR

### Action space
From `IsaacStepHeightResidualEnv`: `Box(low=-1, high=1, shape=(1,), dtype=float32)`.

Policy output `a` is mapped to meters:
- `residual_m = a * rl_step_height_residual_max_abs`

### SAC training objective
Standard SAC (continuous control): maximize expected return + entropy term via actor/critic updates.

---

## 3) RL -> MPC outputs

RL writes exactly one control signal each step:
- `set_step_height_residual(residual_m)` in `IsaacMPCController`

Inside MPC controller:
1. residual is clamped to `[-rl_step_height_residual_max_abs, +rl_step_height_residual_max_abs]`
2. residual is smoothed (EMA) with `rl_step_height_residual_smoothing`
3. effective step height is computed:
   - `effective = nominal_step_height + residual`
   - clipped to `[rl_step_height_min, rl_step_height_max]`
4. applied to WB layer:
   - `self.wb_interface.step_height = effective`

This means MPC remains primary; RL is an additive residual assistant.

---

## 4) MPC -> RL outputs

Per env step, MPC/controller provide RL-observable state through:
- `get_rl_observation_vector()` (25D observation)
- `get_rl_state_snapshot()` (dict used for reward and termination)

Snapshot includes at least:
- `position`
- `base_ori_euler_xyz`
- `linear_velocity`
- `base_ang_vel_B`
- `ref_base_lin_vel`
- `ref_base_ang_vel`
- `current_contact`
- `step_height_effective`

---

## 5) Reward and goal

Current reward in `IsaacStepHeightResidualEnv._reward_and_done`:

- `tracking_error = |vx - command_forward|`
- `tracking_term = -tracking_error`
- `progress_term = +0.5 * vx`
- `attitude_penalty = -0.25 * (roll^2 + pitch^2)`
- `action_penalty = -0.05 * (residual_m / max_residual)^2`
- `stumble_penalty = -0.08 * early_stance_count`
- `reward = 1.0 + tracking_term + progress_term + attitude_penalty + action_penalty + stumble_penalty`

Termination:
- base too low: `z < 0.12`, or
- unstable attitude: `|roll| > 1.2` or `|pitch| > 1.2`

### Practical goal
Learn a residual step-height policy that improves obstacle/stair traversal while preserving velocity tracking and base stability.

---

## 6) Config knobs controlling RL-MPC coupling

In `pympc/quadruped_pympc/config.py`:
- `rl_step_height_enabled`
- `rl_step_height_residual_max_abs`
- `rl_step_height_residual_smoothing`
- `rl_step_height_min`
- `rl_step_height_max`

Use these to bound and stabilize residual behavior.

Note: in RL training (`IsaacStepHeightResidualEnv`), `rl_step_height_enabled` is forced to `True` at runtime, even if the global config default is `False`.
