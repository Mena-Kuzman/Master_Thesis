from __future__ import annotations

import faulthandler
import os
import queue
import random
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

# Ensure LD_LIBRARY_PATH includes ACADOS/Isaac libs before any imports
# This is critical for parallel actors and the IsaacLab terrain generator.
_REQUIRED_LIB_DIR = "/home/mena/miniconda3/envs/pympc/lib"
_ld = os.environ.get("LD_LIBRARY_PATH", "")
_ld_parts = [p for p in _ld.split(":") if p]
if _REQUIRED_LIB_DIR not in _ld_parts:
    os.environ["LD_LIBRARY_PATH"] = ":".join([_REQUIRED_LIB_DIR] + _ld_parts)

import multiprocessing as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dog.isaac_step_height_residual_env import IsaacStepHeightResidualEnv
from src.rl_config import RL_PARALLEL_DEFAULTS, RLTrainArgs as BaseArgs, build_env_kwargs


LOG_STD_MAX = 2
LOG_STD_MIN = -5
HIDDEN_SIZE = 256


class _NullWriter:
    def add_scalar(self, *args, **kwargs):
        pass

    def close(self):
        pass


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, capacity: int, device: torch.device):
        self.capacity = int(capacity)
        self.device = device
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "observations": torch.tensor(self.obs[idx], device=self.device),
            "next_observations": torch.tensor(self.next_obs[idx], device=self.device),
            "actions": torch.tensor(self.actions[idx], device=self.device),
            "rewards": torch.tensor(self.rewards[idx], device=self.device),
            "dones": torch.tensor(self.dones[idx], device=self.device),
        }


class SoftQNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + act_dim, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc3 = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_low: np.ndarray, act_high: np.ndarray):
        super().__init__()
        act_dim = int(np.prod(act_low.shape))
        self.fc1 = nn.Linear(obs_dim, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc_mean = nn.Linear(HIDDEN_SIZE, act_dim)
        self.fc_logstd = nn.Linear(HIDDEN_SIZE, act_dim)

        self.register_buffer("action_scale", torch.tensor((act_high - act_low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((act_high + act_low) / 2.0, dtype=torch.float32))

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, obs):
        mean, log_std = self(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action


def _build_env_kwargs(base: BaseArgs) -> dict[str, Any]:
    # Keep Isaac rendering headless but allow pygame control windows.
    return build_env_kwargs(base, show_pygame_ui=True)


def _to_cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in module.state_dict().items()}


@dataclass
class ParallelArgs:
    base: BaseArgs = field(default_factory=BaseArgs)

    # Parallel actor-learner settings
    num_actors: int = RL_PARALLEL_DEFAULTS.num_actors
    num_envs_per_actor: int = RL_PARALLEL_DEFAULTS.num_envs_per_actor
    rollout_fragment_steps: int = RL_PARALLEL_DEFAULTS.rollout_fragment_steps
    learner_updates_per_batch: int = RL_PARALLEL_DEFAULTS.learner_updates_per_batch
    policy_broadcast_interval_updates: int = RL_PARALLEL_DEFAULTS.policy_broadcast_interval_updates
    transition_queue_size: int = RL_PARALLEL_DEFAULTS.transition_queue_size
    actor_seed_stride: int = RL_PARALLEL_DEFAULTS.actor_seed_stride
    actor_start_stagger_sec: float = RL_PARALLEL_DEFAULTS.actor_start_stagger_sec

    # Logging / run naming
    run_name: str = RL_PARALLEL_DEFAULTS.run_name


def _replay_add_batch(rb: ReplayBuffer, obs_np, next_obs_np, act_np, rew_np, done_np) -> int:
    """Insert a whole actor payload into replay with vectorized writes.

    This avoids Python per-transition loops on the learner side, which directly
    improves SPS when actor payloads are large.
    """
    n = int(obs_np.shape[0])
    if n <= 0:
        return 0

    cap = int(rb.capacity)
    ptr = int(rb.ptr)
    first = min(n, cap - ptr)

    rb.obs[ptr:ptr + first] = obs_np[:first]
    rb.next_obs[ptr:ptr + first] = next_obs_np[:first]
    rb.actions[ptr:ptr + first] = act_np[:first]
    rb.rewards[ptr:ptr + first] = rew_np[:first]
    rb.dones[ptr:ptr + first] = done_np[:first]

    if n > first:
        rem = n - first
        rb.obs[:rem] = obs_np[first:first + rem]
        rb.next_obs[:rem] = next_obs_np[first:first + rem]
        rb.actions[:rem] = act_np[first:first + rem]
        rb.rewards[:rem] = rew_np[first:first + rem]
        rb.dones[:rem] = done_np[first:first + rem]

    rb.ptr = (ptr + n) % cap
    rb.size = min(cap, int(rb.size) + n)
    return n


def _run_actor(
    actor_id: int,
    base_args_dict: dict[str, Any],
    env_kwargs: dict[str, Any],
    num_envs_per_actor: int,
    rollout_fragment_steps: int,
    random_explore_steps_per_actor: int,
    actor_seed_stride: int,
    transition_queue: mp.Queue,
    metric_queue: mp.Queue,
    weights_queue: mp.Queue,
    shared_curriculum_stage: Any,
    external_curriculum_control: bool,
    stop_event: Any,
) -> None:
    # Ensure actor process has correct LD_LIBRARY_PATH for ACADOS/Isaac libs
    _REQUIRED_LIB_DIR = "/home/mena/miniconda3/envs/pympc/lib"
    _ld = os.environ.get("LD_LIBRARY_PATH", "")
    _ld_parts = [p for p in _ld.split(":") if p]
    if _REQUIRED_LIB_DIR not in _ld_parts:
        os.environ["LD_LIBRARY_PATH"] = ":".join([_REQUIRED_LIB_DIR] + _ld_parts)

    print(f"[{os.getpid()}] Actor {actor_id} started.", flush=True)

    progress_path = f"/tmp/actor_{actor_id}_progress.log"

    def _mark(stage: str) -> None:
        try:
            with open(progress_path, "a") as progress_file:
                progress_file.write(f"{stage}\n")
                progress_file.flush()
        except Exception:
            pass

    def _write_trace(path: str) -> None:
        import traceback

        try:
            with open(path, "w") as trace_file:
                trace_file.write(traceback.format_exc())
        except Exception:
            pass
        try:
            metric_queue.put_nowait({"actor_id": actor_id, "actor_exception": traceback.format_exc()})
        except Exception:
            pass

    _mark("entry")
    _mark("before_base")
    base = BaseArgs(**base_args_dict)
    _mark("after_base")

    faulthandler_file = None
    env = None
    try:
        try:
            faulthandler_file = open(f"/tmp/actor_{actor_id}_faulthandler.log", "w")
            faulthandler.enable(file=faulthandler_file, all_threads=True)
        except Exception:
            faulthandler_file = None

        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        random.seed(base.seed + actor_seed_stride * actor_id)
        np.random.seed(base.seed + actor_seed_stride * actor_id)
        torch.manual_seed(base.seed + actor_seed_stride * actor_id)

        _mark("before_make_env")
        env = IsaacStepHeightResidualEnv.make_env(num_envs=num_envs_per_actor, **env_kwargs)
        _mark("after_make_env")
        is_vector_env = hasattr(env, "single_observation_space") and hasattr(env, "single_action_space")

        def _sync_global_curriculum_stage() -> None:
            if (not external_curriculum_control) or is_vector_env:
                return
            if hasattr(env, "set_curriculum_stage"):
                try:
                    env.set_curriculum_stage(int(shared_curriculum_stage.value))
                except Exception:
                    pass

        if is_vector_env:
            obs_space = env.single_observation_space
            act_space = env.single_action_space
        else:
            if int(num_envs_per_actor) != 1:
                raise RuntimeError("single-env actor path requires num_envs_per_actor == 1")
            obs_space = env.observation_space
            act_space = env.action_space

        obs_dim = int(np.prod(obs_space.shape))
        act_dim = int(np.prod(act_space.shape))
        actor = Actor(obs_dim, act_space.low, act_space.high).cpu()
        actor.eval()

        _mark("before_first_weights")
        first_weights = weights_queue.get()
        _mark("after_first_weights")
        actor.load_state_dict(first_weights)

        _mark("before_reset")
        _sync_global_curriculum_stage()
        obs, _ = env.reset(seed=base.seed + actor_seed_stride * actor_id)
        _mark("after_reset")
        if not is_vector_env:
            obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        local_env_steps = 0
        episodic_return = np.zeros((num_envs_per_actor,), dtype=np.float64)
        episodic_length = np.zeros((num_envs_per_actor,), dtype=np.int32)

        def _extract_env_info_for_metric(info_obj: Any, env_idx: int) -> dict[str, Any]:
            if isinstance(info_obj, dict):
                if env_idx != 0:
                    return {}
                return dict(info_obj)
            if not isinstance(info_obj, (list, tuple)):
                return {}
            if env_idx >= len(info_obj):
                return {}
            env_info = info_obj[env_idx]
            if not isinstance(env_info, dict):
                return {}

            out: dict[str, Any] = {}
            final_info = env_info.get("final_info")
            if isinstance(final_info, dict):
                out.update(final_info)
            for k, v in env_info.items():
                if k in {"final_info", "final_observation", "reset_info"}:
                    continue
                if isinstance(v, (int, float, np.floating, np.integer, bool, str)):
                    out[k] = v
            return out

        while not stop_event.is_set():
            rollout_t0 = time.perf_counter()
            while True:
                try:
                    newer = weights_queue.get_nowait()
                    actor.load_state_dict(newer)
                except queue.Empty:
                    break

            obs_batch = []
            act_batch = []
            rew_batch = []
            done_batch = []
            next_obs_batch = []

            for _ in range(rollout_fragment_steps):
                _sync_global_curriculum_stage()
                if local_env_steps == 0:
                    _mark("before_first_action")
                if local_env_steps < random_explore_steps_per_actor:
                    action = np.random.uniform(-1.0, 1.0, size=(num_envs_per_actor, act_dim)).astype(np.float32)
                else:
                    with torch.no_grad():
                        obs_t = torch.tensor(obs, dtype=torch.float32)
                        a, _, _ = actor.get_action(obs_t)
                    action = a.cpu().numpy().astype(np.float32)
                if local_env_steps == 0:
                    _mark("after_first_action")

                if local_env_steps == 0:
                    _mark("before_first_step")
                if is_vector_env:
                    next_obs, reward, terminated, truncated, info = env.step(action)
                else:
                    next_obs_s, reward_s, terminated_s, truncated_s, info_s = env.step(action[0])
                    next_obs = np.asarray(next_obs_s, dtype=np.float32).reshape(1, -1)
                    reward = np.asarray([reward_s], dtype=np.float32)
                    terminated = np.asarray([terminated_s], dtype=np.bool_)
                    truncated = np.asarray([truncated_s], dtype=np.bool_)
                    info = [info_s]
                done = np.logical_or(terminated, truncated)
                if local_env_steps == 0:
                    _mark("after_first_step")

                if (not is_vector_env) and bool(done[0]):
                    _sync_global_curriculum_stage()
                    reset_obs_s, reset_info_s = env.reset()
                    reset_obs = np.asarray(reset_obs_s, dtype=np.float32).reshape(1, -1)
                    env_info = info[0] if isinstance(info, list) and len(info) > 0 and isinstance(info[0], dict) else {}
                    info = [{
                        **env_info,
                        "final_observation": next_obs[0].copy(),
                        "final_info": dict(env_info),
                        "reset_info": reset_info_s if isinstance(reset_info_s, dict) else {},
                    }]
                    next_obs = reset_obs

                obs_batch.append(obs.copy())
                act_batch.append(action.copy())
                rew_batch.append(np.asarray(reward, dtype=np.float32).reshape(-1, 1))
                done_batch.append(done.astype(np.float32).reshape(-1, 1))
                next_obs_batch.append(next_obs.copy())

                episodic_return += np.asarray(reward, dtype=np.float64)
                episodic_length += 1
                for env_i in range(num_envs_per_actor):
                    if bool(done[env_i]):
                        metric_payload = {
                            "actor_id": actor_id,
                            "episodic_return": float(episodic_return[env_i]),
                            "episodic_length": int(episodic_length[env_i]),
                        }
                        metric_payload.update(_extract_env_info_for_metric(info, env_i))
                        try:
                            metric_queue.put_nowait(metric_payload)
                        except queue.Full:
                            pass
                        episodic_return[env_i] = 0.0
                        episodic_length[env_i] = 0

                obs = next_obs
                local_env_steps += num_envs_per_actor

            payload = {
                "observations": np.concatenate(obs_batch, axis=0),
                "actions": np.concatenate(act_batch, axis=0),
                "rewards": np.concatenate(rew_batch, axis=0),
                "dones": np.concatenate(done_batch, axis=0),
                "next_observations": np.concatenate(next_obs_batch, axis=0),
                "env_steps": int(rollout_fragment_steps * num_envs_per_actor),
                "actor_rollout_sec": float(time.perf_counter() - rollout_t0),
            }
            put_t0 = time.perf_counter()
            transition_queue.put(payload)
            put_block_sec = float(time.perf_counter() - put_t0)
            if put_block_sec > 1e-6:
                try:
                    metric_queue.put_nowait({
                        "actor_id": actor_id,
                        "actor_queue_put_block_sec": put_block_sec,
                    })
                except queue.Full:
                    pass
    except Exception:
        _write_trace(f"/tmp/actor_{actor_id}_trace_top.log")
        raise
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        if faulthandler_file is not None:
            try:
                faulthandler_file.close()
            except Exception:
                pass


def main() -> None:
    args = ParallelArgs()
    base = args.base

    # Build ACADOS once in the main process to avoid parallel make races.
    try:
        repo_root = os.path.abspath(os.path.join(project_root, ".."))
        pympc_root = os.path.join(repo_root, "pympc")
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        if pympc_root not in sys.path:
            sys.path.insert(0, pympc_root)

        from pympc.quadruped_pympc.controllers.gradient.nominal.centroidal_nmpc_nominal import Acados_NMPC_Nominal

        print("[Main] Pre-building Acados OCP solver to avoid multiprocessing race conditions...", flush=True)
        _ = Acados_NMPC_Nominal()
        print("[Main] Acados OCP solver built successfully.", flush=True)
    except Exception as e:
        print(f"[Main] Warning: Could not pre-build Acados OCP. Workers may fail. {e}", flush=True)

    # Keep compatibility with the updated SAC defaults unless explicitly changed.
    random.seed(base.seed)
    np.random.seed(base.seed)
    torch.manual_seed(base.seed)
    torch.backends.cudnn.deterministic = bool(base.torch_deterministic)
    torch.backends.cudnn.benchmark = False
    # Keep learner-side CPU thread pools bounded as actor count grows.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() and base.cuda else "cpu")

    env_kwargs = _build_env_kwargs(base)
    per_actor_learning_starts_steps = int(max(1, base.learning_starts // max(1, args.num_actors)))
    env_kwargs["learning_starts_steps"] = per_actor_learning_starts_steps
    env_kwargs["external_curriculum_control"] = True
    print(
        f"[PARALLEL SAC] per-actor learning_starts_steps={per_actor_learning_starts_steps} "
        f"(global learning_starts={base.learning_starts}, num_actors={args.num_actors})",
        flush=True,
    )
    # Do not create/close a temporary Isaac env in the main process: closing
    # SimulationApp here can terminate the whole launcher before actors start.
    obs_dim = int(IsaacStepHeightResidualEnv.RL_OBS_DIM)
    act_dim = int(IsaacStepHeightResidualEnv.RL_ACT_DIM)
    print(f"[Main] Using RL dims from env class: obs_dim={obs_dim}, act_dim={act_dim}", flush=True)
    act_low = np.full((act_dim,), -1.0, dtype=np.float32)
    act_high = np.full((act_dim,), 1.0, dtype=np.float32)

    actor = Actor(obs_dim, act_low, act_high).to(device)
    qf1 = SoftQNetwork(obs_dim, act_dim).to(device)
    qf2 = SoftQNetwork(obs_dim, act_dim).to(device)
    qf1_target = SoftQNetwork(obs_dim, act_dim).to(device)
    qf2_target = SoftQNetwork(obs_dim, act_dim).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=base.q_lr)
    actor_optimizer = optim.Adam(actor.parameters(), lr=base.policy_lr)

    if base.autotune:
        target_entropy = -float(act_dim)
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        log_alpha.requires_grad_(True)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=base.q_lr)
    else:
        alpha = base.alpha
        log_alpha = None
        a_optimizer = None

    latest_alpha_loss = float("nan")
    latest_policy_entropy = float("nan")
    latest_entropy_gap = float("nan")
    latest_log_alpha = float(log_alpha.item()) if log_alpha is not None else float("nan")

    rb = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, capacity=base.buffer_size, device=device)

    os.makedirs(base.log_dir, exist_ok=True)
    os.makedirs(base.checkpoint_dir, exist_ok=True)
    run_name = f"{args.run_name}_seed{base.seed}_{int(time.time())}"
    writer = SummaryWriter(os.path.join(base.log_dir, run_name))
    # Log the active env configuration once so TensorBoard captures the exact
    # terrain and completion settings used for this run.
    writer.add_text("config/terrain_type", str(base.terrain_type), 0)
    writer.add_scalar("config/procedural_narrow_success_x", float(base.procedural_narrow_success_x), 0)
    writer.add_scalar("config/success_reset_x_margin_m", float(getattr(base, "success_reset_x_margin_m", 0.0)), 0)
    writer.add_scalar("config/fixed_course", float(bool(base.fixed_course)), 0)
    writer.add_scalar("config/randomize_stairs", float(bool(base.randomize_stairs)), 0)


    ctx = mp.get_context("spawn")
    transition_queue: mp.Queue = ctx.Queue(maxsize=max(8, args.transition_queue_size))
    metric_queue: mp.Queue = ctx.Queue(maxsize=2048)
    shared_curriculum_stage = ctx.Value("i", 1)
    stop_event = ctx.Event()

    requested_envs_per_actor = int(max(1, args.num_envs_per_actor))
    if requested_envs_per_actor != 1:
        print(
            f"[PARALLEL SAC] overriding num_envs_per_actor={requested_envs_per_actor} to 1 to avoid nested Isaac vectorization and SPS collapse",
            flush=True,
        )
    effective_envs_per_actor = 1
    weights_queues = [ctx.Queue(maxsize=2) for _ in range(args.num_actors)]
    actor_procs: list[mp.Process] = []

    random_explore_steps_per_actor = int(max(1, base.learning_starts // max(1, args.num_actors)))

    for actor_id in range(args.num_actors):
        p = ctx.Process(
            target=_run_actor,
            args=(
                actor_id,
                vars(base),
                env_kwargs,
                int(effective_envs_per_actor),
                int(max(1, args.rollout_fragment_steps)),
                random_explore_steps_per_actor,
                int(max(1, args.actor_seed_stride)),
                transition_queue,
                metric_queue,
                weights_queues[actor_id],
                shared_curriculum_stage,
                True,
                stop_event,
            ),
            daemon=False,
        )
        p.start()
        actor_procs.append(p)
        # Sequential actor warmup reduces startup resource contention when
        # multiple actors initialize Isaac and ACADOS at the same time.
        if actor_id < (args.num_actors - 1) and float(args.actor_start_stagger_sec) > 0.0:
            time.sleep(float(args.actor_start_stagger_sec))

    init_weights = _to_cpu_state_dict(actor)
    for wq in weights_queues:
        wq.put(init_weights)

    total_env_steps = 0
    total_updates = 0
    total_episodes = 0
    start_time = time.time()
    reached_stairs_count = 0
    climbed_up_count = 0
    descended_to_flat_count = 0
    course_complete_count = 0
    deviation_lateral_count = 0
    episode_end_case_counts: dict[str, int] = defaultdict(int)
    episode_end_type_counts: dict[str, int] = defaultdict(int)
    actor_rollout_sec_sum = 0.0
    actor_put_block_sec_sum = 0.0
    payload_count = 0
    learner_wait_sec_sum = 0.0
    learner_update_sec_sum = 0.0
    curriculum_max_stage = int(base.curriculum_max_stage)
    curriculum_min_stage_episodes = int(base.curriculum_min_stage_episodes)
    curriculum_stage_episodes_since_advance = 0
    success_rate_window_episodes = int(base.curriculum_window_episodes)
    success_rate_recent_window = deque(maxlen=success_rate_window_episodes)

    try:
        while total_env_steps < base.total_timesteps:
            wait_t0 = time.perf_counter()
            try:
                payload = transition_queue.get(timeout=60.0)
            except queue.Empty:
                dead = [p.pid for p in actor_procs if (not p.is_alive())]
                if dead:
                    raise RuntimeError(f"Actor process died unexpectedly: {dead}")
                continue
            learner_wait_sec_sum += float(time.perf_counter() - wait_t0)
            actor_rollout_sec_sum += float(payload.get("actor_rollout_sec", 0.0))
            payload_count += 1

            obs_np = payload["observations"]
            next_obs_np = payload["next_observations"]
            act_np = payload["actions"]
            rew_np = payload["rewards"]
            done_np = payload["dones"]

            batch_n = _replay_add_batch(rb, obs_np, next_obs_np, act_np, rew_np, done_np)
            total_env_steps += int(payload.get("env_steps", batch_n))

            while True:
                try:
                    metric = metric_queue.get_nowait()
                except queue.Empty:
                    break
                if "actor_queue_put_block_sec" in metric:
                    actor_put_block_sec_sum += float(metric.get("actor_queue_put_block_sec", 0.0))
                    continue
                # Minimal per-episode logging: keep only essential summary metrics.
                total_episodes += 1
                writer.add_scalar("charts/episodic_return", float(metric.get("episodic_return", 0.0)), total_env_steps)
                writer.add_scalar("charts/episodic_length", float(metric.get("episodic_length", 0.0)), total_env_steps)

                # Course complete flag and count (kept per request).
                course_complete = float(metric.get("episode_course_complete", 0.0))
                writer.add_scalar("episode/course_complete", course_complete, total_env_steps)
                course_complete_count += int(course_complete > 0.5)

                # Command forward (kept per request).
                cmd_fwd = float(metric.get("episode_command_forward", base.command_forward))
                writer.add_scalar("episode/command_forward", cmd_fwd, total_env_steps)

                # Command yaw.
                cmd_yaw = float(metric.get("episode_command_yaw", base.command_yaw))
                writer.add_scalar("episode/command_yaw", cmd_yaw, total_env_steps)

                # Progression / recent success rate (kept per request).
                progression_success = float(metric.get("episode_curriculum_progression_success", 0.0))
                success_rate_recent_window.append(1.0 if progression_success > 0.5 else 0.0)
                curriculum_stage_episodes_since_advance += 1
                success_rate_recent = float(np.mean(success_rate_recent_window))
                writer.add_scalar("episode/success_rate_recent", success_rate_recent, total_env_steps)

                current_stage = int(shared_curriculum_stage.value)
                if (
                    current_stage < curriculum_max_stage
                    and curriculum_stage_episodes_since_advance >= curriculum_min_stage_episodes
                    and len(success_rate_recent_window) >= success_rate_window_episodes
                    and success_rate_recent >= float(base.curriculum_success_threshold)
                ):
                    shared_curriculum_stage.value = current_stage + 1
                    curriculum_stage_episodes_since_advance = 0
                    success_rate_recent_window.clear()
                    current_stage = int(shared_curriculum_stage.value)
                    print(
                        f"[PARALLEL SAC] advancing curriculum to stage {current_stage} "
                        f"(global recent success={success_rate_recent:.2%}, "
                        f"threshold={float(base.curriculum_success_threshold):.2%})",
                        flush=True,
                    )

                # Update and log global curriculum stage.
                metric["episode_curriculum_stage"] = float(current_stage)
                writer.add_scalar("episode/global_curriculum_stage", float(current_stage), total_env_steps)

                # Minimal episode end counts.
                episode_end_reason = str(metric.get("episode_end_reason", "unknown"))
                episode_end_type_counts[episode_end_reason] += 1

                episode_term_reason = str(metric.get("episode_termination_reason", "none"))
                if episode_end_reason == "terminated" and episode_term_reason != "none":
                    episode_end_case_counts[episode_term_reason] += 1

                writer.add_scalar("episode_end_counts/terminated_total", float(episode_end_type_counts.get('terminated', 0)), total_env_steps)
                writer.add_scalar("episode_end_counts/truncated_total", float(episode_end_type_counts.get('truncated', 0)), total_env_steps)
                writer.add_scalar("episode_end_counts/course_complete", float(course_complete_count), total_env_steps)

                writer.add_scalar("episode_end_counts/reason_stagnation", float(episode_end_case_counts.get('stagnation', 0)), total_env_steps)
                writer.add_scalar("episode_end_counts/reason_attitude", float(episode_end_case_counts.get('attitude', 0)), total_env_steps)
                writer.add_scalar("episode_end_counts/reason_attitude_hard", float(episode_end_case_counts.get('attitude_hard', 0)), total_env_steps)

                # Per-episode decomposed reward sums: only log currently used components.
                for component_name in (
                    "goal_proximity_reward",
                    "velocity_tracking_penalty",
                    "action_penalty",
                    "residual_smoothness_penalty",
                    "swing_time_penalty",
                    "swing_time_smoothness_penalty",
                    "total_reward",
                ):
                    key = f"reward_comp_sum_{component_name}"
                    if key in metric:
                        writer.add_scalar(f"episode/reward_component_sum/{component_name}", float(metric[key]), total_env_steps)

                # Step-level env and MPC diagnostics at episode boundary (trimmed).
                for key, tag in (
                    ("tracking_error", "env/tracking_error"),
                    ("base_z", "env/base_z"),
                    ("base_x", "env/base_x"),
                    ("base_roll", "env/base_roll"),
                    ("base_pitch", "env/base_pitch"),
                    ("residual_m", "env/residual_m"),
                    ("residual_m_applied", "env/residual_m_applied"),
                    ("residual_gate", "env/residual_gate"),
                ):
                    if key in metric:
                        writer.add_scalar(tag, float(metric[key]), total_env_steps)

                # Replace step-height and swing-time TensorBoard channels with
                # values read from the active swing trajectory generator.
                if "swing_generator_step_height" in metric:
                    writer.add_scalar("env/step_height_effective", float(metric["swing_generator_step_height"]), total_env_steps)
                if "swing_generator_swing_period" in metric:
                    writer.add_scalar("env/swing_time_s", float(metric["swing_generator_swing_period"]), total_env_steps)

                for key, tag in (
                    ("mpc_status", "mpc/status"),
                    ("mpc_compute_ms", "mpc/compute_ms"),
                    ("mpc_torque_max_abs", "mpc/torque_max_abs"),
                    ("mpc_torque_sat_ratio", "mpc/torque_sat_ratio"),
                    ("mpc_sqp_iter", "mpc/sqp_iter"),
                    ("mpc_qp_iter", "mpc/qp_iter"),
                    ("mpc_qp_status", "mpc/qp_status"),
                    ("mpc_maxiter_count_episode", "mpc/maxiter_count_episode"),
                    ("mpc_minstep_count_episode", "mpc/minstep_count_episode"),
                    ("mpc_failure_count_episode", "mpc/failure_count_episode"),
                ):
                    if key in metric:
                        writer.add_scalar(tag, float(metric[key]), total_env_steps)

                if total_episodes % max(1, int(base.episode_print_interval)) == 0:
                    sg_step_height = float(metric.get("swing_generator_step_height", 0.0))
                    sg_swing_period = float(metric.get("swing_generator_swing_period", 0.0))
                    print(
                        f"[PARALLEL SAC][episode {total_episodes}] "
                        f"swing_generator_step_height={sg_step_height:.4f}m "
                        f"swing_generator_swing_time={sg_swing_period:.4f}s",
                        flush=True,
                    )

            if total_env_steps > base.learning_starts and rb.size >= base.batch_size:
                update_t0 = time.perf_counter()
                payload_env_steps = int(max(1, payload.get("env_steps", batch_n)))
                for _ in range(payload_env_steps):
                    # Replay sampling quality drives learning efficiency.
                    # This is currently uniform sampling; prioritized replay can
                    # be plugged in here to focus updates on high-TD-error data.
                    data = rb.sample(base.batch_size)

                    with torch.no_grad():
                        next_actions, next_log_pi, _ = actor.get_action(data["next_observations"])
                        qf1_next_target = qf1_target(data["next_observations"], next_actions)
                        qf2_next_target = qf2_target(data["next_observations"], next_actions)
                        min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_log_pi
                        next_q_value = data["rewards"].flatten() + (1.0 - data["dones"].flatten()) * base.gamma * min_qf_next_target.view(-1)

                    qf1_values = qf1(data["observations"], data["actions"]).view(-1)
                    qf2_values = qf2(data["observations"], data["actions"]).view(-1)
                    qf1_loss = F.mse_loss(qf1_values, next_q_value)
                    qf2_loss = F.mse_loss(qf2_values, next_q_value)
                    qf_loss = qf1_loss + qf2_loss

                    q_optimizer.zero_grad(set_to_none=True)
                    qf_loss.backward()
                    q_optimizer.step()

                    if total_updates % base.policy_frequency == 0:
                        for _ in range(int(max(1, base.policy_frequency))):
                            pi, log_pi, _ = actor.get_action(data["observations"])
                            qf1_pi = qf1(data["observations"], pi)
                            qf2_pi = qf2(data["observations"], pi)
                            min_qf_pi = torch.min(qf1_pi, qf2_pi)
                            actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                            actor_optimizer.zero_grad(set_to_none=True)
                            actor_loss.backward()
                            actor_optimizer.step()

                            if base.autotune:
                                with torch.no_grad():
                                    _, log_pi_new, _ = actor.get_action(data["observations"])
                                latest_policy_entropy = float((-log_pi_new).mean().item())
                                latest_entropy_gap = float((-(log_pi_new + target_entropy)).mean().item())
                                alpha_loss = (-log_alpha.exp() * (log_pi_new + target_entropy)).mean()
                                latest_alpha_loss = float(alpha_loss.item())
                                a_optimizer.zero_grad()
                                alpha_loss.backward()
                                a_optimizer.step()
                                alpha = float(log_alpha.exp().item())
                                latest_log_alpha = float(log_alpha.item())

                    if total_updates % base.target_network_frequency == 0:
                        for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                            target_param.data.copy_(base.tau * param.data + (1 - base.tau) * target_param.data)
                        for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                            target_param.data.copy_(base.tau * param.data + (1 - base.tau) * target_param.data)

                    total_updates += 1

                    if total_updates % max(1, args.policy_broadcast_interval_updates) == 0:
                        latest = _to_cpu_state_dict(actor)
                        for wq in weights_queues:
                            while True:
                                try:
                                    wq.get_nowait()
                                except queue.Empty:
                                    break
                            wq.put(latest)

                    if total_env_steps % 100 == 0:
                        elapsed = max(time.time() - start_time, 1e-6)
                        writer.add_scalar("losses/qf1_values", qf1_values.mean().item(), total_env_steps)
                        writer.add_scalar("losses/qf2_values", qf2_values.mean().item(), total_env_steps)
                        writer.add_scalar("losses/qf1_loss", qf1_loss.item(), total_env_steps)
                        writer.add_scalar("losses/qf2_loss", qf2_loss.item(), total_env_steps)
                        writer.add_scalar("losses/qf_loss", (qf_loss.item() / 2.0), total_env_steps)
                        if 'actor_loss' in locals():
                            writer.add_scalar("losses/actor_loss", actor_loss.item(), total_env_steps)
                        writer.add_scalar("losses/alpha", alpha, total_env_steps)
                        if base.autotune:
                            writer.add_scalar("losses/alpha_loss", latest_alpha_loss, total_env_steps)
                            writer.add_scalar("losses/log_alpha", latest_log_alpha, total_env_steps)
                            writer.add_scalar("losses/policy_entropy", latest_policy_entropy, total_env_steps)
                            writer.add_scalar("losses/entropy_gap", latest_entropy_gap, total_env_steps)
                        writer.add_scalar("charts/SPS", int(total_env_steps / elapsed), total_env_steps)
                        writer.add_scalar("charts/total_updates", float(total_updates), total_env_steps)
                        writer.add_scalar("charts/replay_size", float(rb.size), total_env_steps)
                        writer.add_scalar("perf/learner_queue_wait_sec_total", float(learner_wait_sec_sum), total_env_steps)
                        writer.add_scalar("perf/learner_update_sec_total", float(learner_update_sec_sum), total_env_steps)
                        writer.add_scalar("perf/actor_rollout_sec_total", float(actor_rollout_sec_sum), total_env_steps)
                        writer.add_scalar("perf/actor_put_block_sec_total", float(actor_put_block_sec_sum), total_env_steps)
                        writer.add_scalar("perf/payload_count", float(payload_count), total_env_steps)
                        if payload_count > 0:
                            writer.add_scalar("perf/actor_rollout_sec_per_payload", float(actor_rollout_sec_sum / payload_count), total_env_steps)
                            writer.add_scalar("perf/actor_put_block_sec_per_payload", float(actor_put_block_sec_sum / payload_count), total_env_steps)

                learner_update_sec_sum += float(time.perf_counter() - update_t0)

            if total_env_steps > 0 and (total_env_steps % max(1, base.checkpoint_every)) == 0:
                ckpt_path = os.path.join(base.checkpoint_dir, f"{run_name}_step{total_env_steps}.pt")
                torch.save(
                    {
                        "actor": actor.state_dict(),
                        "qf1": qf1.state_dict(),
                        "qf2": qf2.state_dict(),
                        "args": vars(base),
                        "parallel_args": vars(args),
                        "global_step": total_env_steps,
                        "updates": total_updates,
                    },
                    ckpt_path,
                )
                print(f"[PARALLEL SAC] checkpoint saved: {ckpt_path}", flush=True)

        final_ckpt = os.path.join(base.checkpoint_dir, f"{run_name}_final.pt")
        torch.save({"actor": actor.state_dict(), "args": vars(base), "parallel_args": vars(args)}, final_ckpt)
        print(f"[PARALLEL SAC] final actor saved: {final_ckpt}", flush=True)
    finally:
        stop_event.set()
        for p in actor_procs:
            p.join(timeout=20.0)
            if p.is_alive():
                p.terminate()
        writer.close()


if __name__ == "__main__":
    main()
