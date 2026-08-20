"""
Standalone SAC Actor network for inference.

This module contains ONLY the Actor class needed for checkpoint loading
and inference. It intentionally avoids importing heavy dependencies like
gymnasium, tensorboard, or tensorflow that would conflict with Isaac Sim's
native grpc/protobuf libraries.

The architecture must match train_sac_step_height_residual_parallel.py exactly.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    """SAC actor (squashed Gaussian policy) for residual and swing-time control."""

    def __init__(
        self,
        obs_dim: int,
        act_low: np.ndarray,
        act_high: np.ndarray,
        hidden_dim1: int = 256,
        hidden_dim2: int | None = None,
    ):
        super().__init__()
        act_dim = int(np.prod(act_low.shape))
        h1 = int(hidden_dim1)
        h2 = int(hidden_dim2) if hidden_dim2 is not None else int(hidden_dim1)
        self.fc1 = nn.Linear(obs_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc_mean = nn.Linear(h2, act_dim)
        self.fc_logstd = nn.Linear(h2, act_dim)

        self.register_buffer(
            "action_scale",
            torch.tensor((act_high - act_low) / 2.0, dtype=torch.float32),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((act_high + act_low) / 2.0, dtype=torch.float32),
        )

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
