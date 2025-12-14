from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from tensordict.nn import TensorDictModule, TensorDictModuleBase, TensorDictSequential
from tensordict.tensordict import TensorDict
from torchrl.envs.transforms import CatTensors
from torchrl.envs.utils import ExplorationType, exploration_type
from utils import make_mlp, vec_to_world


class ReplayBuffer:
    """Simple replay buffer storing raw observations and transitions on device."""

    def __init__(
        self,
        lidar_shape,
        state_dim: int,
        dyn_shape,
        action_dim: int,
        size: int,
        batch_size: int,
        device: torch.device,
    ):
        self.size = int(size)
        self.batch_size = int(batch_size)
        self.device = device

        self.ptr = 0
        self.current_size = 0

        self.buffer = {
            # current obs
            "lidar": torch.zeros((self.size, *lidar_shape), device=self.device),
            "state": torch.zeros((self.size, state_dim), device=self.device),
            "dyn": torch.zeros((self.size, *dyn_shape), device=self.device),
            # next obs
            "next_lidar": torch.zeros((self.size, *lidar_shape), device=self.device),
            "next_state": torch.zeros((self.size, state_dim), device=self.device),
            "next_dyn": torch.zeros((self.size, *dyn_shape), device=self.device),
            # transition stuff
            "acts": torch.zeros(self.size, action_dim, device=self.device),
            "rews": torch.zeros(self.size, 1, device=self.device),
            "done": torch.zeros(self.size, 1, device=self.device),
        }

    def store(
        self,
        lidar: torch.Tensor,
        state: torch.Tensor,
        dyn: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_lidar: torch.Tensor,
        next_state: torch.Tensor,
        next_dyn: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        """
        Inserts a batch of collected transitions into the replay buffer.
        """
        n = lidar.shape[0]
        if n == 0:
            return

        idxs = (torch.arange(n, device=self.device) + self.ptr) % self.size

        self.buffer["lidar"][idxs] = lidar
        self.buffer["state"][idxs] = state
        self.buffer["dyn"][idxs] = dyn
        self.buffer["acts"][idxs] = action
        self.buffer["rews"][idxs] = reward
        self.buffer["next_lidar"][idxs] = next_lidar
        self.buffer["next_state"][idxs] = next_state
        self.buffer["next_dyn"][idxs] = next_dyn
        self.buffer["done"][idxs] = done

        self.ptr = int((self.ptr + n) % self.size)
        self.current_size = min(self.current_size + n, self.size)

    def sample_batch(self) -> Dict[str, torch.Tensor]:
        idxs = torch.randint(0, self.current_size, (self.batch_size,), device=self.device)
        return {key: self.buffer[key][idxs] for key in self.buffer}

    def __len__(self) -> int:
        return self.current_size


def init_layer_ortho(layer: nn.Linear, gain: float = 1.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


class ResidualBlock(nn.Module):
    """256-d residual MLP block with LayerNorm."""

    def __init__(self, dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

        init_layer_ortho(self.fc1)
        init_layer_ortho(self.fc2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.ln1(h)
        h = F.relu(h)

        h = self.fc2(h)
        h = self.ln2(h)

        return F.relu(x + h)


class Actor(nn.Module):
    """Gaussian policy with Tanh squashing, returning actions in [-1, 1]."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        log_std_min: float = -3.0,
        log_std_max: float = 0.5,
    ):
        super().__init__()
        hidden_dim = 256

        self.fc_in = nn.Linear(in_dim, hidden_dim)
        init_layer_ortho(self.fc_in)

        self.block1 = ResidualBlock(hidden_dim)
        self.block2 = ResidualBlock(hidden_dim)

        self.ln_out = nn.LayerNorm(hidden_dim)

        self.mu_layer = init_layer_ortho(nn.Linear(hidden_dim, out_dim))
        self.log_std_layer = init_layer_ortho(nn.Linear(hidden_dim, out_dim))

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, feat: torch.Tensor, deterministic: bool = False):
        x = self.fc_in(feat)
        x = F.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.ln_out(x)

        mu = self.mu_layer(x)

        log_std = self.log_std_layer(x).tanh()
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1.0)
        std = torch.exp(log_std)

        if deterministic:
            return torch.tanh(mu), None

        dist = torch.distributions.Normal(mu, std)
        z = dist.rsample()
        action = torch.tanh(z)

        log_prob = dist.log_prob(z) - torch.log(1.0 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob


class CriticQ(nn.Module):
    """Q(s, a) network with 3x256 MLP + residual blocks."""

    def __init__(self, feat_dim: int, action_dim: int):
        super().__init__()
        hidden_dim = 256
        in_dim = feat_dim + action_dim

        self.fc_in = nn.Linear(in_dim, hidden_dim)
        init_layer_ortho(self.fc_in)

        self.block1 = ResidualBlock(hidden_dim)
        self.block2 = ResidualBlock(hidden_dim)

        self.ln_out = nn.LayerNorm(hidden_dim)
        self.out = init_layer_ortho(nn.Linear(hidden_dim, 1))

    def forward(self, feat: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([feat, action], dim=-1)
        x = self.fc_in(x)
        x = F.relu(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.ln_out(x)
        return self.out(x)


class SAC(TensorDictModuleBase):
    """
    SAC with:
    - Shared encoder trained only via critic TD-error (feat_actor is detached)
    - Twin Q-critics + target networks
    - Automatic entropy tuning with alpha init from cfg
    - Gradient clipping on actor and critic
    """

    def __init__(self, cfg, observation_spec, action_spec, device):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(device)

        self.gamma = 0.99
        self.tau = float(self.cfg.critic.tau)

        # gradient clipping
        self.max_grad_norm_actor = 5.0
        self.max_grad_norm_critic = 5.0

        # Shared feature extractor
        feature_extractor_network = nn.Sequential(
            nn.LazyConv2d(out_channels=4, kernel_size=(5, 3), padding=(2, 1)),
            nn.ELU(),
            nn.LazyConv2d(out_channels=16, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1)),
            nn.ELU(),
            nn.LazyConv2d(out_channels=16, kernel_size=(5, 3), stride=(2, 2), padding=(2, 1)),
            nn.ELU(),
            Rearrange("n c w h -> n (c w h)"),
            nn.LazyLinear(128),
            nn.LayerNorm(128),
        ).to(self.device)

        dynamic_obstacle_network = nn.Sequential(
            Rearrange("n c w h -> n (c w h)"),
            make_mlp([128, 64]),
        ).to(self.device)

        self.feature_extractor = TensorDictSequential(
            TensorDictModule(
                feature_extractor_network,
                [("agents", "observation", "lidar")],
                ["_cnn_feature"],
            ),
            TensorDictModule(
                dynamic_obstacle_network,
                [("agents", "observation", "dynamic_obstacle")],
                ["_dynamic_obstacle_feature"],
            ),
            CatTensors(
                ["_cnn_feature", ("agents", "observation", "state"), "_dynamic_obstacle_feature"],
                "_feature",
                del_keys=False,
            ),
            TensorDictModule(
                make_mlp([256, 256]),
                ["_feature"],
                ["_feature"],
            ),
        ).to(self.device)

        # Action space
        self.n_agents, self.action_dim = action_spec.shape
        self.action_limit = float(self.cfg.actor.action_limit)

        # alpha tuning with initialization from cfg
        init_alpha = float(self.cfg.temperature.init_alpha)
        self.target_entropy = -float(self.action_dim)
        self.log_alpha = nn.Parameter(torch.log(torch.tensor(init_alpha, device=self.device)))
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha],
            lr=float(self.cfg.temperature.learning_rate),
        )

        # ------------------------------------------------------------------
        # Infer feature dimension and replay buffer shapes
        # ------------------------------------------------------------------
        dummy_obs = observation_spec.zero().to(self.device)
        self.feature_extractor(dummy_obs)
        feat_dim = dummy_obs["_feature"].shape[-1]

        dummy_lidar = dummy_obs["agents", "observation", "lidar"]
        dummy_state = dummy_obs["agents", "observation", "state"]
        dummy_dyn = dummy_obs["agents", "observation", "dynamic_obstacle"]

        lidar_shape = dummy_lidar.shape[1:]  # drop batch dim
        state_dim = dummy_state.shape[-1]
        dyn_shape = dummy_dyn.shape[1:]

        # Networks
        self.actor = Actor(feat_dim, self.action_dim).to(self.device)

        # Twin critics
        self.q1 = CriticQ(feat_dim, self.action_dim).to(self.device)
        self.q2 = CriticQ(feat_dim, self.action_dim).to(self.device)
        self.q1_target = CriticQ(feat_dim, self.action_dim).to(self.device)
        self.q2_target = CriticQ(feat_dim, self.action_dim).to(self.device)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Replay buffer
        capacity = int(self.cfg.replay_capacity)
        batch_size = int(self.cfg.batch_size)
        self.memory = ReplayBuffer(
            lidar_shape=lidar_shape,
            state_dim=state_dim,
            dyn_shape=dyn_shape,
            action_dim=self.action_dim,
            size=capacity,
            batch_size=batch_size,
            device=self.device,
        )

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            list(self.actor.parameters()),
            lr=float(self.cfg.actor.learning_rate),
        )

        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters())
            + list(self.q2.parameters())
            + list(self.feature_extractor.parameters()),
            lr=float(self.cfg.critic.learning_rate),
        )

        self.total_steps = 0
        self.total_updates = 0

    # Policy interface used by TorchRL collector
    def __call__(self, tensordict: TensorDict) -> TensorDict:
        tensordict = tensordict.to(self.device)
        self.feature_extractor(tensordict)
        feat = tensordict["_feature"]

        et = exploration_type()
        deterministic = et is ExplorationType.MEAN

        with torch.no_grad():
            action_norm, _ = self.actor(feat, deterministic=deterministic)

        tensordict.set(("agents", "action_normalized"), action_norm)
        scaled = action_norm * self.action_limit
        direction = tensordict.get(("agents", "observation", "direction"))
        actions_world = vec_to_world(scaled, direction)
        tensordict.set(("agents", "action"), actions_world)
        return tensordict

    # Training step called from train.py
    def train(self, tensordict: TensorDict) -> Dict[str, float]:
        """
        Called once per collector batch.
        tensordict batch dims: [E, T, ...]
        """
        with torch.no_grad():
            lidar = tensordict["agents", "observation", "lidar"]
            state = tensordict["agents", "observation", "state"]
            dyn = tensordict["agents", "observation", "dynamic_obstacle"]

            next_td = tensordict["next"]
            next_lidar = next_td["agents", "observation", "lidar"]
            next_state = next_td["agents", "observation", "state"]
            next_dyn = next_td["agents", "observation", "dynamic_obstacle"]

            E, T = lidar.shape[0], lidar.shape[1]
            B = E * T

            # flatten over env and time
            lidar_flat = lidar.reshape(B, *lidar.shape[2:])
            state_flat = state.reshape(B, state.shape[-1])
            dyn_flat = dyn.reshape(B, *dyn.shape[2:])

            next_lidar_flat = next_lidar.reshape(B, *next_lidar.shape[2:])
            next_state_flat = next_state.reshape(B, next_state.shape[-1])
            next_dyn_flat = next_dyn.reshape(B, *next_dyn.shape[2:])

            act_norm = tensordict["agents", "action_normalized"]  # [E, T, A]
            rewards = tensordict["next", "agents", "reward"]  # [E, T, 1]
            terminated = tensordict["next", "terminated"]  # [E, T, 1]
            dones = terminated.float()

            act_flat = act_norm.reshape(B, -1)
            rewards_flat = rewards.reshape(B, 1).float()
            dones_flat = dones.reshape(B, 1).float()

            self.memory.store(
                lidar_flat,
                state_flat,
                dyn_flat,
                act_flat,
                rewards_flat,
                next_lidar_flat,
                next_state_flat,
                next_dyn_flat,
                dones_flat,
            )
            self.total_steps += B

        stats: Dict[str, float] = {
            "buffer_size": float(len(self.memory)),
            "total_steps": float(self.total_steps),
            "alpha": float(self.log_alpha.exp().item()),
        }

        warmup_steps = int(self.cfg.warmup_steps)
        batch_size = int(self.cfg.batch_size)
        if self.total_steps < warmup_steps or len(self.memory) < batch_size:
            return stats

        updates_per_step = float(self.cfg.updates_per_step)
        num_updates = max(1, int(updates_per_step * B / batch_size))

        actor_loss_acc = 0.0
        critic_loss_acc = 0.0
        last_update_stats: Dict[str, float] = {}

        for _ in range(num_updates):
            update_stats = self._update_once()
            actor_loss_acc += update_stats["actor_loss"]
            critic_loss_acc += update_stats["critic_loss"]
            last_update_stats = update_stats

        stats.update(
            {
                "actor_loss": actor_loss_acc / num_updates,
                "critic_loss": critic_loss_acc / num_updates,
                "alpha": float(self.log_alpha.exp().item()),
            }
        )

        # Add extra diagnostics from last gradient step
        stats.update(
            {
                "alpha_loss": last_update_stats["alpha_loss"],
                "q1_pred_mean": last_update_stats["q1_pred_mean"],
                "q2_pred_mean": last_update_stats["q2_pred_mean"],
                "target_q_mean": last_update_stats["target_q_mean"],
                "log_prob_mean": last_update_stats["log_prob_mean"],
            }
        )

        return stats

    # One SAC gradient step
    def _update_once(self) -> Dict[str, float]:
        samples = self.memory.sample_batch()

        lidar = samples["lidar"]  # [B, 1, H, W]
        state_obs = samples["state"]  # [B, S]
        dyn = samples["dyn"]  # [B, 1, N_dyn, D]

        next_lidar = samples["next_lidar"]
        next_state_obs = samples["next_state"]
        next_dyn = samples["next_dyn"]

        action = samples["acts"]  # [B, A]
        reward = samples["rews"]  # [B, 1]
        done = samples["done"]  # [B, 1]

        B = lidar.shape[0]

        # Recompute features with gradients (encoder trained via critic only)
        cur_td = TensorDict(
            {
                ("agents", "observation", "lidar"): lidar,
                ("agents", "observation", "state"): state_obs,
                ("agents", "observation", "dynamic_obstacle"): dyn,
            },
            batch_size=[B],
            device=self.device,
        )
        self.feature_extractor(cur_td)
        feat = cur_td["_feature"]  # [B, F]

        next_td = TensorDict(
            {
                ("agents", "observation", "lidar"): next_lidar,
                ("agents", "observation", "state"): next_state_obs,
                ("agents", "observation", "dynamic_obstacle"): next_dyn,
            },
            batch_size=[B],
            device=self.device,
        )
        self.feature_extractor(next_td)
        next_feat = next_td["_feature"]  # [B, F]

        # Critic / Q update
        with torch.no_grad():
            next_action, next_log_prob = self.actor(next_feat, deterministic=False)

            q1_next = self.q1_target(next_feat, next_action)
            q2_next = self.q2_target(next_feat, next_action)
            q_next = torch.min(q1_next, q2_next)

            alpha = self.log_alpha.exp().detach()
            target_q = reward + self.gamma * (1.0 - done) * (q_next - alpha * next_log_prob)

        q1_pred = self.q1(feat, action)
        q2_pred = self.q2(feat, action)

        critic_loss_1 = F.mse_loss(q1_pred, target_q)
        critic_loss_2 = F.mse_loss(q2_pred, target_q)
        critic_loss = 0.5 * (critic_loss_1 + critic_loss_2)

        self.q_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.q1.parameters())
            + list(self.q2.parameters())
            + list(self.feature_extractor.parameters()),
            self.max_grad_norm_critic,
        )
        self.q_optimizer.step()

        # # freeze encoder
        # for p in self.feature_extractor.parameters():
        #     p.requires_grad_(False)

        # recompute features for actor so it has its own graph
        # actor_td = TensorDict(
        #     {
        #         ("agents", "observation", "lidar"): lidar,
        #         ("agents", "observation", "state"): state_obs,
        #         ("agents", "observation", "dynamic_obstacle"): dyn,
        #     },
        #     batch_size=[B],
        #     device=self.device,
        # )
        # self.feature_extractor(actor_td)
        # feat_actor = actor_td["_feature"]

        # new_action, log_prob = self.actor(feat_actor, deterministic=False)

        feat_actor = feat.detach()
        new_action, log_prob = self.actor(feat_actor, deterministic=False)

        q1_new = self.q1(feat_actor, new_action)
        q2_new = self.q2(feat_actor, new_action)
        q_new = torch.min(q1_new, q2_new)

        alpha = self.log_alpha.exp().detach()
        actor_loss = (alpha * log_prob - q_new).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm_actor)
        self.actor_optimizer.step()

        # Entropy temperature update
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # # unfreeze encoder
        # for p in self.feature_extractor.parameters():
        #     p.requires_grad_(True)

        # Target Q update (Polyak averaging)
        with torch.no_grad():
            for p, p_targ in zip(self.q1.parameters(), self.q1_target.parameters()):
                p_targ.data.mul_(1.0 - self.tau)
                p_targ.data.add_(self.tau * p.data)

            for p, p_targ in zip(self.q2.parameters(), self.q2_target.parameters()):
                p_targ.data.mul_(1.0 - self.tau)
                p_targ.data.add_(self.tau * p.data)

        self.total_updates += 1

        # Return scalars to be logged by train.py via wandb
        return {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "q1_pred_mean": float(q1_pred.mean().item()),
            "q2_pred_mean": float(q2_pred.mean().item()),
            "target_q_mean": float(target_q.mean().item()),
            "log_prob_mean": float(log_prob.mean().item()),
        }
