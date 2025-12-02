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
    ):
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
        self.current_size = int(min(self.current_size + n, self.size))

    def sample_batch(self):
        if self.current_size < self.batch_size:
            raise ValueError("Not enough samples in ReplayBuffer.")
        idxs = torch.randint(0, self.current_size, (self.batch_size,), device=self.device)
        return {key: self.buffer[key][idxs] for key in self.buffer}

    def __len__(self):
        return self.current_size


def init_layer_uniform(layer: nn.Linear, gain: float = 1.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


class Actor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.hidden1 = nn.Linear(in_dim, 256)
        self.hidden2 = nn.Linear(256, 256)
        self.log_std_layer = init_layer_uniform(nn.Linear(256, out_dim))
        self.mu_layer = init_layer_uniform(nn.Linear(256, out_dim))

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, state: torch.Tensor, deterministic: bool = False):
        """
        Returns:
          action: [B, A] in [-1, 1]
          log_prob: [B, 1] (None if deterministic=True)
        """
        x = F.relu(self.hidden1(state))
        x = F.relu(self.hidden2(x))

        mu = self.mu_layer(x).tanh()
        log_std = self.log_std_layer(x).tanh()
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1.0)
        std = torch.exp(log_std)

        if deterministic:
            action = mu
            return action, None

        dist = torch.distributions.Normal(mu, std)
        z = dist.rsample()
        action = z.tanh()
        log_prob = dist.log_prob(z) - torch.log(1.0 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob


class CriticQ(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.hidden1 = nn.Linear(in_dim, 256)
        self.hidden2 = nn.Linear(256, 256)
        self.out = init_layer_uniform(nn.Linear(256, 1))

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat((state, action), dim=-1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        return self.out(x)


class CriticV(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.hidden1 = nn.Linear(in_dim, 256)
        self.hidden2 = nn.Linear(256, 256)
        self.out = init_layer_uniform(nn.Linear(256, 1))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.hidden1(state))
        x = F.relu(self.hidden2(x))
        return self.out(x)


class SAC(TensorDictModuleBase):
    """
    SAC agent that:
    - Exposes a TensorDict policy interface for TorchRL / Isaac Sim.
    - Uses Actor / CriticQ / CriticV and a custom ReplayBuffer.
    """

    def __init__(self, cfg, observation_spec, action_spec, device):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(device)
        self.gamma = 0.99
        self.tau = float(self.cfg.critic.tau)
        self.policy_update_freq = self.cfg.policy_update_freq
        self.vf_update_freq = self.cfg.vf_update_freq
        self.log_alpha = torch.tensor(
            float(torch.log(torch.tensor(float(self.cfg.temperature.init_alpha)))),
            device=self.device,
            requires_grad=True,
        )
        self.alpha_optim = torch.optim.Adam(
            [self.log_alpha],
            lr=float(self.cfg.temperature.learning_rate),
        )

        # Feature extractor
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

        # Initialize feature dim
        dummy_obs = observation_spec.zero()
        dummy_obs = dummy_obs.to(self.device)
        self.feature_extractor(dummy_obs)
        feature_dim = dummy_obs["_feature"].shape[-1]

        # per-sample observation shapes
        dummy_lidar = dummy_obs["agents", "observation", "lidar"]
        dummy_state = dummy_obs["agents", "observation", "state"]
        dummy_dyn = dummy_obs["agents", "observation", "dynamic_obstacle"]
        lidar_shape = dummy_lidar
        state_dim = dummy_state.shape[-1]
        dyn_shape = dummy_dyn

        # Networks
        self.actor = Actor(feature_dim, self.action_dim).to(self.device)
        self.qf_1 = CriticQ(feature_dim + self.action_dim).to(self.device)
        self.qf_2 = CriticQ(feature_dim + self.action_dim).to(self.device)
        self.vf = CriticV(feature_dim).to(self.device)
        self.vf_target = CriticV(feature_dim).to(self.device)
        self.vf_target.load_state_dict(self.vf.state_dict())

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
            self.actor.parameters(),
            lr=float(self.cfg.actor.learning_rate),
        )

        encoder_params = list(self.feature_extractor.parameters())
        self.qf_1_optimizer = torch.optim.Adam(
            list(self.qf_1.parameters()) + encoder_params,
            lr=float(self.cfg.critic.learning_rate),
        )

        self.qf_2_optimizer = torch.optim.Adam(
            self.qf_2.parameters(),
            lr=float(self.cfg.critic.learning_rate),
        )
        self.vf_optimizer = torch.optim.Adam(
            self.vf.parameters(),
            lr=float(self.cfg.critic.learning_rate),
        )

        self.total_steps = 0
        self.total_updates = 0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        tensordict = tensordict.to(self.device)

        # Extract features
        self.feature_extractor(tensordict)
        feat = tensordict["_feature"]

        # Determine exploration type (stochastic vs deterministic)
        et = exploration_type()
        deterministic = et is ExplorationType.MEAN
        with torch.no_grad():
            action_norm, _ = self.actor(feat, deterministic=deterministic)

        tensordict.set(("agents", "action_normalized"), action_norm)

        # Scale and map to world frame
        scaled = action_norm * self.action_limit
        direction = tensordict.get(("agents", "observation", "direction"))
        actions_world = vec_to_world(scaled, direction)
        tensordict.set(("agents", "action"), actions_world)
        return tensordict

    def train(self, tensordict: TensorDict) -> Dict[str, float]:
        """
        Called once per collector batch.
        tensordict batch dims: [E, T, ...], where
        E = num_envs
        T = training_frame_num
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
            B = E * T  # total env frames in this batch

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

            # Store transitions (raw obs, not features)
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
            "alpha": float(self.alpha),
            "total_steps": float(self.total_steps),
        }

        warmup_steps = int(self.cfg.warmup_steps)
        batch_size = int(self.cfg.batch_size)
        if self.total_steps < warmup_steps or len(self.memory) < batch_size:
            # Still in replay warmup: collect more data before learning
            return stats

        # Number of gradient steps for this collector batch.
        # updates_per_step is "grad steps per env frame".
        # B / batch_size is "batches per collector batch".
        updates_per_step = float(self.cfg.updates_per_step)
        num_updates = max(1, int(updates_per_step * B / batch_size))

        actor_loss_acc = 0.0
        critic_loss_acc = 0.0
        value_loss_acc = 0.0

        for _ in range(num_updates):
            actor_loss, critic_loss, value_loss = self._update_once()
            actor_loss_acc += actor_loss
            critic_loss_acc += critic_loss
            value_loss_acc += value_loss

        stats.update(
            {
                "actor_loss": actor_loss_acc / num_updates,
                "critic_loss": critic_loss_acc / num_updates,
                "value_loss": value_loss_acc / num_updates,
            }
        )
        return stats

    # One SAC gradient step
    def _update_once(self):
        samples = self.memory.sample_batch()

        # raw observations from buffer
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

        # ---- Recompute features with gradients ----
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

        # ---- Q-function loss ----
        with torch.no_grad():
            v_next = self.vf_target(next_feat)  # [B, 1]
            v_target = reward + self.gamma * v_next * (1.0 - done)

        q1 = self.qf_1(feat, action)
        q2 = self.qf_2(feat, action)
        qf_1_loss = F.mse_loss(q1, v_target)
        qf_2_loss = F.mse_loss(q2, v_target)
        critic_loss = qf_1_loss + qf_2_loss

        # ---- V-function loss ----
        with torch.no_grad():
            new_action, log_prob = self.actor(feat, deterministic=False)
            q1_pi = self.qf_1(feat, new_action)
            q2_pi = self.qf_2(feat, new_action)
            q_pi = torch.min(q1_pi, q2_pi)
            v_target_pi = q_pi - self.alpha * log_prob  # [B, 1]

        v_pred = self.vf(feat)
        vf_loss = F.mse_loss(v_pred, v_target_pi)

        # ---- Policy loss ----
        actor_loss = torch.zeros(1, device=self.device)
        if self.total_updates % self.policy_update_freq == 0:
            feat_actor = feat.detach()  # do not backprop actor loss into encoder

            new_action, log_prob = self.actor(feat_actor, deterministic=False)
            q1_pi = self.qf_1(feat_actor, new_action)
            q2_pi = self.qf_2(feat_actor, new_action)
            q_pi = torch.min(q1_pi, q2_pi)
            actor_loss = (self.alpha * log_prob - q_pi).mean()

            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()

        # ---- Temperature / entropy objective ----
        with torch.no_grad():
            target_entropy = -float(self.cfg.temperature.target_entropy_scale) * float(
                self.action_dim
            )

        alpha_loss = -(self.log_alpha * (log_prob.detach() + target_entropy)).mean()

        self.alpha_optim.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optim.step()

        # ---- Optimize Q and V ----
        self.qf_1_optimizer.zero_grad(set_to_none=True)
        self.qf_2_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.qf_1_optimizer.step()
        self.qf_2_optimizer.step()

        # ---- Conditional V update ----
        if self.total_updates % self.vf_update_freq == 0:
            self.vf_optimizer.zero_grad(set_to_none=True)
            vf_loss.backward()
            self.vf_optimizer.step()

        # ---- Update V-target ----
        with torch.no_grad():
            for p, p_targ in zip(self.vf.parameters(), self.vf_target.parameters()):
                p_targ.data.mul_(1.0 - self.tau)
                p_targ.data.add_(self.tau * p.data)

        self.total_updates += 1

        return (
            float(actor_loss.item()),
            float(critic_loss.item()),
            float(vf_loss.item()),
        )
