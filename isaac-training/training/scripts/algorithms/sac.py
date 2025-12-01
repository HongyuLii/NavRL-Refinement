# training/scripts/algorithms/sac.py

import copy
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from tensordict.nn import TensorDictModule, TensorDictModuleBase, TensorDictSequential
from tensordict.tensordict import TensorDict
from torchrl.envs.transforms import CatTensors
from torchrl.modules import ProbabilisticActor
from utils import Actor, IndependentNormal, make_mlp, vec_to_world


class SAC(TensorDictModuleBase):
    def __init__(self, cfg, observation_spec, action_spec, device):
        super().__init__()
        self.cfg = cfg
        self.device = device

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
            TensorDictModule(make_mlp([256, 256]), ["_feature"], ["_feature"]),
        ).to(self.device)

        # Actor: Gaussian policy over actions
        self.n_agents, self.action_dim = action_spec.shape

        actor_core = TensorDictModule(
            Actor(self.action_dim),
            in_keys=["_feature"],
            out_keys=["loc", "scale"],
        )

        self.actor = ProbabilisticActor(
            actor_core,
            in_keys=["loc", "scale"],
            out_keys=[("agents", "action_normalized")],
            distribution_class=IndependentNormal,
            return_log_prob=True,
        ).to(self.device)

        # Critics: twin Q networks + targets
        hidden_sizes = list(self.cfg.critic.hidden_sizes)

        def build_q_net():
            layers = []
            in_dim = None
            for h in hidden_sizes:
                layers.append(nn.LazyLinear(h))
                layers.append(nn.ReLU())
            layers.append(nn.LazyLinear(1))
            return nn.Sequential(*layers).to(self.device)

        self.q1 = build_q_net()
        self.q2 = build_q_net()
        self.q1_target = copy.deepcopy(self.q1).to(self.device)
        self.q2_target = copy.deepcopy(self.q2).to(self.device)
        for p in self.q1_target.parameters():
            p.requires_grad = False
        for p in self.q2_target.parameters():
            p.requires_grad = False

        # Entropy tuning
        self.target_entropy = -float(self.cfg.temperature.target_entropy_scale) * float(
            self.action_dim
        )
        init_alpha = float(self.cfg.temperature.init_alpha)
        self.log_alpha = torch.tensor(
            float(torch.log(torch.tensor(init_alpha))),
            device=self.device,
            requires_grad=True,
        )
        self.alpha_optim = torch.optim.Adam(
            [self.log_alpha], lr=float(self.cfg.actor.learning_rate)
        )

        # Optimizers
        self.actor_optim = torch.optim.Adam(
            self.actor.parameters(), lr=float(self.cfg.actor.learning_rate)
        )
        self.q1_optim = torch.optim.Adam(
            self.q1.parameters(), lr=float(self.cfg.critic.learning_rate)
        )
        self.q2_optim = torch.optim.Adam(
            self.q2.parameters(), lr=float(self.cfg.critic.learning_rate)
        )

        # SAC hyperparameters
        self.gamma = 0.99
        self.tau = float(self.cfg.critic.tau)

        # Initialize lazy modules and replay buffer
        dummy_input = observation_spec.zero()
        self.__call__(dummy_input)
        feature_dim = dummy_input["_feature"].shape[-1]

        # Initialize Q networks
        dummy_sa = torch.zeros(1, feature_dim + self.action_dim, device=self.device)
        self.q1(dummy_sa)
        self.q2(dummy_sa)
        self.q1_target(dummy_sa)
        self.q2_target(dummy_sa)

        self._init_replay_buffer(feature_dim)
        self.total_steps = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def _init_replay_buffer(self, feature_dim: int):
        capacity = int(self.cfg.replay_capacity)
        self.capacity = capacity
        self.buffer_ptr = 0
        self.buffer_size = 0

        self.replay = {
            "feat": torch.zeros(capacity, feature_dim, device=self.device, dtype=torch.float32),
            "act": torch.zeros(capacity, self.action_dim, device=self.device, dtype=torch.float32),
            "rew": torch.zeros(capacity, 1, device=self.device, dtype=torch.float32),
            "done": torch.zeros(capacity, 1, device=self.device, dtype=torch.float32),
            "next_feat": torch.zeros(
                capacity, feature_dim, device=self.device, dtype=torch.float32
            ),
        }

    def _add_to_buffer(
        self,
        feat: torch.Tensor,
        act: torch.Tensor,
        rew: torch.Tensor,
        done: torch.Tensor,
        next_feat: torch.Tensor,
    ):
        n = feat.shape[0]
        if n == 0:
            return

        for i in range(n):
            j = (self.buffer_ptr + i) % self.capacity
            self.replay["feat"][j] = feat[i]
            self.replay["act"][j] = act[i]
            self.replay["rew"][j] = rew[i]
            self.replay["done"][j] = done[i]
            self.replay["next_feat"][j] = next_feat[i]

        self.buffer_ptr = (self.buffer_ptr + n) % self.capacity
        self.buffer_size = min(self.buffer_size + n, self.capacity)

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        # Extract features and sample actions
        self.feature_extractor(tensordict)
        self.actor(tensordict)

        # Rescale to world-frame velocity commands
        a_norm = tensordict["agents", "action_normalized"].tanh()
        actions = a_norm * self.cfg.actor.action_limit

        actions_world = vec_to_world(actions, tensordict["agents", "observation", "direction"])
        tensordict["agents", "action"] = actions_world
        return tensordict

    def train(self, tensordict: TensorDict) -> Dict[str, float]:
        with torch.no_grad():
            lidar = tensordict["agents", "observation", "lidar"]
            state = tensordict["agents", "observation", "state"]
            dyn = tensordict["agents", "observation", "dynamic_obstacle"]

            E, T = lidar.shape[0], lidar.shape[1]
            B = E * T

            # current state features
            cur_td = TensorDict(
                {
                    ("agents", "observation", "lidar"): lidar.reshape(B, *lidar.shape[2:]),
                    ("agents", "observation", "state"): state.reshape(B, state.shape[-1]),
                    ("agents", "observation", "dynamic_obstacle"): dyn.reshape(B, *dyn.shape[2:]),
                },
                batch_size=[B],
                device=self.device,
            )

            self.feature_extractor(cur_td)
            feat = cur_td["_feature"].reshape(E, T, -1)

            # next state features
            next = tensordict["next"]
            next_lidar = next["agents", "observation", "lidar"]
            next_state = next["agents", "observation", "state"]
            next_dyn = next["agents", "observation", "dynamic_obstacle"]

            next_td = TensorDict(
                {
                    ("agents", "observation", "lidar"): next_lidar.reshape(
                        B, *next_lidar.shape[2:]
                    ),
                    ("agents", "observation", "state"): next_state.reshape(B, next_state.shape[-1]),
                    ("agents", "observation", "dynamic_obstacle"): next_dyn.reshape(
                        B, *next_dyn.shape[2:]
                    ),
                },
                batch_size=[B],
                device=self.device,
            )

            self.feature_extractor(next_td)
            next_feat = next_td["_feature"].reshape(E, T, -1)

            # actions, rewards, dones
            act_norm = tensordict["agents", "action_normalized"]  # [E, T, A]
            rewards = tensordict["next", "agents", "reward"]  # [E, T, 1]
            dones = tensordict["next", "terminated"]  # [E, T, 1]

            feat_flat = feat.reshape(B, -1)
            next_feat_flat = next_feat.reshape(B, -1)
            act_flat = act_norm.reshape(B, -1)
            rewards_flat = rewards.reshape(B, 1).float()
            dones_flat = dones.reshape(B, 1).float()

            self._add_to_buffer(feat_flat, act_flat, rewards_flat, dones_flat, next_feat_flat)
            self.total_steps += B

        stats: Dict[str, float] = {
            "buffer_size": float(self.buffer_size),
            "alpha": float(self.alpha.item()),
        }

        # Warmup
        if self.total_steps < int(self.cfg.warmup_steps) or self.buffer_size < int(
            self.cfg.batch_size
        ):
            return stats

        num_updates = max(1, int(self.cfg.updates_per_step))

        actor_loss_acc = 0.0
        critic_loss_acc = 0.0
        alpha_loss_acc = 0.0

        for _ in range(num_updates):
            actor_loss, critic_loss, alpha_loss = self._update_once()
            actor_loss_acc += actor_loss
            critic_loss_acc += critic_loss
            alpha_loss_acc += alpha_loss

        stats.update(
            {
                "actor_loss": actor_loss_acc / num_updates,
                "critic_loss": critic_loss_acc / num_updates,
                "temperature_loss": alpha_loss_acc / num_updates,
            }
        )
        return stats

    # SAC gradient step
    def _update_once(self):
        batch_size = int(self.cfg.batch_size)
        idx = torch.randint(0, self.buffer_size, (batch_size,), device=self.device)

        feat = self.replay["feat"][idx]  # [B, F]
        act = self.replay["act"][idx]  # [B, A]
        rew = self.replay["rew"][idx].squeeze(-1)  # [B]
        done = self.replay["done"][idx].squeeze(-1)  # [B]
        next_feat = self.replay["next_feat"][idx]  # [B, F]

        alpha = self.alpha.detach()

        # Critic target
        with torch.no_grad():
            td_next = TensorDict(
                {"_feature": next_feat},
                batch_size=[batch_size],
            )
            self.actor(td_next)
            next_act = td_next["agents", "action_normalized"]  # [B, A]
            next_log_prob = td_next["sample_log_prob"].reshape(-1)  # [B]

            sa_next = torch.cat([next_feat, next_act], dim=-1)
            q1_next = self.q1_target(sa_next).squeeze(-1)
            q2_next = self.q2_target(sa_next).squeeze(-1)
            min_q_next = torch.min(q1_next, q2_next) - alpha * next_log_prob

            target_q = rew + (1.0 - done) * self.gamma * min_q_next
            target_q = target_q.detach()

        # Critic update
        sa = torch.cat([feat, act], dim=-1)
        q1 = self.q1(sa).squeeze(-1)
        q2 = self.q2(sa).squeeze(-1)

        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.q1_optim.zero_grad(set_to_none=True)
        self.q2_optim.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.q1_optim.step()
        self.q2_optim.step()

        # Actor and temperature update
        td_pi = TensorDict({"_feature": feat}, batch_size=[batch_size])
        self.actor(td_pi)
        act_pi = td_pi["agents", "action_normalized"]  # [B, A]
        log_prob_pi = td_pi["sample_log_prob"].reshape(-1)  # [B]

        sa_pi = torch.cat([feat, act_pi], dim=-1)
        q1_pi = self.q1(sa_pi).squeeze(-1)
        q2_pi = self.q2(sa_pi).squeeze(-1)
        q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (self.alpha.detach() * log_prob_pi - q_pi).mean()

        self.actor_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optim.step()

        alpha_loss = -(self.log_alpha * (log_prob_pi.detach() + self.target_entropy)).mean()

        self.alpha_optim.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optim.step()

        # Polyak averaging of targets
        with torch.no_grad():
            for p, p_targ in zip(self.q1.parameters(), self.q1_target.parameters()):
                p_targ.data.lerp_(p.data, self.tau)
            for p, p_targ in zip(self.q2.parameters(), self.q2_target.parameters()):
                p_targ.data.lerp_(p.data, self.tau)

        return (
            float(actor_loss.item()),
            float(critic_loss.item()),
            float(alpha_loss.item()),
        )
