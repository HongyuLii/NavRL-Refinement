import datetime
import os
from typing import Any, Dict, Optional, Tuple

import hydra
import numpy as np
import torch
from env import NavigationEnv  # same directory as train.py
from gymnasium import Env, spaces
from omni.isaac.kit import SimulationApp
from omni_drones.controllers import LeePositionController
from omni_drones.utils.torchrl.transforms import VelController
from stable_baselines3 import SAC
from torchrl.envs.transforms import Compose, TransformedEnv

FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cfg")


class NavRLGymWrapper(Env):
    """Gymnasium wrapper around a TorchRL TransformedEnv (NavigationEnv).

    It:
      - Keeps an internal TensorDict for all envs.
      - Exposes only env index 0 as a single Gym environment.
      - Flattens the observation dict into a 1D float32 vector.
    """

    metadata = {"render_modes": []}

    def __init__(self, trl_env: TransformedEnv, device: str = "cuda:0", env_index: int = 0):
        super().__init__()
        self._env = trl_env
        self.device = torch.device(device)
        self.env_index = env_index
        self._td = None  # last TensorDict

        # Initial reset to infer obs space
        self._td = self._env.reset()
        obs_vec = self._td_to_obs(self._td)[self.env_index]
        obs_np = obs_vec.detach().cpu().numpy().astype(np.float32)

        # Observation space: unbounded continuous Box
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_np.shape,
            dtype=np.float32,
        )

        # Action space: continuous Box, assume [-1, 1] after VelController
        act_spec = self._env.action_spec[("agents", "action")]
        # act_spec.shape is (num_envs, act_dim, ...) – we only care about per-env last dimension
        act_shape = act_spec.shape[1:]
        if len(act_shape) == 0:
            # scalar action
            act_shape = (1,)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=act_shape,
            dtype=np.float32,
        )

    # ---- helper to flatten TensorDict observation ----
    def _td_to_obs(self, td) -> torch.Tensor:
        """Flatten multi-part observation into [num_envs, obs_dim]."""
        obs = td["agents", "observation"]

        parts = []

        # state: [num_envs, 8]
        parts.append(obs["state"])

        # lidar: [num_envs, 1, H, V] -> [num_envs, H*V]
        lidar = obs["lidar"]
        parts.append(lidar.reshape(lidar.shape[0], -1))

        # direction: [num_envs, 1, 3] -> [num_envs, 3]
        direction = obs["direction"]
        parts.append(direction.reshape(direction.shape[0], -1))

        # dynamic_obstacle: [num_envs, 1, N, D] -> [num_envs, N*D] (if present)
        if "dynamic_obstacle" in obs.keys(True, True):
            dyn = obs["dynamic_obstacle"]
            parts.append(dyn.reshape(dyn.shape[0], -1))

        return torch.cat(parts, dim=-1)

    # ---- Gymnasium API ----
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            # NavigationEnv already uses cfg.seed; this is just for completeness
            self._env.set_seed(seed)
        self._td = self._env.reset()
        obs_vec = self._td_to_obs(self._td)[self.env_index]
        obs_np = obs_vec.detach().cpu().numpy().astype(np.float32)
        info: Dict[str, Any] = {}
        return obs_np, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Ensure torch tensor on the right device, shaped per-env
        action = np.asarray(action, dtype=np.float32)
        action_t = torch.as_tensor(action, device=self.device)

        # We keep a full TensorDict for all envs; only overwrite index env_index
        td = self._td.clone()
        act_td = td["agents", "action"]
        # Broadcast scalar -> correct shape if needed
        if act_td[self.env_index].shape != action_t.shape:
            action_t = action_t.view_as(act_td[self.env_index])

        act_td[self.env_index] = action_t
        td["agents", "action"] = act_td

        next_td = self._env.step(td)
        self._td = next_td

        # Build next observation
        obs_vec = self._td_to_obs(next_td)[self.env_index]
        obs_np = obs_vec.detach().cpu().numpy().astype(np.float32)

        # Rewards and termination flags
        reward = float(next_td["agents", "reward"][self.env_index].item())
        terminated = bool(next_td["terminated"][self.env_index].item())
        truncated = bool(next_td["truncated"][self.env_index].item())

        info: Dict[str, Any] = {
            "terminated": terminated,
            "truncated": truncated,
        }

        return obs_np, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        # Isaac Sim rendering is driven by SimulationApp; no-op here.
        return None

    def close(self) -> None:
        # Underlying Isaac env cleanup is handled by SimulationApp
        pass


@hydra.main(config_path=FILE_PATH, config_name="train", version_base=None)
def main(cfg) -> None:
    # Simulation App (same as train.py)
    sim_app = SimulationApp({"headless": cfg.headless, "anti_aliasing": 1})

    # Instantiate Isaac/OmniDrones env
    env = NavigationEnv(cfg)

    # Same transforms as train.py: LeePositionController + VelController
    controller = LeePositionController(9.81, env.drone.params).to(cfg.device)
    vel_transform = VelController(controller, yaw_control=False)
    transformed_env = TransformedEnv(env, Compose(vel_transform)).train()
    transformed_env.set_seed(cfg.seed)

    # Wrap in Gymnasium
    gym_env = NavRLGymWrapper(transformed_env, device=cfg.device, env_index=0)

    # SB3 SAC hyperparameters mapped from sac.yaml where sensible
    # (you can refine this mapping as needed)
    algo_cfg = cfg.algo  # sac.yaml
    total_timesteps = int(cfg.max_frame_num)  # 12e8 in your config

    model = SAC(
        policy="MlpPolicy",
        env=gym_env,
        learning_rate=algo_cfg.actor.learning_rate,
        buffer_size=int(algo_cfg.replay_capacity),
        batch_size=int(algo_cfg.batch_size),
        tau=float(algo_cfg.critic.tau),
        gamma=0.99,
        train_freq=1,
        gradient_steps=-1,  # one gradient step per env step (default-style)
        ent_coef="auto",
        verbose=1,
        seed=cfg.seed,
        device=cfg.device,
    )

    # Simple training loop (no WandB here yet; can be added via callbacks)
    model.learn(total_timesteps=total_timesteps)

    # Save final SB3 model
    timestamp = datetime.datetime.now().strftime("%m-%d_%H-%M")
    ckpt_dir = os.path.join(os.getcwd(), "sb3_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    model_path = os.path.join(ckpt_dir, f"sac_navrl_{timestamp}.zip")
    model.save(model_path)
    print(f"[NavRL-SB3]: Saved model to {model_path}")

    gym_env.close()
    sim_app.close()


if __name__ == "__main__":
    main()
