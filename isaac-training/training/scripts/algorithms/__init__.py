from .ppo import PPO
from .sac import SAC


def make_algo(algo_cfg, observation_spec, action_spec, device):
    name = algo_cfg.name.lower()
    if name == "ppo":
        return PPO(algo_cfg, observation_spec, action_spec, device)
    elif name == "sac":
        return SAC(algo_cfg, observation_spec, action_spec, device)
    else:
        raise ValueError(f"Unknown algorithm '{algo_cfg.name}'")
