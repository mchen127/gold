# eval/eval_iql.py
from dataclasses import asdict, dataclass
import pyrallis
import torch
import numpy as np
from osrl.algorithms import IQL
from dsrl.offline_env import OfflineEnvWrapper, wrap_env
from osrl.common.exp_util import load_config_and_model, seed_all


@dataclass
class EvalConfig:
    path: str = "log/.../checkpoint/model.pt"
    eval_episodes: int = 20
    best: bool = False
    device: str = "cpu"
    threads: int = 4


@pyrallis.wrap()
def eval(args: EvalConfig):
    # Load configuration and the saved model
    cfg, model_state = load_config_and_model(args.path, args.best)
    seed_all(cfg["seed"])
    if args.device == "cpu":
        torch.set_num_threads(args.threads)

    import gymnasium as gym

    env = wrap_env(gym.make(cfg["task"]), reward_scale=cfg["reward_scale"])
    env = OfflineEnvWrapper(env)
    env.set_target_cost(cfg["cost_limit"])

    # Initialize IQL model and load state
    iql_model = IQL(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        max_action=env.action_space.high[0],
        hidden_sizes=cfg["hidden_sizes"],
        gamma=cfg["gamma"],
        tau=cfg["tau"],
        expectile=cfg["expectile"],
        beta=cfg["beta"],
        num_q=cfg["num_q"],
        device=args.device,
    )
    iql_model.load_state_dict(model_state["model_state"])
    iql_model.to(args.device)

    # Evaluation loop
    iql_model.eval()
    episode_rewards = []
    for _ in range(args.eval_episodes):
        obs = env.reset()[0]
        ep_reward = 0.0
        done = False
        while not done:
            action = iql_model.act(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        episode_rewards.append(ep_reward)
    avg_reward = np.mean(episode_rewards)
    print(f"Evaluation Reward: {avg_reward}")


if __name__ == "__main__":
    eval()
