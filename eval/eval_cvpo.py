import csv
import os
import os.path as osp
import yaml
from dataclasses import asdict, dataclass
import numpy as np
import bullet_safety_gym

try:
    import safety_gymnasium
except ImportError:
    print("safety_gymnasium is not found.")
import gymnasium as gym
import pyrallis
import torch
from tianshou.env import BaseVectorEnv, DummyVectorEnv, ShmemVectorEnv
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb
from torch.distributions import Independent, Normal

from fsrl.data import FastCollector
from fsrl.policy import CVPO
from fsrl.utils import BaseLogger
from fsrl.utils.exp_util import load_config_and_model, seed_all
from fsrl.utils.net.continuous import DoubleCritic, SingleCritic


@dataclass
class EvalConfig:
    model_path: str = "logs"
    output_path: str = "log/.../eval_result.csv"
    best: bool = False
    eval_episodes: int = 20
    worker: BaseVectorEnv = ShmemVectorEnv
    device: str = "cpu"
    render: bool = False

def load_config_and_model_considering_device(path: str, best: bool = False, device: str = "cpu"):
    """
    Load the configuration and model from the specified path.
    :param path: Path to the model directory.
    :param best: Whether to load the best model.
    :param device: Device to load the model on (e.g., "cpu", "cuda").
    :return: Tuple of configuration and model.
    """
    if osp.exists(path):
        config_file = osp.join(path, "config.yaml")
        print(f"load config from {config_file}")
        with open(config_file) as f:
            config = yaml.load(f.read(), Loader=yaml.FullLoader)
        model_file = "model.pt"
        if best:
            model_file = "model_best.pt"
        model_path = osp.join(path, "checkpoint/" + model_file)
        print(f"load model from {model_path}")
        model = torch.load(model_path, map_location=torch.device(device=device))
        return config, model
    else:
        raise ValueError(f"{path} doesn't exist!")


@pyrallis.wrap()
def eval(args: EvalConfig):
    cfg, model = load_config_and_model_considering_device(args.model_path, args.best, args.device)
    # Seed
    seed_all(cfg["seed"])
    torch.set_num_threads(cfg["thread"])

    logger = BaseLogger()

    env = gym.make(cfg["task"])
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    max_action = env.action_space.high[0]

    # Model setup
    net = Net(state_shape, hidden_sizes=cfg["hidden_sizes"], device=args.device)
    actor = ActorProb(
        net,
        action_shape,
        max_action=max_action,
        device=args.device,
        unbounded=cfg["unbounded"],
        conditioned_sigma=cfg["conditioned_sigma"],  # ✅ Ensures sigma is a model
    ).to(args.device)

    critics = []
    for i in range(2):
        if cfg["double_critic"]:
            net1 = Net(
                state_shape,
                action_shape,
                hidden_sizes=cfg["hidden_sizes"],
                concat=True,
                device=args.device,
            )
            net2 = Net(
                state_shape,
                action_shape,
                hidden_sizes=cfg["hidden_sizes"],
                concat=True,
                device=args.device,
            )
            critics.append(DoubleCritic(net1, net2, device=args.device).to(args.device))
        else:
            net_c = Net(
                state_shape,
                action_shape,
                hidden_sizes=cfg["hidden_sizes"],
                concat=True,
                device=args.device,
            )
            critics.append(SingleCritic(net_c, device=args.device).to(args.device))

    def dist(*logits):
        return Independent(Normal(*logits), 1)

    policy = CVPO(
        actor=actor,
        critics=critics,
        actor_optim=None,
        critic_optim=None,
        logger=logger,
        dist_fn=dist,
        cost_limit=cfg["cost_limit"],
        action_space=env.action_space,
        max_episode_steps=env.spec.max_episode_steps,
    )
    # Print actor architecture to verify `sigma` structure
    print("🚀 Actor Architecture in CVPO Evaluation:")
    print(policy.actor)

    # Print sigma layer details
    print("\n🚀 Sigma Layer Parameters:")
    for name, param in policy.actor.named_parameters():
        if "sigma" in name:
            print(f"{name}: {param.shape}, requires_grad={param.requires_grad}")

    policy.load_state_dict(model["model"])
    policy.eval()

    # # Collector
    # test_envs = args.worker(
    #     [lambda: gym.make(cfg["task"]) for _ in range(args.eval_episodes)]
    # )
    # eval_collector = FastCollector(policy, test_envs)
    # result = eval_collector.collect(n_episode=args.eval_episodes, render=args.render)

    # rews, lens, costs = result["rew"], result["len"], result["cost"]
    # print(f"Eval reward: {rews}, cost: {costs}, length: {lens}")

    batch_size = 5  # 一次啟動 5 個環境
    total_episodes = args.eval_episodes  # 總共需要 50 個 episodes
    num_batches = total_episodes // batch_size  # 50 / 5 = 10 批次

    all_rewards, all_lengths, all_costs = [], [], []

    for batch in range(num_batches):
        print(f"Running batch {batch+1}/{num_batches}...")

        test_envs = args.worker(
            [lambda: gym.make(cfg["task"]) for _ in range(batch_size)]
        )

        eval_collector = FastCollector(policy, test_envs)
        result = eval_collector.collect(n_episode=batch_size, render=args.render)

        all_rewards.append(result["rew"])
        all_lengths.append(result["len"])
        all_costs.append(result["cost"])

        test_envs.close()  # 關閉環境，釋放資源

    normalized_costs = [cost / (cfg["cost_limit"] + 1e-8) for cost in all_costs]
    rews, lens, costs, normalized_cost = np.mean(all_rewards), np.mean(all_lengths), np.mean(all_costs), np.mean(normalized_costs)
    # 最後統計 50 個 episodes 的結果
    print(f"Final Eval reward: {rews}, cost: {costs}, length: {lens}")

    # Save to CSV
    try:
        # save_results_to_csv(
        #     args.output_path, cfg["task"], cfg["name"], cfg["cost_limit"], cfg["seed"], args.best, rews, costs, normalized_cost, lens
        # )
        save_all_results_to_csv(
            args.output_path, cfg["task"], cfg["name"], cfg["cost_limit"], cfg["seed"], args.best, all_rewards, all_costs, normalized_costs, all_lengths
        )
    except Exception as e:
        import traceback
        print("❌ Error saving results to CSV:")
        print(traceback.format_exc())


def save_results_to_csv(csv_path, task_name, model_name, cost_limit, seed, is_best, reward, cost, normalized_cost, length):
    """Append evaluation results to a CSV file."""
    header = ["Task", "Model", "Cost Limit", "Seed", "Best", "Reward", "Cost", "Normalized Cost", "Length"]
    write_header = not os.path.exists(csv_path)  # Only write header if the file is new
    # Ensure the directory exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    # Append results to the CSV file
    # Open the CSV file in append mode
    # and write the header if it's the first time
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow([task_name, model_name, cost_limit, seed, is_best, reward, cost, normalized_cost, length])

    print(f"Results saved to {csv_path}")

def save_all_results_to_csv(csv_path, task_name, model_name, cost_limit, seed, is_best, all_rewards, all_costs, normalized_costs, all_lengths):
    """Append evaluation results to a CSV file."""
    header = ["Task", "Model", "Cost Limit", "Seed", "Best", "Reward", "Cost", "Normalized Cost", "Length"]
    write_header = not os.path.exists(csv_path)  # Only write header if the file is new
    # Ensure the directory exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    # Append results to the CSV file
    # Open the CSV file in append mode
    # and write the header if it's the first time
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        for reward, cost, normalized_cost, length in zip(all_rewards, all_costs, normalized_costs, all_lengths):
            # Write each result to the CSV file
            writer.writerow([task_name, model_name, cost_limit, seed, is_best, reward, cost, normalized_cost, length])

    print(f"Results saved to {csv_path}")

if __name__ == "__main__":
    eval()
