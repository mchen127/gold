import os
import csv
import sys

from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import dsrl
import numpy as np
import pyrallis
import torch
from pyrallis import field

from osrl.algorithms import BC, BCTrainer
from osrl.common.exp_util import load_config_and_model, seed_all


@dataclass
class EvalConfig:
    model_path: str = "log/.../checkpoint/model.pt"
    noise_scale: List[float] = None
    costs: List[float] = field(default=[1, 10, 20, 30, 40], is_mutable=True)
    eval_episodes: int = 20
    best: bool = False
    device: str = "cpu"
    threads: int = 4
    output_path: str = "log/.../eval_result.csv"


@pyrallis.wrap()
def eval(args: EvalConfig):
    results = []
    print(f"Evaluating model: {args.model_path}") 

    cfg, model = load_config_and_model(args.model_path, args.best)
    seed_all(cfg["seed"])
    if args.device == "cpu":
        torch.set_num_threads(args.threads)

    if "Metadrive" in cfg["task"]:
        import gym
    else:
        import gymnasium as gym  # noqa

    env = gym.make(cfg["task"])
    env.set_target_cost(cfg["cost_limit"])

    # model & optimizer & scheduler setup
    state_dim = env.observation_space.shape[0]
    if cfg["bc_mode"] == "multi-task":
        state_dim += 1
    bc_model = BC(
        state_dim=state_dim,
        action_dim=env.action_space.shape[0],
        max_action=env.action_space.high[0],
        a_hidden_sizes=cfg["a_hidden_sizes"],
        episode_len=cfg["episode_len"],
        device=args.device,
    )
    bc_model.load_state_dict(model["model_state"])
    bc_model.to(args.device)

    trainer = BCTrainer(
        bc_model,
        env,
        bc_mode=cfg["bc_mode"],
        cost_limit=cfg["cost_limit"],
        device=args.device,
    )

    if cfg["bc_mode"] == "multi-task":
        for target_cost in args.costs:
            env.set_target_cost(target_cost)
            trainer.set_target_cost(target_cost)
            ret, cost, length = trainer.evaluate(args.eval_episodes)
            normalized_ret, normalized_cost = env.get_normalized_score(ret, cost)
            print(
                f"Eval reward: {ret}, normalized reward: {normalized_ret}; target cost {target_cost}, real cost {cost}, normalized cost: {normalized_cost}"
            )
            results.append(
                [
                    cfg["task"],
                    cfg["name"],
                    args.best,
                    target_cost,
                    ret,
                    normalized_ret,
                    cost,
                    normalized_cost,
                    length,
                ]
            )
    else:
        ret, cost, length = trainer.evaluate(args.eval_episodes)
        normalized_ret, normalized_cost = env.get_normalized_score(ret, cost)
        print(
            f"Eval reward: {ret}, normalized reward: {normalized_ret}; cost: {cost}, normalized cost: {normalized_cost}; length: {length}"
        )
        results.append(
            [
                cfg["task"],
                cfg["name"],
                cfg["cost_limit"],
                cfg["seed"],
                args.best,
                "-",
                ret,
                normalized_ret,
                cost,
                normalized_cost,
                length,
            ]
        )

    # Save results to CSV
    save_results_to_csv(args.output_path, results)


def save_results_to_csv(csv_path, results):
    """Save evaluation results to CSV."""
    header = [
        "Task",
        "Model",
        "Cost Limit",
        "Seed",
        "Best",
        "Target Cost",
        "Reward",
        "Normalized Reward",
        "Cost",
        "Normalized Cost",
        "Length",
    ]
    write_header = not os.path.exists(
        csv_path
    )  # Only write the header if the file is new

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerows(results)

    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    eval()
