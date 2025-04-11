import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pprint
from dataclasses import asdict
import uuid

import bullet_safety_gym

try:
    import safety_gymnasium
except ImportError:
    print("safety_gymnasium is not found.")
import gymnasium as gym
import pyrallis
import torch
import torch.nn as nn
from tianshou.data import VectorReplayBuffer
from tianshou.env import BaseVectorEnv, ShmemVectorEnv, SubprocVectorEnv
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb
from torch.distributions import Independent, Normal

from train.train_cvpo import TASK_TO_CFG
from fsrl.policy import CVPO
from fsrl.trainer import OffpolicyTrainer
from fsrl.utils import TensorboardLogger, WandbLogger
from fsrl.utils.exp_util import auto_name, seed_all
from fsrl.utils.net.common import ActorCritic
from fsrl.utils.net.continuous import DoubleCritic, SingleCritic
from fsrl.data import FastCollector

from mixed_policy_utils.guided_train_cfg import GuidedTrainCfg
from mixed_policy_utils.bc_policy_wrapper import BCPolicyWrapper
from mixed_policy_utils.mixed_policy_wrapper import MixedPolicyWrapper
from mixed_policy_utils.guided_fast_collector import GuidedFastCollector
from osrl.algorithms import BC
from osrl.common.exp_util import load_config_and_model


@pyrallis.wrap()
def train(args: GuidedTrainCfg):
    seed_all(args.seed)
    torch.set_num_threads(args.thread)

    task = args.task
    default_cfg = TASK_TO_CFG[task]() if task in TASK_TO_CFG else GuidedTrainCfg()
    if args.use_default_cfg:
        default_cfg.task = args.task
        default_cfg.seed = args.seed
        default_cfg.device = args.device
        default_cfg.logdir = args.logdir
        default_cfg.project = args.project
        default_cfg.group = args.group
        default_cfg.suffix = args.suffix
        args = default_cfg

    cfg = asdict(args)
    default_cfg = asdict(default_cfg)
    if args.name is None:
        args.name = auto_name(default_cfg, cfg, args.prefix, args.suffix)
    else:
        args.name = args.prefix + "_" + args.name
        args.name = f"{args.name}-{str(uuid.uuid4())[:4]}"
    if args.group is None:
        args.group = args.task + "-cost-" + str(int(args.cost_limit))
    if args.logdir is not None:
        args.logdir = os.path.join(args.logdir, args.project, args.group)
    logger = WandbLogger(cfg, args.project, args.group, args.name, args.logdir)
    logger.save_config(cfg, verbose=args.verbose)

    training_num = min(args.training_num, args.episode_per_collect)
    worker = eval(args.worker)
    train_envs = worker([lambda: gym.make(args.task) for _ in range(training_num)])
    test_envs = worker([lambda: gym.make(args.task) for _ in range(args.testing_num)])

    env = gym.make(args.task)
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    max_action = env.action_space.high[0]

    assert hasattr(env.spec, "max_episode_steps")

    net = Net(state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    actor = ActorProb(
        net,
        action_shape,
        max_action=max_action,
        device=args.device,
        conditioned_sigma=args.conditioned_sigma,
        unbounded=args.unbounded,
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)

    critics = []
    for _ in range(2):
        if args.double_critic:
            net1 = Net(
                state_shape,
                action_shape,
                hidden_sizes=args.hidden_sizes,
                concat=True,
                device=args.device,
            )
            net2 = Net(
                state_shape,
                action_shape,
                hidden_sizes=args.hidden_sizes,
                concat=True,
                device=args.device,
            )
            critics.append(DoubleCritic(net1, net2, device=args.device).to(args.device))
        else:
            net_c = Net(
                state_shape,
                action_shape,
                hidden_sizes=args.hidden_sizes,
                concat=True,
                device=args.device,
            )
            critics.append(SingleCritic(net_c, device=args.device).to(args.device))

    critic_optim = torch.optim.Adam(
        nn.ModuleList(critics).parameters(), lr=args.critic_lr
    )

    if not args.conditioned_sigma:
        torch.nn.init.constant_(actor.sigma_param, -0.5)
    actor_critic = ActorCritic(actor, critics)
    for m in actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)

    if args.last_layer_scale:
        for m in actor.mu.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.zeros_(m.bias)
                m.weight.data.copy_(0.01 * m.weight.data)

    def dist(*logits):
        return Independent(Normal(*logits), 1)

    policy = CVPO(
        actor=actor,
        critics=critics,
        actor_optim=actor_optim,
        critic_optim=critic_optim,
        logger=logger,
        action_space=env.action_space,
        dist_fn=dist,
        max_episode_steps=env.spec.max_episode_steps,
        cost_limit=args.cost_limit,
        tau=args.tau,
        gamma=args.gamma,
        n_step=args.n_step,
        estep_iter_num=args.estep_iter_num,
        estep_kl=args.estep_kl,
        estep_dual_max=args.estep_dual_max,
        estep_dual_lr=args.estep_dual_lr,
        sample_act_num=args.sample_act_num,
        mstep_iter_num=args.mstep_iter_num,
        mstep_kl_mu=args.mstep_kl_mu,
        mstep_kl_std=args.mstep_kl_std,
        mstep_dual_max=args.mstep_dual_max,
        mstep_dual_lr=args.mstep_dual_lr,
        deterministic_eval=args.deterministic_eval,
        action_scaling=args.action_scaling,
        action_bound_method=args.action_bound_method,
        lr_scheduler=None,
    )

    cfg, model = load_config_and_model(args.offline_model_path, args.best)
    bc_state_dim = env.observation_space.shape[0]
    if cfg["bc_mode"] == "multi-task":
        bc_state_dim += 1
    bc_model = BC(
        state_dim=bc_state_dim,
        action_dim=env.action_space.shape[0],
        max_action=env.action_space.high[0],
        a_hidden_sizes=cfg["a_hidden_sizes"],
        episode_len=cfg["episode_len"],
        device=args.device,
    )
    bc_model.load_state_dict(model["model_state"])
    bc_model.to(args.device)
    offline_policy = BCPolicyWrapper(bc_model)

    mixed_policy = MixedPolicyWrapper(
        offline_policy=offline_policy, online_policy=policy, h=args.h
    )

    train_collector = GuidedFastCollector(
        mixed_policy,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=False,
    )
    test_collector = FastCollector(policy, test_envs)

    def stop_fn(reward, cost):
        return reward > args.reward_threshold and cost < args.cost_limit

    def checkpoint_fn():
        return {"model": policy.state_dict()}

    if args.save_ckpt:
        logger.setup_checkpoint_fn(checkpoint_fn)

    trainer = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epoch,
        batch_size=args.batch_size,
        cost_limit=args.cost_limit,
        step_per_epoch=args.step_per_epoch,
        update_per_step=args.update_per_step,
        episode_per_test=args.testing_num,
        episode_per_collect=args.episode_per_collect,
        stop_fn=stop_fn,
        logger=logger,
        resume_from_log=args.resume,
        save_model_interval=args.save_interval,
        verbose=args.verbose,
    )

    for epoch, epoch_stat, info in trainer:
        logger.store(tab="train", cost_limit=args.cost_limit)
        print(f"Epoch: {epoch}")
        print(info)

    # mixed_policy.save_log_to_csv(f"{args.name}_policy_log.csv")

    if __name__ == "__main__":
        pprint.pprint(info)
        env = gym.make(args.task)
        policy.eval()
        collector = FastCollector(policy, env)
        result = collector.collect(n_episode=10, render=args.render)
        rews, lens, cost = result["rew"], result["len"], result["cost"]
        print(f"Final eval reward: {rews.mean()}, cost: {cost}, length: {lens.mean()}")

        policy.train()
        collector = FastCollector(policy, env)
        result = collector.collect(n_episode=10, render=args.render)
        rews, lens, cost = result["rew"], result["len"], result["cost"]
        print(f"Final train reward: {rews.mean()}, cost: {cost}, length: {lens.mean()}")


if __name__ == "__main__":
    train()
