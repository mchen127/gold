# train/train_iql.py
import os
import types
from dataclasses import asdict
import numpy as np
import pyrallis
import torch
from torch.utils.data import DataLoader
from tqdm.auto import trange

from iql import IQL  # Make sure IQL is available in the algorithms folder
from osrl.common.exp_util import auto_name, seed_all
from osrl.common import TransitionDataset  # Reuse dataset loader if available
from config.iql_configs import IQLTrainConfig, IQL_DEFAULT_CONFIG
from fsrl.utils import WandbLogger


@pyrallis.wrap()
def train(args: IQLTrainConfig):
    # Update configuration by comparing with the default configuration.
    cfg, old_cfg = asdict(args), asdict(IQLTrainConfig())
    differing_values = {key: cfg[key] for key in cfg.keys() if cfg[key] != old_cfg[key]}
    cfg = asdict(IQL_DEFAULT_CONFIG[args.task]())
    cfg.update(differing_values)
    args = types.SimpleNamespace(**cfg)

    # Set up logger
    default_cfg = asdict(IQL_DEFAULT_CONFIG[args.task]())
    if args.name is None:
        args.name = auto_name(default_cfg, cfg, args.prefix, args.suffix)
    if args.group is None:
        args.group = args.task + "-iql"
    if args.logdir is not None:
        args.logdir = os.path.join(args.logdir, args.group, args.name)
    logger = WandbLogger(cfg, args.project, args.group, args.name, args.logdir)
    logger.save_config(cfg, verbose=args.verbose)

    # Set seed and device configuration
    seed_all(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(args.threads)

    # Initialize environment and get the offline dataset
    import gymnasium as gym

    env = gym.make(args.task)
    data = env.get_dataset()
    env.set_target_cost(args.cost_limit)

    # Pre-process dataset if needed (using your environment's method)
    data = env.pre_process_data(data, args.outliers_percent, args.noise_scale)

    # Wrap environment for offline RL
    from dsrl.offline_env import wrap_env, OfflineEnvWrapper

    env = wrap_env(env, reward_scale=args.reward_scale)
    env = OfflineEnvWrapper(env)

    # Initialize the IQL model
    model = IQL(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        max_action=env.action_space.high[0],
        hidden_sizes=args.hidden_sizes,
        gamma=args.gamma,
        tau=args.tau,
        expectile=args.expectile,
        beta=args.beta,
        num_q=args.num_q,
        device=args.device,
    )
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    # Define a checkpoint function
    def checkpoint_fn():
        return {"model_state": model.state_dict()}

    logger.setup_checkpoint_fn(checkpoint_fn)

    # Define the IQL trainer class
    class IQLTrainer:
        def __init__(self, model, env, logger, actor_lr, critic_lr, value_lr, device):
            self.model = model
            self.env = env
            self.logger = logger
            self.device = device
            self.actor_optim = torch.optim.Adam(model.actor.parameters(), lr=actor_lr)
            self.q_optim = torch.optim.Adam(model.q_net.parameters(), lr=critic_lr)
            self.value_optim = torch.optim.Adam(
                model.value_net.parameters(), lr=value_lr
            )

        def train_one_step(
            self, observations, actions, rewards, next_observations, done
        ):
            # Update Q network
            loss_q = self.model.q_loss(
                observations, actions, rewards, next_observations, done
            )
            self.q_optim.zero_grad()
            loss_q.backward()
            self.q_optim.step()

            # Update value network
            loss_value = self.model.value_loss(observations, actions)
            self.value_optim.zero_grad()
            loss_value.backward()
            self.value_optim.step()

            # Update actor network
            loss_actor = self.model.actor_loss(observations)
            self.actor_optim.zero_grad()
            loss_actor.backward()
            self.actor_optim.step()

            # Soft update target networks
            self.model.sync_target()

            return {
                "loss/q": loss_q.item(),
                "loss/value": loss_value.item(),
                "loss/actor": loss_actor.item(),
            }

        def evaluate(self, eval_episodes):
            self.model.eval()
            episode_rewards = []
            for _ in range(eval_episodes):
                obs = self.env.reset()[0]
                ep_reward = 0.0
                done = False
                while not done:
                    action = self.model.act(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = self.env.step(action)
                    ep_reward += reward
                    done = terminated or truncated
                episode_rewards.append(ep_reward)
            self.model.train()
            return np.mean(episode_rewards)

    trainer = IQLTrainer(
        model,
        env,
        logger,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        value_lr=args.value_lr,
        device=args.device,
    )

    # Build a dataloader from the transition dataset (reusing your TransitionDataset class)
    dataset = TransitionDataset(data, reward_scale=args.reward_scale)
    trainloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    trainloader_iter = iter(trainloader)

    best_reward = -np.inf
    for step in trange(args.update_steps, desc="Training"):
        batch = next(trainloader_iter)
        obs, next_obs, actions, rewards, _, done = [b.to(args.device) for b in batch]
        stats = trainer.train_one_step(obs, actions, rewards, next_obs, done)
        logger.store(**stats)

        if (step + 1) % args.eval_every == 0 or step == args.update_steps - 1:
            eval_reward = trainer.evaluate(args.eval_episodes)
            logger.store(tab="eval", Reward=eval_reward)
            logger.save_checkpoint()
            logger.write(step)
        else:
            logger.write_without_reset(step)


if __name__ == "__main__":
    train()
