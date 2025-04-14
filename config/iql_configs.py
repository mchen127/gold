# configs/iql_configs.py
from dataclasses import asdict, dataclass
from typing import List, Optional
from pyrallis import field


@dataclass
class IQLTrainConfig:
    # WandB parameters
    project: str = "OSRL-baselines"
    group: Optional[str] = None
    name: Optional[str] = None
    prefix: Optional[str] = "IQL"
    suffix: Optional[str] = ""
    logdir: Optional[str] = "logs"
    verbose: bool = True
    # Training parameters
    task: str = "OfflineCarCircle-v0"
    seed: int = 0
    device: str = "cpu"
    threads: int = 4
    reward_scale: float = 0.1
    batch_size: int = 512
    update_steps: int = 100_000
    num_workers: int = 8
    # Model parameters
    hidden_sizes: List[int] = field(default=[256, 256], is_mutable=True)
    gamma: float = 0.99
    tau: float = 0.005
    expectile: float = 0.7
    beta: float = 3.0
    num_q: int = 1
    # Evaluation parameters
    eval_episodes: int = 10
    eval_every: int = 2500
    # Optimizer parameters
    actor_lr: float = 0.001
    critic_lr: float = 0.001
    value_lr: float = 0.001


@dataclass
class IQLCarCircleConfig(IQLTrainConfig):
    pass


IQL_DEFAULT_CONFIG = {
    "OfflineCarCircle-v0": IQLCarCircleConfig,
    # Extend with additional task-specific configurations as needed.
}
