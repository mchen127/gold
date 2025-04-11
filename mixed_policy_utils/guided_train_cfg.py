from dataclasses import dataclass
from fsrl.config.cvpo_cfg import TrainCfg  # Import base config


@dataclass
class GuidedTrainCfg(TrainCfg):
    offline_model_path: str = (
        "/home/mc/gold/logs/OfflineCarCircle1Gymnasium-v0-cost-20/BC_bc_modefrontier_cost20_seed30-0254/BC_bc_modefrontier_cost20_seed30-0254"  # Path to pretrained offline agent
    )
    best: bool = True  # Load the best checkpoint
    h: int = 100  # Number of offline agent steps per episode
