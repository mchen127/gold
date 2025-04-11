import numpy as np
import torch
from tianshou.data import Batch


class BCPolicyWrapper(torch.nn.Module):
    """
    A lightweight wrapper to make a BC model compatible with collectors expecting
    a forward-callable torch.nn.Module.

    This wrapper calls `bc_model.act(obs)` internally and handles batched or
    single observations, as well as Tianshou-style Batch inputs.
    """

    def __init__(self, bc_model):
        super().__init__()
        self.bc_model = bc_model

    def forward(self, obs: Batch, state=None, info=None):
        # Unwrap the raw obs
        raw_obs = obs.obs if hasattr(obs, "obs") else obs

        # Convert to NumPy if needed
        if not isinstance(raw_obs, np.ndarray):
            raw_obs = np.array(raw_obs)

        # Handle single obs (1D) or batched obs (2D)
        if raw_obs.ndim == 1:
            act = self.bc_model.act(raw_obs)  # [obs_dim]
            return Batch(act=act)
        else:
            acts = [self.bc_model.act(o) for o in raw_obs]  # [[act], [act], ...]
            return Batch(act=np.stack(acts))

    def to(self, device):
        self.bc_model.to(device)
        return self
