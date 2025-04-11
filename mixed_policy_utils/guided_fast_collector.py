from fsrl.data import FastCollector
import numpy as np
from tianshou.data import Batch
from typing import Union, List, Optional, Dict, Any


class GuidedFastCollector(FastCollector):
    def _reset_env_with_ids(
        self,
        local_ids: Union[List[int], np.ndarray],
        global_ids: Union[List[int], np.ndarray],
        gym_reset_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Call parent logic
        super()._reset_env_with_ids(local_ids, global_ids, gym_reset_kwargs)

        # Call the policy reset hook if available
        if hasattr(self.policy, "reset_envs") and callable(
            getattr(self.policy, "reset_envs")
        ):
            self.policy.reset_envs(global_ids)
