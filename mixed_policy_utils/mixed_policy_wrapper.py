from typing import Optional
import numpy as np
import torch
import pandas as pd
from tianshou.data import Batch


class MixedPolicyWrapper:
    def __init__(self, offline_policy, online_policy, h):
        self.offline_policy = offline_policy
        self.online_policy = online_policy
        self.h = h
        self.step_counters = {}  # use dict with env_id as key
        self.log_records = []  # for tracking who made each decision

    def __call__(self, data: Batch, state: Optional[torch.Tensor] = None) -> Batch:
        obs = data.obs
        result = Batch()

        # Get env_id per sample (or fallback to range if not present)
        env_ids = data.env_id if hasattr(data, "env_id") else np.arange(len(obs))

        acts, policies, states, sources = [], [], [], []

        for i in range(len(obs)):
            env_id = env_ids[i]
            self.step_counters.setdefault(env_id, 0)

            if self.step_counters[env_id] < self.h:
                act = self.offline_policy(obs[i : i + 1]).act[0]
                policy_data = Batch()
                state_data = None
                source = "offline"
            else:
                online_result = self.online_policy(Batch(obs=obs[i : i + 1]), None)
                act = online_result.get("act")[0]
                policy_data = online_result.get("policy", Batch())
                state_data = online_result.get("state", None)
                source = "online"

            acts.append(act)
            policies.append(policy_data)
            states.append(state_data)
            sources.append(source)

            self.step_counters[env_id] += 1

            # Log the decision source
            # self.log_records.append(
            #     {
            #         "env_id": env_id,
            #         "step": self.step_counters[env_id] - 1,
            #         "source": source,
            #     }
            # )

        result.act = np.stack(acts)
        result.policy = Batch.stack(policies)
        result.source = sources  # for debugging
        if any(state is not None for state in states):
            result.state = states

        return result

    def reset_envs(self, env_ids):
        for env_id in env_ids:
            self.step_counters[env_id] = 0

    def map_action(self, act):
        return self.online_policy.map_action(act)

    def map_action_inverse(self, act):
        return self.online_policy.map_action_inverse(act)

    def exploration_noise(self, act, data):
        return self.online_policy.exploration_noise(act, data)

    def to(self, device):
        self.offline_policy.to(device)
        self.online_policy.to(device)
        return self

    def save_log_to_csv(self, filename="policy_source_log.csv"):
        pd.DataFrame(self.log_records).to_csv(filename, index=False)
