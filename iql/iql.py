# algorithms/iql.py
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse our squashed actor and ensemble Q critic from common/net.py
from osrl.common.net import SquashedGaussianMLPActor, EnsembleQCritic


# A simple value network using an MLP
class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_sizes, activation):
        """
        Creates an MLP-based value network.

        Args:
            state_dim (int): Dimension of the state input.
            hidden_sizes (list): List of hidden layer sizes.
            activation (nn.Module): Activation function class to use.
        """
        super(ValueNet, self).__init__()
        layers = []
        in_size = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(activation())
            in_size = h
        layers.append(nn.Linear(in_size, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, state):
        """Computes the scalar value for the given state."""
        return self.model(state)


def expectile_loss(diff: torch.Tensor, tau: float):
    """
    Computes the expectile loss.

    Args:
        diff (torch.Tensor): Difference (e.g., Q(s, a) - V(s)).
        tau (float): Expectile parameter (typically between 0.5 and 1).

    Returns:
        torch.Tensor: The weighted squared error.
    """
    weight = torch.where(diff > 0, tau, 1 - tau)
    return weight * diff.pow(2)


class IQL(nn.Module):
    """
    Implicit Q-Learning (IQL) Model.

    This model contains a policy (actor), an ensemble Q network and a value network.
    The value network is trained using expectile regression.

    Args:
        state_dim (int): Dimension of the state space.
        action_dim (int): Dimension of the action space.
        max_action (float): Maximum absolute value for actions.
        hidden_sizes (list): Hidden layer sizes used in actor, Q network, and value network.
        gamma (float): Discount factor.
        tau (float): Soft update coefficient for target networks.
        expectile (float): Expectile regression parameter.
        beta (float): Temperature parameter for policy extraction.
        num_q (int): Number of Q networks in the ensemble.
        device (str): Device to run the networks on.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        hidden_sizes: list = [256, 256],
        gamma: float = 0.99,
        tau: float = 0.005,
        expectile: float = 0.7,
        beta: float = 3.0,
        num_q: int = 1,
        device: str = "cpu",
    ):
        super(IQL, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.expectile = expectile
        self.beta = beta
        self.device = device

        # Build networks
        self.actor = SquashedGaussianMLPActor(
            state_dim, action_dim, hidden_sizes, nn.ReLU
        ).to(device)
        self.q_net = EnsembleQCritic(
            state_dim, action_dim, hidden_sizes, nn.ReLU, num_q=num_q
        ).to(device)
        self.value_net = ValueNet(state_dim, hidden_sizes, nn.ReLU).to(device)

        # Create target networks (deepcopies)
        self.actor_old = deepcopy(self.actor)
        self.q_net_old = deepcopy(self.q_net)
        self.value_net_old = deepcopy(self.value_net)

    def _soft_update(self, target: nn.Module, source: nn.Module):
        """
        Perform a soft update of the target network.
        """
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1 - self.tau) * target_param.data
            )

    def q_loss(self, obs, actions, rewards, next_obs, done):
        """
        Compute the Q-network loss (MSE loss against Bellman backup).
        """
        q_pred, _ = self.q_net.predict(obs, actions)
        with torch.no_grad():
            v_next = self.value_net(next_obs)
            target_q = rewards + self.gamma * (1 - done) * v_next
        loss_q = F.mse_loss(q_pred, target_q)
        return loss_q

    def value_loss(self, obs, actions):
        """
        Compute the value network loss using expectile regression.
        """
        with torch.no_grad():
            q_val, _ = self.q_net.predict(obs, actions)
        v_val = self.value_net(obs)
        diff = q_val - v_val
        loss_v = expectile_loss(diff, self.expectile).mean()
        return loss_v

    def actor_loss(self, obs):
        """
        Compute the actor loss using weighted log-probabilities.
        """
        actions, logp = self.actor(obs, deterministic=False, with_logprob=True)
        with torch.no_grad():
            q_val, _ = self.q_net.predict(obs, actions)
            v_val = self.value_net(obs)
            weights = torch.exp((q_val - v_val) / self.beta)
        loss_actor = -(weights * logp).mean()
        return loss_actor

    def sync_target(self):
        """
        Soft update the target networks.
        """
        self._soft_update(self.q_net_old, self.q_net)
        self._soft_update(self.value_net_old, self.value_net)
        self._soft_update(self.actor_old, self.actor)

    def act(self, obs, deterministic=False):
        """
        Select an action given a state.

        Args:
            obs (numpy.array): The input state.
            deterministic (bool): Whether to select a deterministic action.

        Returns:
            numpy.array: Selected action.
        """
        obs_tensor = torch.tensor(obs[None, ...], dtype=torch.float32).to(self.device)
        action, _ = self.actor(obs_tensor, deterministic, with_logprob=False)
        return (action * self.max_action).detach().cpu().numpy()[0]
