from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal

from .abstractmodel import Model

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class Random(Model):
    def __init__(self, action_space_dims):
        super().__init__()
        self.action_space_dim = action_space_dims
        self.model_name = "random"

    def sample_action(self, state):
        """
        Sample an action from a policy.

        :param env: A gym environmnet instance.
        :type env: gym.wrappers.time_limit.TimeLimit
        :param state: The current state, not used.
        :type state: numpy.ndarray
        """
        return torch.tensor(np.random.uniform(size=(self.action_space_dim,)))


class Reinforce(Model):
    """REINFORCE algorithm."""

    def __init__(
        self,
        policy_net,
        optimizer=torch.optim.Adam,
        optimizer_kwargs={"lr": 1e-3},
        gamma=0.99,
        action_rescale=1.0,
    ):
        super().__init__()

        self.model_name = "reinforce"
        # Hyperparameters
        self.gamma = gamma

        self.probs = []  # Stores probability values of the sampled action
        self.rewards = []  # Stores the corresponding rewards

        net = policy_net
        optimizer = optimizer(net.parameters(), **optimizer_kwargs)

        self.approximator = {"policy": {"net": net, "optimizer": optimizer}}
        self.action_rescale = torch.tensor(action_rescale)

    def sample_action(self, state, reparametrization=False):
        """
        Sample an action from a policy. The action is
        rescale according to ``action_rescale`` defined
        in the __init__.

        :param state: The current state.
        :type state: numpy.ndarray
        """
        mu, log_sigma = self.approximator["policy"]["net"](state)
        probabilities = Normal(mu + 1e-9, log_sigma.exp() + 1e-9)

        if reparametrization:
            sample = probabilities.rsample()
        else:
            sample = probabilities.sample()

        log_probs = torch.sum(probabilities.log_prob(sample), dim=-1)

        # rescaling actions and probs
        action = torch.tanh(sample) * self.action_rescale
        log_probs -= torch.log(self.action_rescale * (1 - action.pow(2)) + 1e-6).sum(
            dim=-1
        )

        self.probs.append(log_probs)

        return action

    def optimization_step_out(self, rewards):
        """Updates the policy network's weights."""
        running_g = 0
        gs = []

        # Discounted return (backwards) - [::-1] will return an array in reverse
        for R in rewards[::-1]:
            running_g = R + self.gamma * running_g
            gs.append(running_g)

        # Update the policy network
        self.approximator["policy"]["optimizer"].zero_grad()
        loss = -torch.mean(torch.stack(self.probs) * torch.tensor(gs[::-1]))
        loss.backward()
        self.approximator["policy"]["optimizer"].step()

        # Empty / zero out all episode-centric/related variables
        self.probs = []


class SAC(Model):
    """Self Actor Critic algorithm."""

    def __init__(
        self,
        value,
        actor,
        critics,
        optimizer_value=torch.optim.Adam,
        optimizer_value_kwargs={"lr": 3e-4},
        optimizer_actor=torch.optim.Adam,
        optimizer_actor_kwargs={"lr": 3e-4},
        optimizer_critics=[torch.optim.Adam, torch.optim.Adam],
        optimizer_critics_kwargs=[{"lr": 3e-4}, {"lr": 3e-4}],
        gamma=0.99,
        tau=0.005,
        alpha=0.1,
        batch_size=264,
        reparametrization=True,
        action_rescale=1,
        update_every=1,
        deterministic=False,
    ):
        """
        Builds the Soft Actor Critic method for solving
        RL problems.

        :param torch.nn.Module value: The state-value network.
        :param torch.nn.Module actor: The policy network.
        :param list[torch.nn.Module] critics: A list of critic networks, original paper uses 2.
        :param torch.optim optimizer_value: Optimizer for stat-value network,
            defaults to torch.optim.Adam
        :param dict optimizer_value_kwargs: Additional keywords arguments passed to
            ``optimizer_value``, defaults to {'lr': 3e-4}
        :param torch.optim optimizer_actor: Optimizer for actor network,
            defaults to torch.optim.Adam
        :param dict optimizer_actor_kwargs: Additional keywords arguments passed to
            ``optimizer_value``, defaults to {'lr': 3e-4}
        :param optimizer_critics: List of optimizers for the critics network, one for each.
            Defaults to [torch.optim.Adam, torch.optim.Adam]
        :type optimizer_critics: list, optional
        :param optimizer_critics_kwargs: List of additional keywords arguments passed to
            ``optimizer_critics``, defaults to [{ 'lr': 3e-4 }, { 'lr': 3e-4 }]
        :type optimizer_critics_kwargs: list, optional
        :param gamma: Discout factor, defaults to 0.99
        :type gamma: float, optional
        :param tau: Exponential moving average parameter, defaults to 0.005
        :type tau: float, optional
        :param alpha: Entropy regularizer strength, defaults to 0.1
        :type alpha: float, optional
        :param batch_size: Batch size for optimization, defaults to 256
        :type batch_size: int, optional
        :param reparametrization: Reparametrization trick option, defaults to True.
            If True, the reparametrization trick is applied for sampling.
        :type reparametrization: bool, optional
        :param action_rescale: Rescaling action space, defaults to 1.
        :type action_rescale: int, optional
        :param update_every: Number of iterations before performing an update, defaults to 1.
        :type update_every: int, optional
        :param deterministic: Use a deterministic policy for training, defaults to False.
        :type deterministic: bool, optional
        """

        super().__init__()
        self.model_name = "sac"
        # Hyperparameters
        self.tau = tau  # small number for mathematical stability
        self.gamma = gamma
        self.alpha = alpha
        self.batch_size = batch_size
        self._len_critics = 2

        # small checks
        assert len(optimizer_critics) == self._len_critics
        assert len(optimizer_critics_kwargs) == self._len_critics

        # make networks
        value_net = value
        value_target_net = deepcopy(value)
        actor_net = actor
        critic1_net = critics[0]
        critic2_net = critics[1]

        # make the optimizers
        optimizer_value_net = optimizer_value(
            value_net.parameters(), **optimizer_value_kwargs
        )
        optimizer_actor_net = optimizer_actor(
            actor_net.parameters(), **optimizer_actor_kwargs
        )
        optimizer_critic1_net = optimizer_critics[0](
            critic1_net.parameters(), **optimizer_critics_kwargs[0]
        )
        optimizer_critic2_net = optimizer_critics[1](
            critic2_net.parameters(), **optimizer_critics_kwargs[1]
        )

        # collect networks and correspondent optimizers
        self.approximator = {
            "value": {"net": value_net, "optimizer": optimizer_value_net},
            "value_target": {"net": value_target_net},
            "actor": {"net": actor_net, "optimizer": optimizer_actor_net},
            "critic1": {"net": critic1_net, "optimizer": optimizer_critic1_net},
            "critic2": {"net": critic2_net, "optimizer": optimizer_critic2_net},
        }

        # freeze params for target net
        for p in self.approximator["value_target"]["net"].parameters():
            p.requires_grad = False

        # setting action range and reparametrization
        self.action_rescale = torch.tensor(action_rescale)
        self.reparametrization = reparametrization
        self.update_every = update_every
        self.counter = 0
        self.deterministic = deterministic

    def sample_action(self, state, return_prob=False):
        """
        Sample an action from the policy network by
        using the reparametrization trick.

        :param np.ndarray state: The state from where to pick the action.
        :param bool return_prob: Returing also log probabilities,
            defaults to False.
        :return: Actions given the current state, log probability if
            ``return_prob=True``.
        :rtype: np.ndarray
        """

        # calculate mean and log standard deviation
        mu, log_sigma = self.approximator["actor"]["net"](state)
        # clamp the log std, as in the original paper
        sigma = torch.clamp(log_sigma, LOG_STD_MIN, LOG_STD_MAX).exp()
        # sample from a normal distribution
        probabilities = Normal(mu + 1e-9, sigma + 1e-9)
        # choose deterministic policy
        if self.deterministic:
            sample = mu
        else:  # choose a sample
            # apply reparametrization
            if self.reparametrization:
                sample = probabilities.rsample()
            else:
                sample = probabilities.sample()

        # log probabilities for the sampled action
        log_probs = torch.sum(probabilities.log_prob(sample), dim=-1)
        # rescaling actions and probs in action_rescale * [-1, 1] range
        action = torch.tanh(sample) * self.action_rescale
        log_probs -= torch.log(self.action_rescale * (1 - action.pow(2)) + 1e-6).sum(
            dim=-1
        )

        if return_prob:
            return action, log_probs.flatten()

        return action

    def update_network_parameters(self):
        """
        Exponential moving average to update
        target value network.
        """
        for target_param, param in zip(
            self.approximator["value_target"]["net"].parameters(),
            self.approximator["value"]["net"].parameters(),
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def optimization_step_in(self, buffer):
        """
        Performing the optimization step

        :param buffer: The buffer where historical data are saved.
        :type buffer: ReplayBuffer
        """
        if self.counter % self.update_every == 0:
            self._optimization(buffer)

        self.counter += 1

    def _optimization(self, buffer):
        """
        Optimization step for updating networks' weights.

        :param buffer: The buffer where historical data are saved.
        :type buffer: ReplayBuffer
        """

        # initial exploration phase
        if buffer.mem_cntr < self.batch_size:
            return

        # sampling from buffer
        state, action, reward, new_state, done = buffer.sample_batch_buffer(
            self.batch_size
        )

        # convert to torch tensor
        reward = torch.tensor(reward, dtype=torch.float)
        done = torch.tensor(done, dtype=torch.float)
        state_ = torch.tensor(new_state, dtype=torch.float)
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)

        # ==== State-Value function optimization ==== #

        # get the state-value function for current state
        value = self.approximator["value"]["net"](state).flatten()

        # get an action for current state from policy
        action_, log_probs = self.sample_action(state, return_prob=True)  #### TODO

        # get two critics outputs and choose the min
        q_value = torch.min(
            self.approximator["critic1"]["net"]([state, action_]),
            self.approximator["critic2"]["net"]([state, action_]),
        ).flatten()

        # optimize value function
        self.approximator["value"]["optimizer"].zero_grad()
        value_target = q_value - self.alpha * log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target.detach())
        value_loss.backward()
        self.approximator["value"]["optimizer"].step()

        # ==== Critics  optimization ==== #

        self.approximator["critic1"]["optimizer"].zero_grad()
        self.approximator["critic2"]["optimizer"].zero_grad()

        # compute q_hat + loss critics
        q_hat = reward + self.gamma * self.approximator["value_target"]["net"](
            state_
        ).flatten() * (1 - done)
        loss_critic_1 = 0.5 * F.mse_loss(
            self.approximator["critic1"]["net"]([state, action]).flatten(),
            q_hat.detach(),
        )
        loss_critic_2 = 0.5 * F.mse_loss(
            self.approximator["critic2"]["net"]([state, action]).flatten(),
            q_hat.detach(),
        )

        # backward and optimize
        loss_critic_1.backward()
        loss_critic_2.backward()
        self.approximator["critic1"]["optimizer"].step()
        self.approximator["critic2"]["optimizer"].step()

        # ==== Actor  optimization ==== #

        # get an action
        action_, log_probs = self.sample_action(state, return_prob=True)

        # get two critics outputs and choose the min
        q_value = torch.min(
            self.approximator["critic1"]["net"]([state, action_]),
            self.approximator["critic2"]["net"]([state, action_]),
        ).flatten()
        # optimize actor
        self.approximator["actor"]["optimizer"].zero_grad()
        actor_loss = (self.alpha * log_probs - q_value).mean()
        actor_loss.backward()
        self.approximator["actor"]["optimizer"].step()

        # update with EMA
        self.update_network_parameters()

        # reset counter
        self.counter = 0
