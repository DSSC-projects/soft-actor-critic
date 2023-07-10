import torch
import torch.nn as nn


class FeedForward(torch.nn.Module):
    """
    Implementation of feedforward network, also refered as multilayer
    perceptron, adapted from https://github.com/mathLab/PINA/tree/master/pina

    :param list(str) input_variables: the list containing the labels
        corresponding to the input components of the model.
    :param list(str) output_variables: the list containing the labels
        corresponding to the components of the output computed by the model.
    :param int inner_size: number of neurons in the hidden layer(s). Default is
        20.
    :param int n_layers: number of hidden layers. Default is 2.
    :param func: the activation function to use. If a single
        :class:`torch.nn.Module` is passed, this is used as activation function
        after any layers, except the last one. If a list of Modules is passed,
        they are used as activation functions at any layers, in order.
    :param iterable(int) layers: a list containing the number of neurons for
        any hidden layers. If specified, the parameters `n_layers` e
        `inner_size` are not considered.
    :param iterable(torch.nn.Module) extra_features: the additional input
        features to use ad augmented input.
    :param bool bias: If `True` the MLP will consider some bias.
    """

    def __init__(
        self,
        input_variables,
        output_variables,
        inner_size=20,
        n_layers=2,
        func=nn.Tanh,
        layers=None,
        bias=True,
    ):
        """ """
        super().__init__()

        if isinstance(input_variables, int):
            self.input_variables = None
            self.input_dimension = input_variables
        elif isinstance(input_variables, (tuple, list)):
            self.input_variables = input_variables
            self.input_dimension = len(input_variables)

        if isinstance(output_variables, int):
            self.output_variables = None
            self.output_dimension = output_variables
        elif isinstance(output_variables, (tuple, list)):
            self.output_variables = output_variables
            self.output_dimension = len(output_variables)

        if layers is None:
            layers = [inner_size] * n_layers

        tmp_layers = layers.copy()
        tmp_layers.insert(0, self.input_dimension)
        tmp_layers.append(self.output_dimension)

        self.layers = []
        for i in range(len(tmp_layers) - 1):
            self.layers.append(nn.Linear(tmp_layers[i], tmp_layers[i + 1], bias=bias))

        if isinstance(func, list):
            self.functions = func
        else:
            self.functions = [func for _ in range(len(self.layers) - 1)]

        if len(self.layers) != len(self.functions) + 1:
            raise RuntimeError("uncosistent number of layers and functions")

        unique_list = []
        for layer, func in zip(self.layers[:-1], self.functions):
            unique_list.append(layer)
            if func is not None:
                unique_list.append(func())
        unique_list.append(self.layers[-1])

        self.model = nn.Sequential(*unique_list)

    def forward(self, x):
        """
        Defines the computation performed at every call.
        """

        return self.model(x)


class PolicyNetwork(nn.Module):
    """
    Parametrized Policy Network.
    Code adapted from https://gymnasium.farama.org/tutorials/training_agents/reinforce_invpend_gym_v26/
    """

    def __init__(
        self,
        obs_space_dims,
        action_space_dims,
        hidden_space=None,
        kwargs_shared_net={},
        kwargs_mean_net={},
        kwargs_std_net={},
    ):
        """
        Initializes a neural network that estimates the mean and standard deviation
        of a normal distribution from which an action is sampled from.

        TODO
        """
        super().__init__()

        if hidden_space is None:
            hidden_space = 2 * action_space_dims

        # Shared Network
        self.shared_net = FeedForward(
            input_variables=obs_space_dims,
            output_variables=hidden_space,
            **kwargs_shared_net
        )

        # Policy Mean specific Linear Layer
        self.policy_mean_net = FeedForward(
            input_variables=hidden_space,
            output_variables=action_space_dims,
            **kwargs_mean_net
        )

        # Policy Std Dev specific Linear Layer
        self.policy_stddev_net = FeedForward(
            input_variables=hidden_space,
            output_variables=action_space_dims,
            **kwargs_std_net
        )

    def forward(self, x):
        """
        Conditioned on the observation, returns the mean and standard deviation
        of a normal distribution from which an action is sampled from.

        :param torch.tesor x: Observation from the environment
        :return: predicted mean and standard deviation of the normal distribution
        :return type: tuple(torch.tensor)
        """

        shared_features = self.shared_net(x.float())

        action_means = self.policy_mean_net(shared_features)
        log_action_stddevs = self.policy_stddev_net(shared_features)

        return action_means, log_action_stddevs


class CriticNetwork(nn.Module):
    """
    TODO
    """

    def __init__(self, obs_space_dims, action_space_dims, kwargs_shared_net={}):
        """
        TODO
        """
        super().__init__()

        # Shared Network
        self.shared_net = FeedForward(
            input_variables=obs_space_dims + action_space_dims,
            output_variables=1,
            **kwargs_shared_net
        )

    def forward(self, x):
        """
        TODO
        """
        x = torch.hstack(x).float()
        return self.shared_net(x)


class ValueNetwork(nn.Module):
    """
    TODO
    """

    def __init__(self, obs_space_dims, kwargs_shared_net={}):
        """
        TODO
        """
        super().__init__()

        # Shared Network
        self.shared_net = FeedForward(
            input_variables=obs_space_dims, output_variables=1, **kwargs_shared_net
        )

    def forward(self, x):
        """
        TODO
        """

        return self.shared_net(x.float())
