import gymnasium as gym


class RlProblem(object):
    def __init__(self, environment_name, environment_kwargs=None):
        """
        Define the RL environment

        :param str environment_name: A valid openai gym environment name.
        :param Model model: A RL model instance to optimize Bellmann equation.
        :param dict environment_kwargs: Addition keywards arguments for
            defining the environment.
        :param Writer writer: The writer for saving files.

        """

        self._environment_name = environment_name

        # make the environment
        if environment_kwargs is None:
            environment_kwargs = {}
        self._env = gym.make(environment_name, **environment_kwargs)

    @property
    def env(self):
        """
        Environment attribute class.
        """
        return self._env

    @property
    def obs_space_dims(self):
        """
        Model attribute class.
        """
        return self.env.observation_space.shape[0]

    @property
    def action_space_dims(self):
        """
        Number of episodes during optimization.
        """
        return self.env.action_space.shape[0]
