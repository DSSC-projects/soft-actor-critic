from gymnasium.utils.save_video import save_video
from tqdm import tqdm

from .utils import ReplayBuffer, Writer


class Trainer(object):
    def __init__(self, model, problem, writer=None, max_size_buffer=10**6):
        """
        Define the RL environment

        :param str environment_name: A valid openai gym environment name.
        :param Model model: A RL model instance to optimize Bellmann equation.
        :param dict environment_kwargs: Addition keywards arguments for
            defining the environment.
        :param Writer writer: The writer for saving files.
        """

        # make the environment
        self._problem = problem
        self._model = model

        # reward
        self._total_reward = []

        # make the renderer TODO

        # make replay buffer
        self._buffer = ReplayBuffer(
            max_size=max_size_buffer,
            obs_space_dims=self.problem.env.observation_space.shape[0],
            action_space_dims=self.problem.env.action_space.shape[0],
        )

        # make the logger
        if not isinstance(writer, Writer):
            raise ValueError
        else:
            self._logger = writer

    def train(self, number_steps, save_checkpoint_every=1000, path_to_directory=""):
        """
        Training routine for the RL problem.

        :param int number_episodes: Number of episodes to evolve the environment.
        """

        state, _ = self.problem.env.reset()
        episode_reward = []
        episode = 0
        done = False

        # loop over episodes
        for step in tqdm(range(number_steps), position=0, leave=True):
            if done:
                self.model.optimization_step_out(episode_reward)
                self._total_reward.append(sum(episode_reward))
                self._logger.logging_step(episode, sum(episode_reward))
                episode_reward = []
                done = False
                episode += 1
                state, _ = self.problem.env.reset()

            action = self.model.act(state)
            state_prime, reward, terminated, truncated, _ = self.problem.env.step(
                action
            )
            done = truncated or terminated

            self._buffer.store(
                state=state,
                action=action,
                reward=reward,
                done=terminated,
                state_=state_prime,
            )

            # perform optimization every iteration
            self.model.optimization_step_in(self._buffer)

            # loop to next state/action
            state = state_prime

            # accumulate reward and check end
            episode_reward.append(reward)

            # save checkpoint in a directory with the problem's name
            if (episode % save_checkpoint_every == 0) and (
                len(self._model.approximator) > 0
            ):
                self._model.save_checkpoint(path_to_directory, episode)

        self.problem.env.close()

    def save_video(self, n_steps: int, filename: str) -> None:
        """
        Save a video of the agent's behavior.

        :param int n_steps: Number of steps to record.
        :param str filename: Name of the video file.
        """
        state, _ = self.problem.env.reset()

        for step in tqdm.tqdm(range(n_steps), position=0, leave=True):
            action = self.model.act(state)
            state, _ = self.problem.env.step(action)

        save_video(
            self.problem.env.render(),
            "videos",
            fps=int(self.problem.env.metadata["render_fps"] / 1.5),
            name_prefix=filename,
        )

        self.problem.env.close()

    @property
    def problem(self):
        """
        Environment attribute class.
        """
        return self._problem

    @property
    def model(self):
        """
        Model attribute class.
        """
        return self._model

    @property
    def number_episodes(self):
        """
        Number of episodes during optimization.
        """
        return len(self._total_reward)

    @property
    def reward_for_episodes(self):
        """
        Reward accross episodes.
        """
        return self._total_reward
