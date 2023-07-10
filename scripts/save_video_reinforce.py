import sys

sys.path.append("../.")

import torch
import torch.nn as nn

from src.models import SAC, Reinforce
from src.neuralnets import CriticNetwork, PolicyNetwork, ValueNetwork
from src.problem import RlProblem
from src.trainer import Trainer
from src.utils import Writer, plot_reward
from argparse import ArgumentParser

from src.utils import ReplayBuffer, Writer
import os
import tqdm

import gymnasium as gym
from gymnasium.utils.save_video import save_video
def parser():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--env", type=str, default="Hopper-v4")
    arg_parser.add_argument("--seed", type=int, default=0)
    arg_parser.add_argument("--n_steps", type=int, default=500000)
    arg_parser.add_argument("--checkpoint_episodes", type=int, default=100)
    arg_parser.add_argument("--reward_rescale", type=float, default=5)
    arg_parser.add_argument("--action_rescale", type=float, default=1)
    return arg_parser.parse_args()

if __name__ == "__main__":

    args = parser()
    torch.manual_seed(args.seed)
    checkpoint = 500

    # create folder with current date and time
    import os
    from datetime import datetime

    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")

    # create the problem
    problem = RlProblem(environment_name="Humanoid-v4", environment_kwargs=dict(render_mode="rgb_array_list", terminate_when_unhealthy=False))

    # create the model
    # create the model
    critic1 = CriticNetwork(
        obs_space_dims=problem.obs_space_dims,
        action_space_dims=problem.action_space_dims,
        kwargs_shared_net={"inner_size": 256, "n_layers": 2, "func": nn.ReLU},
    )

    critic2 = CriticNetwork(
        obs_space_dims=problem.obs_space_dims,
        action_space_dims=problem.action_space_dims,
        kwargs_shared_net={"inner_size": 256, "n_layers": 2, "func": nn.ReLU},
    )

    value = ValueNetwork(
        obs_space_dims=problem.obs_space_dims,
        kwargs_shared_net={"inner_size": 256, "n_layers": 2, "func": nn.ReLU},
    )

    policy = PolicyNetwork(
        obs_space_dims=problem.obs_space_dims,
        action_space_dims=problem.action_space_dims,
        hidden_space=256,
        kwargs_shared_net={"inner_size": 256, "n_layers": 1, "func": nn.ReLU},
        kwargs_mean_net={"n_layers": 0, "func": nn.ReLU},
        kwargs_std_net={"n_layers": 0, "func": nn.ReLU},
    )


    model = SAC(
        critics=[critic1, critic2],
        actor=policy,
        value=value,
        alpha=1 / args.reward_rescale,
        action_rescale=args.action_rescale,
        update_every=1,
        tau=0.005,
        reparametrization=True,
    )

    model.load_checkpoint("/home/apierro/rl-project/basic_scripts/HumanoidTest_09-07-2023_18-05-10_seed_40/Humanoid-v4", checkpoint)


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
                self._logger = None
            else:
                self._logger = writer

        def train(self, n_steps, checkpoint_episodes=1000):
            """
            Training routine for the RL problem.

            :param int n_steps: Number of steps to train the agent on.
            :param int checkpoint_episodes: Number of episodes between checkpoints.
            """
            state, _ = self.problem.env.reset()
            episode_reward = []
            episode = 0
            done = False

            for step in tqdm.tqdm(range(n_steps), position=0, leave=True):
                if done:
                    episode_reward = []
                    done = False
                    episode += 1
                    save_video(
                        self.problem.env.render(),
                        "videos",
                        fps=self.problem.env.metadata["render_fps"],
                        step_starting_index=0,
                        episode_index=0,
                        name_prefix=f"Hopper-v4-sac-{checkpoint}"
                    )
                    state, _ = self.problem.env.reset()
                    self.problem.env.close()
                    exit()

                action = self.model.act(state)
                state_prime, reward, terminated, truncated, _ = self.problem.env.step(
                    action
                )
                done = truncated or terminated
                done = step == n_steps - 2 or done

                state = state_prime
                episode_reward.append(reward)

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


    # create trainer
    trainer = Trainer(
        model=model,
        problem=problem,
        writer=Writer(n_step=1, stdout=True, filename=f"sac.csv"),
    )

    # train the model
    trainer.train(n_steps=1000, checkpoint_episodes=args.checkpoint_episodes)
