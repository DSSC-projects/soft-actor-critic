import sys

sys.path.append("../.")

from argparse import ArgumentParser

import torch
import torch.nn as nn

from src.models import SAC
from src.neuralnets import CriticNetwork, PolicyNetwork, ValueNetwork
from src.problem import RlProblem
from src.trainer import Trainer


def parser():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--env", type=str, default="Hopper-v4")
    arg_parser.add_argument("--seed", type=int, default=0)
    arg_parser.add_argument("--n_steps", type=int, default=1000)
    arg_parser.add_argument("--checkpoint_episodes", type=int, default=100)
    arg_parser.add_argument("--reward_rescale", type=float, default=5)
    arg_parser.add_argument("--action_rescale", type=float, default=1)
    arg_parser.add_argument("--checkpoint_path", type=str)
    arg_parser.add_argument("--checkpoint_episode", type=int)
    arg_parser.add_argument("--filename", type=str)
    return arg_parser.parse_args()


if __name__ == "__main__":
    args = parser()

    problem = RlProblem(
        environment_name=args.env,
        environment_kwargs=dict(render_mode="rgb_array_list")
        if args.env in ["Swimmer-v4", "HalfCheetah-v4"]
        else dict(render_mode="rgb_array_list", terminate_when_unhealthy=False),
    )

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

    model.load_checkpoint(
        args.checkpoint_path,
        args.checkpoint_episode,
    )

    trainer = Trainer(model=model, problem=problem, writer=None)

    trainer.save_video(n_steps=args.n_steps, filename=args.filename)
