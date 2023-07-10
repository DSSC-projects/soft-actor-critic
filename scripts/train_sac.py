import sys
sys.path.append('../.')
import os

import torch 
import torch.nn as nn

from src.models import SAC
from src.neuralnets import CriticNetwork, PolicyNetwork, ValueNetwork
from src.problem import RlProblem
from src.trainer import Trainer
from src.utils import Writer, plot_reward


if __name__ == "__main__":

    seed = int(sys.argv[1])
    torch.manual_seed(seed)
    
    # create the problem
    problem = RlProblem(environment_name=sys.argv[2])

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
        alpha=float(sys.argv[6]),
        action_rescale=float(sys.argv[3]),
        update_every=1,
        tau=0.005,
        reparametrization=True,
    )

    # path to directory
    path_to_directory = sys.argv[4]

    # create directory
    if not os.path.exists(path_to_directory + problem._environment_name):
        os.makedirs(path_to_directory + problem._environment_name)
    
    # create trainer
    trainer = Trainer(
        model=model,
        problem=problem,
        writer=Writer(n_step=1, stdout=True, filename=path_to_directory+problem._environment_name+"/test_sac"),
    )

    # train the model
    trainer.train(int(sys.argv[5]), save_checkpoint_every=100, path_to_directory=path_to_directory+problem._environment_name)

    plot_reward(trainer.reward_for_episodes, moving_average_window=20,
                filename=path_to_directory+problem._environment_name+"/plot_sac.png")
