import sys
sys.path.append('../.')
import os

from src.models import Reinforce
from src.problem import RlProblem
from src.neuralnets import PolicyNetwork
from src.trainer import Trainer
from src.utils import plot_reward, Writer
import torch


if __name__ == "__main__":

    seed = int(sys.argv[1])
    torch.manual_seed(seed)
    
    # create the problem
    problem = RlProblem(environment_name=sys.argv[2])

    # create the model
    policy_net = PolicyNetwork(obs_space_dims=problem.obs_space_dims,
                               action_space_dims=problem.action_space_dims,
                               kwargs_mean_net={'n_layers':0},
                               kwargs_std_net={'n_layers':0}
                               )

    model = Reinforce(policy_net, action_rescale=int(sys.argv[3]))

    # path to directory
    path_to_directory = sys.argv[4]

    # create directory
    if not os.path.exists(path_to_directory + problem._environment_name):
        os.makedirs(path_to_directory + problem._environment_name)

        # create trainer
        trainer = Trainer(model=model,
                          problem=problem,
                          writer=Writer(n_step=1,
                                        stdout=True,
                                        filename=path_to_directory+problem._environment_name+"/test_reinforce"))

        # train the model
        trainer.train(int(sys.argv[5]), save_checkpoint_every=100, path_to_directory=path_to_directory+problem._environment_name)

        plot_reward(trainer.reward_for_episodes,
                    moving_average_window=20,
                    filename=path_to_directory+problem._environment_name+"/plot_reinforce.png")
