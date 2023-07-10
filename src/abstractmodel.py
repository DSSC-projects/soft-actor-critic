from abc import ABCMeta, abstractmethod

import torch


class Model(metaclass=ABCMeta):
    def __init__(self):
        """
        Abstract class for models.

        :param env: A gym environmnet instance.
        :type env: gym.wrappers.time_limit.TimeLimit
        """

        # dictionary s.t. (name_net:net) for each net in the model, to be overwritten if any
        self.approximator = {}

        pass

    @abstractmethod
    def sample_action(self, state):
        """
        Sample an action from a policy.

        :param state: The current state.
        :type state: numpy.ndarray
        """
        pass

    def act(self, state):
        state = torch.tensor(state)
        return self.sample_action(state).detach().numpy()

    def optimization_step_in(self, buffer):
        """
        The optimization step is used to optimize the
        model. It must return the action corresponding
        to state_prime.
        """
        return

    def optimization_step_out(self, rewards):
        """
        The optimization step is used to optimize the
        model. It must return the action corresponding
        to state_prime.
        """
        return

    def save_checkpoint(self, path_env, episode):
        """
        Save checkpoints of each net of the Model
        """
        for func in self.approximator.items():
            if "optimizer" in func[1]:
                torch.save(
                    {
                        "model_state_dict": func[1]["net"].state_dict(),
                        "optimizer_state_dict": func[1]["optimizer"].state_dict(),
                    },
                    path_env + "/" + self.model_name + "_" + func[0] + f"_{episode}",
                )
            else:
                torch.save(
                    {"model_state_dict": func[1]["net"].state_dict()},
                    path_env + "/" + self.model_name + "_" + func[0] + f"_{episode}",
                )
        return

    def load_checkpoint(self, path_env, episode):
        """
        Load a checkpoint of each net of the model
        """
        path_base = path_env + "/" + self.model_name + "_"

        for func in self.approximator.items():
            print(func[0])
            path = path_base + str(func[0]) + "_" + f"{episode}"
            print(path)
            checkpoint = torch.load(path)
            func[1]["net"].load_state_dict(checkpoint["model_state_dict"])
            if "optimizer" in func[1]:
                func[1]["optimizer"].load_state_dict(checkpoint["optimizer_state_dict"])

        return
