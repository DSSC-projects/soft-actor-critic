import numpy as np


class Writer(object):
    """
    A simple writer routine for saving data.
    """

    def __init__(self, n_step, filename=None, stdout=True):
        if isinstance(filename, str):
            self._filename = filename
            with open(self._filename, "w") as f:
                f.write("episode,reward\n")
        else:
            self._filename = None

        # stdout print
        assert isinstance(stdout, bool)
        self._stdout = stdout

        if isinstance(n_step, int):
            self._n_step = n_step

    def logging_step(self, episode, reward):
        if self._stdout and (episode % self._n_step == 0):
            print(f"Episode {episode} -- Reward {reward}")

        if self._filename:
            with open(self._filename, "a") as f:
                f.write(f"{episode},{reward}\n")


class ReplayBuffer:
    """
    A simple replay buffer, adapted from
    https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/SAC/buffer.py
    """

    def __init__(self, max_size, obs_space_dims, action_space_dims):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, obs_space_dims))
        self.new_state_memory = np.zeros((self.mem_size, obs_space_dims))
        self.action_memory = np.zeros((self.mem_size, action_space_dims))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_batch_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones
