import numpy as np
from PPO import Agent
from utils import plot_learning_curve
from MultiHopEnvironment import MultiHopEnvironment

if __name__ == '__main__':
    env = MultiHopEnvironment()
    state = env.reset()

    N = 30
    batch_size = 15
    n_epochs = 2
    alpha = 0.0003
    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, 
                    alpha=alpha, n_epochs=n_epochs, 
                    input_dims=env.observation_space.shape)

    