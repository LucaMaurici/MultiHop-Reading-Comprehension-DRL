import numpy as np
from PPO import Agent
from utils import plot_learning_curve
from MultiHopEnvironment import MultiHopEnvironment

if __name__ == '__main__':
	env = MultiHopEnvironment()
	env.reset()