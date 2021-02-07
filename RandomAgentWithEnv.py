import pickle
from MultiHopEnvironment import MultiHopEnvironment
import random

env = MultiHopEnvironment()

print("\n--- State: ---\n")
print(env.reset())

for i in range(300):
    print("\n\n--- I: ", i, "---\n")
    state, reward, done = env.step(random.randint(0, 8))
    print("\n--- State: ---\n")
    print('\nQuestion: ', state[0])
    print('\nActions: ', state[1])
    print('\nSentences: ', state[2])
    print("\n--- Reward: ")
    print(reward)
    print("\n--- Done: ")
    print(done)
    if done:
        break