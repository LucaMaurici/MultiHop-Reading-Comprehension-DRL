from MnemonicReader import myPredictor

env = MultiHopEnvironment()
state = env.reset()

n_steps = 30
agent = Agent(batch_size=batch_size, alpha=alpha, n_epochs=n_epochs)
agent.load_models("torch_ppo_1.pth")

done = False
score = 0
idx_steps = 0

while not done and idx_steps < n_steps:
#while idx_steps < n_steps:
    action, prob, val = agent.choose_action(observationOld)
    #print(f"---STEP: {idx_steps} ---")
    #print("STAMPA 1")
    observationNew, reward, done = env.step(action)
    score += reward

print(observationNew)
print(score)
print(done)