#from MnemonicReader import myPredictor
import myPredictor
from MultiHopEnvironment import MultiHopEnvironment, decode_state
from PPO import Agent
import warnings
warnings.filterwarnings("ignore")

env = MultiHopEnvironment()
observationOld, raw_old_state, answer = env.reset()

n_steps = 30
agent = Agent(batch_size=1, alpha=0.003, n_epochs=1)
agent.load_models()

done = False
score = 0
idx_steps = 0

while not done and idx_steps < n_steps:
#while idx_steps < n_steps:
    action, prob, val = agent.choose_action(observationOld)
    #print(f"---STEP: {idx_steps} ---")
    #print("STAMPA 1")
    observationNew, reward, done, raw_new_state = env.step(action)
    idx_steps += 1
    score += reward

#print(observationNew)
print("\n\n------------------------------------------------------------------------")
print(f"Score: {score}")
print(f"Done: {done}")

question = raw_new_state[0]
text_to_read = "".join(raw_new_state[2])
print(f"\nText to read: {text_to_read}")

prediction = myPredictor.myPredict(text_to_read, question)
print(f"\nQuestion: {question}")
print(f"\nPrediction: {prediction}")
print(f"Correct answer: {answer}")
print("------------------------------------------------------------------------\n\n")

#print(decode_state(observationNew, env.encoder))