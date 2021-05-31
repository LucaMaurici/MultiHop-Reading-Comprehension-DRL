#from MnemonicReader import myPredictor
import myPredictor
from MultiHopEnvironment import MultiHopEnvironment, decode_state
from PPO import Agent
import warnings
import utils
warnings.filterwarnings("ignore")

env = MultiHopEnvironment()

n_steps = 30
agent = Agent(batch_size=1, alpha=0.003, n_epochs=1)
agent.load_models()

n_samples = 10

for i in range(n_samples):
    observationOld, raw_old_state, answer = env.reset()
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

    em_score_sample = int(utils.exact_match_score(prediction, answer))
    print(f"EM Score Sample: {em_score_sample}")

    em_score_tot += em_score_sample
    print("------------------------------------------------------------------------\n\n")

print(f"EM Score Final: {em_score_tot/n_samples}")



#print(decode_state(observationNew, env.encoder))

'''
def exact_match(actual, prediction):
    correct = 0.
    total = 0
    for (index, val) in enumerate(actual):
        for (i, v) in enumerate(val):
            if (actual[index][i] == prediction[index][i]):
                correct += 1
            total += 1
    return correct/total
'''