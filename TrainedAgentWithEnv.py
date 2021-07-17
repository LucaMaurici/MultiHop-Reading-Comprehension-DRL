#from MnemonicReader import myPredictor
import myPredictor
#from MultiHopEnvironment import MultiHopEnvironment, decode_state
from MultiHopEnvironment import MultiHopEnvironment
from my_model import Agent
#from PPO import Agent
import warnings
import utils
import random
warnings.filterwarnings("ignore")
import paths
import pickle

env = MultiHopEnvironment(train_mode=False)

graph_path = "CoreferenceGraphsList_dev3.pkl"
'''
with open(paths.graph_path_dev, 'rb') as f:
    graphs_list = pickle.load(f)
n_samples = len(graphs_list)
'''
with open(graph_path, 'rb') as f:
    graphs_list = pickle.load(f)
n_samples = len(graphs_list)

n_steps = 30
agent = Agent(batch_size = 1, alpha = 0.003, n_epochs = 1)
agent.load_models()

em_score_tot = 0

for i in range(n_samples):
    observationOld, raw_old_state, answer, candidates = env.reset()
    done = False
    score = 0
    idx_steps = 0

    #while not done and idx_steps < n_steps:
    while idx_steps < n_steps:
        #action, _, _ = agent.choose_action(observationOld)
        action, _ = agent.choose_action(observationOld)
        #print(f"---STEP: {idx_steps} ---")
        #print("STAMPA 1")
        observationNew, reward, done, raw_new_state = env.step(action)
        #observationNew, reward, done, raw_new_state = env.step(random.randint(0, 31))
        '''
        reward = -0.1
        while(reward == -0.1):
            observationNew, reward, done, raw_new_state = env.step(random.randint(0, 34))
        '''
        idx_steps += 1
        score += reward

    #print(observationNew)
    print("\n\n------------------------------------------------------------------------")
    print(f"Score: {score}")
    print(f"Done: {done}")

    question = raw_new_state[0]
    print(f"\nQuestion: {question}")

    text_to_read = "".join(raw_new_state[2])
    print(f"\nText to read: {text_to_read}")

    #candidates
    #sample = getSampleById(dataset, sampleId)
    print(candidates)

    if text_to_read != "":
        #prediction = myPredictor.myPredict(text_to_read, question, candidates=candidates)
        prediction = myPredictor.myPredict(text_to_read, question)
    else:
        prediction = [("", 1.0)]
    
    print(f"\nPrediction: {prediction}")
    print(f"Correct answer: {answer}")

    if len(prediction) > 0:
        prediction = prediction[0][0]
    else:
        prediction = ""
    em_score_sample = int(utils.exact_match_score(prediction, answer))
    print(f"EM Score - Sample: {em_score_sample*100}%")

    em_score_tot += em_score_sample
    print(f"EM Score - partial: {(em_score_tot/(i+1))*100}%")
    print("------------------------------------------------------------------------\n\n")

print(f"EM Score - Final: {(em_score_tot/n_samples)*100}%")



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