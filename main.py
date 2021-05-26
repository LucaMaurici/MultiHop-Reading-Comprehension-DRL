import numpy as np
from PPO import Agent
from my_utils import plot_learning_curve, plot_learning_curve_average
from MultiHopEnvironment import MultiHopEnvironment
import dill

if __name__ == '__main__':
    env = MultiHopEnvironment()
    state = env.reset()

    encoder = None
    with open('StaticTokenizerEncoder.pkl', 'rb') as f:
        encoder = dill.load(f)
    print('Encoder opened')

    N = 1
    n_actions = 8
    batch_size = 30
    n_epochs = 70
    alpha = 0.003
    n_episodes = 1500
    n_steps = 30
    agent = Agent(batch_size=batch_size, alpha=alpha, n_epochs=n_epochs)

    print(agent.actor)

    learn_iters = 0
    avg_score = 0
    idx_globalSteps = 0
    score_history = list()


    for idx_episodes in range(1, n_episodes+1):
        print(f"\n---EPISODE {idx_episodes} ---")
        observationOld, _, _ = env.reset()
        #print(observationOld)
        done = False
        score = 0
        idx_steps = 0
        while not done and idx_steps < n_steps:
        #while idx_steps < n_steps:
            action, prob, val = agent.choose_action(observationOld)
            #print(f"---STEP: {idx_steps} ---")
            #print("STAMPA 1")
            observationNew, reward, done, _ = env.step(action)
            score += reward
            agent.remember(observationOld, action, prob, val, reward, done)
            #print("STAMPA 2")
            idx_steps += 1
        if idx_episodes % N == 0:
            #print(f"---LEARN {idx_episodes} ---")
            agent.learn()
            agent.save_models()
            learn_iters += 1
        #print("STAMPA 3")
        observationOld = observationNew
        score_history.append(score)
        avg_score = np.mean(score_history[-30:])
        #print("STAMPA 4")
        #Per futura parte di ricerca salvare il modello quando l'avg_score è maggiore del best
        #if avg_score > best_score:
            #best_score = avg_score
        
        #print("STAMPA 5")
        idx_episodes += 1
        idx_globalSteps += idx_steps

        print('episode', idx_episodes, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', idx_globalSteps, 'learning_steps', learn_iters)

    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve_average(x, score_history, "temp\\learning_curve_average.jpg")
    plot_learning_curve(x, score_history, "temp\\learning_curve.jpg")



    