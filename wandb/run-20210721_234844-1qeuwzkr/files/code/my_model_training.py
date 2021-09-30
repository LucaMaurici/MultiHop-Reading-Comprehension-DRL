import numpy as np
from my_model import Agent
from my_utils import plot_learning_curve, plot_learning_curve_average
from MultiHopEnvironment_MyModel import MultiHopEnvironment_MyModel
import wandb

LOAD_CHECKPOINT = False

if __name__ == '__main__':
    env = MultiHopEnvironment_MyModel(train_mode=True)

    N = 50
    n_actions = 9
    batch_size = 600
    n_epochs = 1
    alpha = 0.01 #0.003
    n_episodes = 15000
    n_steps = 30
    agent = Agent(batch_size=batch_size, alpha=alpha, n_epochs=n_epochs)
    if LOAD_CHECKPOINT:
        agent.actor.load_checkpoint()

    print(agent.actor)

    learn_iters = 0
    avg_score = 0
    score_history = list()
    avg_steps = 0
    steps_history = list()
    idx_globalSteps = 0
    

    run = wandb.init(project='Multi-Hop_Reading_Comprehension_myModel_2.0')
    # Log gradients and model parameters
    wandb.watch(agent.actor)

    for idx_episodes in range(1, n_episodes+1):
        print(f"\n---EPISODE {idx_episodes} ---")
        observationOld, _, _, _, _ = env.reset()
        #print(observationOld)
        done = False
        score = 0
        idx_steps = 0
        while not done and idx_steps < n_steps:
        #while idx_steps < n_steps:
            action, output = agent.choose_action(observationOld)
            #print(f"---STEP: {idx_steps} ---")
            #print("STAMPA 1")
            observationNew, reward, done, _, ground_truth = env.step(action)
            score += reward
            agent.remember(observationOld, output, ground_truth)
            #print("STAMPA 2")
            idx_steps += 1
            # Log metrics to visualize performance
        steps_history.append(idx_steps)
        avg_steps = np.mean(steps_history[-500:])
        wandb.log({'avg_idx_steps': avg_steps, 'idx_steps': idx_steps})
        if idx_episodes % N == 0:
            #print(f"---LEARN {idx_episodes} ---")
            agent.learn()
            agent.save_models()
            learn_iters += 1
        #print("STAMPA 3")
        observationOld = observationNew
        score_history.append(score)
        avg_score = np.mean(score_history[-500:])
        #print("STAMPA 4")
        #Per futura parte di ricerca salvare il modello quando l'avg_score è maggiore del best
        #if avg_score > best_score:
            #best_score = avg_score
        
        #print("STAMPA 5")
        idx_episodes += 1
        idx_globalSteps += idx_steps

        print('episode', idx_episodes, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', idx_globalSteps, 'learning_steps', learn_iters)

        # Log metrics to visualize performance
        wandb.log({'Train avg_score': avg_score, 'Score': score})

    run.finish()

    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve_average(x, score_history, "temp\\learning_curve_average_torch_myModel_2.0.jpg")
    plot_learning_curve(x, score_history, "temp\\learning_curve_torch_myModel_2.0.jpg")



    