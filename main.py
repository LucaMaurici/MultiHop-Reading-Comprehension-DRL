import numpy as np
from PPO import Agent
from utils import plot_learning_curve
from MultiHopEnvironment import MultiHopEnvironment

if __name__ == '__main__':
    env = MultiHopEnvironment()
    state = env.reset()

    N = 30
    n_actions = 8
    batch_size = 15
    n_epochs = 2
    alpha = 0.0003
    agent = Agent(batch_size=batch_size, alpha=alpha, n_epochs=n_epochs)

    print(agent.actor)

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)



    