import agent
import numpy as np
from gym import make

if __name__ == '__main__':
    env = make('MountainCarContinuous-v0')
    scores = []
    actor = agent.Agent(2, 1)
    n_episodes = 100
    
    scores = []
    for ep in range(n_episodes):
        state = env.reset()
        total_reward = 0

        done = False
        while not done:
            if ep > n_episodes - 20:
                env.render()

            state = np.array(state).squeeze()
            action = actor.act(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward
        env.close()
        scores.append(total_reward)

    print(f'mean reward for {n_episodes} episodes = ', np.mean(scores))

