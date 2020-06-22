#!/usr/bin/env python3

import gym
from termcolor import colored

env = gym.make('gym_grasper:Grasper-v0')

N_EPISODES = 10  
N_STEPS = 200

for episode in range(1, N_EPISODES+1):
    obs = env.reset()
    print('Starting episode {}!'.format(episode))
    for step in range(N_STEPS):
        print('#################################################################')
        print(colored('EPISODE {} STEP {}'.format(episode, step), color='white', attrs=['bold']))
        print('#################################################################')
        action = env.action_space.sample()
        # action = [100,100]
        observation, reward, done, _ = env.step(action)

env.close()

print('Finished.')