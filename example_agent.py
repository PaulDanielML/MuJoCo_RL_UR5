#!/usr/bin/env python3

import gym

env = gym.make('gym_grasper:Grasper-v0')

N_EPISODES = 3  
N_STEPS = 200


for episode in range(1, N_EPISODES+1):
    obs = env.reset()
    print('Starting episode {}!'.format(episode))
    for steps in range(N_STEPS):
        action = env.action_space.sample()
        observation, reward, done, _ = env.step(action)

env.close()

print('worked.')