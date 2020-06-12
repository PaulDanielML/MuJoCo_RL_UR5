#!/usr/bin/env python3

import gym

env = gym.make('gym_grasper:Grasper-v0')

N_EPISODES = 10  
N_STEPS = 2000

print(env.sim.model.cam_fovy[1])

for episode in range(1, N_EPISODES+1):
    obs = env.reset()
    print('Starting episode {}!'.format(episode))
    for steps in range(N_STEPS):
        action = env.action_space.sample()
        observation, reward, done, _ = env.step(action)

env.close()

print('Finished.')