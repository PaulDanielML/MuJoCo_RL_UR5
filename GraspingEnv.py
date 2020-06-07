#!/usr/bin/env python3

# Author: Paul Daniel (pdd@mp.aau.dk)

import os
import  cv2 as cv
import numpy as np
import mujoco_py
from gym.envs.mujoco import mujoco_env
from gym import utils
from MujocoController import MJ_Controller

class GraspEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, file='/UR5+gripper/UR5gripper_v2.xml'):
        utils.EzPickle.__init__(self)
        path = os.path.dirname(os.path.abspath(__file__)) + file
        self.initialized = False
        mujoco_env.MujocoEnv.__init__(self, path, 1)
        self.viewer = mujoco_py.MjViewer(self.sim)
        self.controller = MJ_Controller(self.model, self.sim, self.viewer)
        self.initialized = True
        print('Action space: ', self.action_space)
        print(self.viewer)

    def step(self, action):
        self.do_simulation(action, 10)


        if self.initialized:
            rgb, depth = self.controller.get_image_data()
            observation = rgb

        else:
            observation = np.zeros((400,400))
        return observation, 1, False, '_'


    def random_action(self, group='All'):
        return np.random.uniform(low=-1.0, high=1.0, size=len(self.controller.groups[group]))


    def reset_model(self):
        rgb, depth = self.controller.get_image_data(show=False)
        print(self.viewer)

        return rgb


if __name__ == '__main__':

    N_EPISODES = 5  
    N_STEPS = 200
    env = GraspEnv()
    done = False

    state = env.reset()


    for episode in range(N_EPISODES):
        for step in range(N_STEPS):
            action = env.random_action()
            next_state, reward, done, _ = env.step(action)
            # cv.imshow('rbg', cv.cvtColor(next_state, cv.COLOR_BGR2RGB))
            # env.render(camera_name='main1')
            env.viewer.render()

        # env.render()

        env.reset()
        cv.destroyAllWindows()


    print(env.random_action())
    print('worked')
