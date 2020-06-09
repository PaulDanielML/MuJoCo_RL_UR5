#!/usr/bin/env python3

# Author: Paul Daniel (pdd@mp.aau.dk)

import os
import time
import  cv2 as cv
import numpy as np
import mujoco_py
from gym.envs.mujoco import mujoco_env
from gym import utils
from MujocoController import MJ_Controller
import traceback

class GraspEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, file='/UR5+gripper/UR5gripper_v2.xml'):
        self.initialized = False
        self.IMAGE_SIZE = 500
        self.step_called = 0
        utils.EzPickle.__init__(self)
        path = os.path.dirname(os.path.abspath(__file__)) + file
        mujoco_env.MujocoEnv.__init__(self, path, 20)
        # render once to initialize a viewer object
        self.render()
        self.controller = MJ_Controller(self.model, self.sim, self.viewer)
        self.initialized = True
        # print(self.init_qpos)


    def step(self, action):
        try:
            assert len(action) == self.action_space.shape[0], 'Wrong action dimensions, should be an array of size {}'.format(self.action_space.shape[0])
            self.do_simulation(action, 1)

            if not self.initialized:
                observation = np.zeros((self.IMAGE_SIZE, self.IMAGE_SIZE))
            else:
                observation = self.get_observation(show=True)

            reward = 1

            done = False

            info = {}
            self.step_called += 1
            # print('Number of times the step method has been called: {}'.format(self.step_called))
            # print(self.sim._render_context_offscreen, self.step_called)

            return observation, reward, done, info

        except Exception as e:
            print(e)
            print(traceback.format_exc())
            print('Could not execute step method.')


    def get_observation(self, show=False):
         rgb, depth = self.controller.get_image_data(width=self.IMAGE_SIZE, height=self.IMAGE_SIZE, show=show)
         # rgb, depth = self.controller.get_image_data(width=self.IMAGE_SIZE, height=self.IMAGE_SIZE)
         return rgb
         # return np.zeros((200,200))


    def reset_model(self):
        # rgb, depth = self.controller.get_image_data(show=False)
        # print(self.viewer)


        # print(self.data.ctrl[:])
        self.controller.move_group_to_joint_target()

        return self.get_observation(show=True)
        # return np.zeros((self.IMAGE_SIZE, self.IMAGE_SIZE))


    # def render(self):
        # self.controller.viewer.render()


if __name__ == '__main__':

    # env = GraspEnv()


    # action = env.action_space.sample()
    # observation, reward, done, _ = env.step(action)

    # print(env.observation_space)

    # print(observation.shape)
    # cv.imshow('RGB', observation)
    # cv.waitKey()





    N_EPISODES = 3  
    N_STEPS = 200
    env = GraspEnv()


    for episode in range(1, N_EPISODES+1):
        obs = env.reset()
        print('Starting episode {}!'.format(episode))
        for steps in range(N_STEPS):
            action = env.action_space.sample()
            observation, reward, done, _ = env.step(action)

            # cv.imshow('RGB', observation)
            # cv.waitKey(1)
            # env.render()


    cv.destroyAllWindows()
