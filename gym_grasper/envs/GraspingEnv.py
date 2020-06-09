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
from pathlib import Path

class GraspEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, file='/UR5+gripper/UR5gripper_v2.xml', mode='reacher'):
        self.initialized = False
        self.IMAGE_SIZE = 500
        self.task_mode = mode
        self.step_called = 0
        utils.EzPickle.__init__(self)
        path = os.path.realpath(__file__)
        path = str(Path(path).parent.parent.parent) + file
        mujoco_env.MujocoEnv.__init__(self, path, 5)
        # render once to initialize a viewer object
        self.render()
        self.controller = MJ_Controller(self.model, self.sim, self.viewer)
        self.initialized = True


    def step(self, action):
        """
        Lets the agent execute the action.
        Depending on the value set when calling mujoco_env.MujocoEnv.__init__(), one step of the agent will correspond to
        frame_skip steps in the simulation. 

        Args:
            action: The action to be performed.

        Returns:
            observation: np-array containing the camera image data
            rewards: The reward obtained 
            done: Flag indicating weather the episode has finished or not 
            info: Extra info
        """

        try:
            assert len(action) == self.action_space.shape[0], 'Wrong action dimensions, should be an array of size {}'.format(self.action_space.shape[0])
            self.do_simulation(action, self.frame_skip)

            # Parent class will step once during init to set up the observation space, controller is not yet available at that time.
            # Therefore we simply return an array of zeros of the appropriate size. 
            if not self.initialized:
                observation = np.zeros((self.IMAGE_SIZE, self.IMAGE_SIZE))
            else:
                observation = self.get_observation()

            # Implementing reacher task as a first tryout.
            if self.task_mode == 'reacher':
                vec = self.get_body_com("gripperfinger_middle_link_3")-self.get_body_com("pick_box_1")
                # minimize distance
                reward_dist = - np.linalg.norm(vec)
                # minimize actuator activations
                reward_ctrl = - np.square(action).sum()
                reward = reward_dist + reward_ctrl


            done = False

            info = {}
            self.step_called += 1

            return observation, reward, done, info

        except Exception as e:
            print(e)
            print(traceback.format_exc())
            print('Could not execute step method.')


    def get_observation(self, show=True):
        """
        Uses the controllers get_image_data method to return an top-down image (as a np-array).

        Args:
            show: If True, displays the observation in an cv2 window.
        """
        rgb, depth = self.controller.get_image_data(width=self.IMAGE_SIZE, height=self.IMAGE_SIZE, show=show)
        return rgb


    def reset_model(self):
        """
        Method to perform additional reset steps and return an observation.
        Gets called in the parent classes reset method.
        """

        # Use the controller to move back to a starting position. In this case just use the default initial controller setpoints.
        self.controller.move_group_to_joint_target()

        # if self.task_mode == 'reacher':
        #     box_x = np.random.uniform(low=-0.25, high=0.25)
        #     box_y = np.random.uniform(low=-0.15, high=0.15)
        #     coordinates = [box_x, box_y]



        # return an observation image
        return self.get_observation()


    def close(self):
        mujoco_env.MujocoEnv.close()
        cv.destroyAllWindows()


if __name__ == '__main__':

    N_EPISODES = 3  
    N_STEPS = 200
    env = GraspEnv()


    for episode in range(1, N_EPISODES+1):
        obs = env.reset()
        print('Starting episode {}!'.format(episode))
        for steps in range(N_STEPS):
            action = env.action_space.sample()
            observation, reward, done, _ = env.step(action)

    env.close()
    # cv.destroyAllWindows()
