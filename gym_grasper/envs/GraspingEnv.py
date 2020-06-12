#!/usr/bin/env python3

# Author: Paul Daniel (pdd@mp.aau.dk)
import sys
sys.path.insert(0, '..')
import os
import time
import  cv2 as cv
import numpy as np
import mujoco_py
from gym.envs.mujoco import mujoco_env
from gym import utils
from gym_grasper.controller.MujocoController import MJ_Controller
import traceback
from pathlib import Path

class GraspEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, file='/UR5+gripper/UR5gripper_reacher.xml', mode='reacher'):
        self.initialized = False
        self.IMAGE_WIDTH = 500
        self.IMAGE_HEIGHT = 500
        self.task_mode = mode
        self.step_called = 0
        utils.EzPickle.__init__(self)
        path = os.path.realpath(__file__)
        path = str(Path(path).parent.parent.parent)
        full_path = path + file
        mujoco_env.MujocoEnv.__init__(self, full_path, 2)
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
            # self.do_simulation(action, 1)
            self.do_simulation(action, self.frame_skip)

            # Parent class will step once during init to set up the observation space, controller is not yet available at that time.
            # Therefore we simply return an array of zeros of the appropriate size. 
            if not self.initialized:
                observation = np.zeros((self.IMAGE_WIDTH, self.IMAGE_HEIGHT, 3))
                reward = 0
            else:
                observation = self.get_observation()

            # Implementing reacher task as a first tryout.
            if self.task_mode == 'reacher' and self.initialized:
                vec = self.get_body_com("gripperfinger_middle_link_3")-self.get_body_com("target")
                # minimize distance
                reward_dist = - np.linalg.norm(vec)
                # minimize actuator activations
                reward_ctrl = - np.square(action).sum()
                reward = reward_dist + reward_ctrl

                # if self.step_called%100==0:
                #     print('Distance to target: ', np.linalg.norm(vec))
                #     print('reward: ', reward)


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

        rgb, depth = self.controller.get_image_data(width=self.IMAGE_WIDTH, height=self.IMAGE_HEIGHT, show=show)
        # return np.zeros((20,20))
        return rgb


    def reset_model(self):
        """
        Method to perform additional reset steps and return an observation.
        Gets called in the parent classes reset method.
        """

        if self.task_mode == 'reacher':
            target_values = []
            target_values.append(np.random.uniform(low=-0.2, high=0.2))
            target_values.append(np.random.uniform(low=-0.2, high=0.05))
            target_values.append(np.random.uniform(low=-0.15, high=0.15))
            qpos = self.data.qpos
            qvel = self.data.qvel
            qpos[-3:] = target_values
            # Option 1: Just set the desired starting joint angles
            qpos[self.controller.actuated_joint_ids] = [-1.57, -1.57, 1.57, -0.8, 0.5, 1.0, 0.2, 0.2, 0.0, -0.1]
            self.set_state(qpos, qvel)

        # Option 2: Use the controller to move back to a starting position. In this case just use the default initial controller setpoints.
        # self.controller.move_group_to_joint_target(group='Arm')

        # return an observation image
        # TODO: include depth data in observation
        return self.get_observation()


    def close(self):
        mujoco_env.MujocoEnv.close(self)
        cv.destroyAllWindows()


    def print_info(self):
        print('Model timestep:', self.model.opt.timestep)
        print('Set number of frames skipped: ', self.frame_skip)
        print('dt = timestep * frame_skip: ', self.dt)
        print('Frames per second = 1/dt: ', self.metadata['video.frames_per_second'])
