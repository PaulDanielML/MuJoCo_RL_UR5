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
from gym import utils, spaces
from gym_grasper.controller.MujocoController import MJ_Controller
import traceback
from pathlib import Path
import copy
from collections import defaultdict
from termcolor import colored


class GraspEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, file='/UR5+gripper/UR5gripper_v2.xml', mode='reacher'):
        self.initialized = False
        self.IMAGE_WIDTH = 200
        self.IMAGE_HEIGHT = 200
        self.task_mode = mode
        self.step_called = 0
        utils.EzPickle.__init__(self)
        path = os.path.realpath(__file__)
        path = str(Path(path).parent.parent.parent)
        full_path = path + file
        mujoco_env.MujocoEnv.__init__(self, full_path, 1)
        # render once to initialize a viewer object
        self.render()
        self.controller = MJ_Controller(self.model, self.sim, self.viewer)
        self.initialized = True
        self.result_dict = {'success': True}
       

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
          
        done = False
        info = {}
        # Parent class will step once during init to set up the observation space, controller is not yet available at that time.
        # Therefore we simply return a dictionary of zeros of the appropriate size. 
        if not self.initialized:
            self.current_observation = defaultdict()
            self.current_observation['rgb'] = np.zeros((self.IMAGE_WIDTH,self.IMAGE_HEIGHT,3))
            self.current_observation['depth'] = np.zeros((self.IMAGE_WIDTH,self.IMAGE_HEIGHT))
            reward = 0
        else:
            if self.step_called == 1:
                self.current_observation = self.get_observation(show=False)

            x = action[0]
            y = action[1]

            coordinates = self.controller.pixel_2_world(pixel_x=x, pixel_y=y, depth=self.current_observation['depth'][y][x], height=self.IMAGE_HEIGHT, width=self.IMAGE_WIDTH)


            print(colored('Action: Pixel X: {}, Pixel Y: {}'.format(x, y), color='blue', attrs=['bold']))
            print(colored('Transformed into world coordinates: {}'.format(coordinates), color='blue', attrs=['bold']))
            # Check for coordinates we don't need to try
            if coordinates[2] < 0.8 or coordinates[2] > 1.0 or coordinates[1] > -0.3:
                print(colored('Skipping execution due to bad depth value!', color='red', attrs=['bold']))
                reward = -10
                return self.current_observation, reward, done, info

            grasped_something = self.move_and_grasp(coordinates, render=False)

            if grasped_something:
                reward = 100
            else:
                reward = 0

            self.current_observation = self.get_observation(show=True)

        self.step_called += 1

        # for _ in range(self.frame_skip):
            # self.sim.step()

        return self.current_observation, reward, done, info


    def step2(self, action):
        """
        Alternative step method for simply setting the calculated joint values instead of moving to them. 
        Potentially way faster, might be used later for actual training.
        """

            # Parent class will step once during init to set up the observation space, controller is not yet available at that time.
            # Therefore we simply return an array of zeros of the appropriate size. 
        if not self.initialized:
            observation = np.zeros((self.IMAGE_WIDTH, self.IMAGE_HEIGHT, 3))
            reward = 0
        else:
            observation = self.get_observation(show=False)

            x = action[0]
            y = action[1]

            coordinates = self.controller.pixel_2_world(pixel_x=x, pixel_y=y, depth=observation['depth'][y][x], height=self.IMAGE_HEIGHT, width=self.IMAGE_WIDTH)

            coordinates[2] -= 0.05

            self.set_grasp_position(coordinates)

            self.controller.grasp(render=True)

            reward = 0
            observation = self.get_observation()



        done = False



        info = {}
        self.step_called += 1

        for _ in range(self.frame_skip):
            self.sim.step()

        return observation, reward, done, info

        # except Exception as e:
        #     print(e)
        #     print(traceback.format_exc())
        #     print('Could not execute step method.')



    def _set_action_space(self):
        self.action_space = spaces.MultiDiscrete([self.IMAGE_HEIGHT, self.IMAGE_WIDTH])
        return self.action_space


    def set_grasp_position(self, position):
        joint_angles = self.controller.ik(position)
        qpos = self.data.qpos
        idx = self.controller.actuated_joint_ids[self.controller.groups['Arm']]
        for i, index in enumerate(idx):
            qpos[index] = joint_angles[i]

        self.controller.set_group_joint_target(group='Arm', target=joint_angles)

        idx_2 = self.controller.actuated_joint_ids[self.controller.groups['Gripper']]

        open_gripper_values = [0.2, 0.2, 0.0, -0.1]

        for i, index in enumerate(idx_2):
            qpos[index] = open_gripper_values[i]

        qvel = np.zeros(len(self.data.qvel))
        self.set_state(qpos, qvel)
        self.data.ctrl[:] = 0


    def move_and_grasp(self, coordinates, render=False):

        # Move to pre grasping position and height
        # coordinates_1 = copy.deepcopy(coordinates)
        # coordinates_1[2] = 1.1
        result1 = self.controller.move_ee([0.0, -0.6, 1.1], marker=True, max_steps=10000, quiet=True, render=render, plot=False)
        # self.controller.move_ee(coordinates_1, marker=True, max_steps=1000)

        result_open = self.controller.open_gripper(render=render)

        # Move to grasping height
        coordinates_2 = copy.deepcopy(coordinates)
        coordinates_2[2] = 0.895

        result2 = self.controller.move_ee(coordinates_2, marker=True, max_steps=10000, quiet=True, render=render)

        result_grasp = self.controller.grasp(render=render)

        # Move up again
        # coordinates_3 = copy.deepcopy(coordinates)
        # coordinates_3[2] = 1.1

        result3 = self.controller.move_ee([0.0, -0.6, 1.1], marker=True, max_steps=10000, quiet=True, render=render)

        # self.controller.move_ee(coordinates_3, marker=True, max_steps=1000)

        result_final = self.controller.close_gripper(max_steps=3000, render=render)

        print('Results: ')
        print('Move to pre-grasp: ', result1)
        print('Open gripper: ', result_open)
        print('Move to grasping position: ', result2)
        print('Grasping: ', result_grasp)
        print('Move to post-grasp: ', result3)
        print('Final finger check: ', result_final)


        if result1 == result2 == result3 == result_open == 'success':
            print(colored('Executed all movements successfully.', color='green', attrs=['bold']))
        else:
            print(colored('Could not execute all movements successfully.', color='red', attrs=['bold']))

        if result_final == 'max_steps reached' and result_grasp:
            print(colored('Successful grasp!', color='green', attrs=['bold'])) 
            capture_rgb, depth = self.controller.get_image_data(width=1000, height=1000)
            cv.imwrite('Grasp.png', capture_rgb)

            return True         
        else:
            print(colored('Did not grasp anything.', color='red', attrs=['bold']))
            return False   
            

        # self.data.ctrl[:] = 0


    def get_observation(self, show=True):
        """
        Uses the controllers get_image_data method to return an top-down image (as a np-array).

        Args:
            show: If True, displays the observation in an cv2 window.
        """

        rgb, depth = self.controller.get_image_data(width=self.IMAGE_WIDTH, height=self.IMAGE_HEIGHT, show=show)
        # rgb, depth = self.controller.get_image_data(width=self.IMAGE_WIDTH, height=self.IMAGE_HEIGHT, show=show)
        depth = self.controller.depth_2_meters(depth)
        observation = defaultdict()
        observation['rgb'] = rgb
        observation['depth'] = depth
        return observation


    def reset_model(self):
        """
        Method to perform additional reset steps and return an observation.
        Gets called in the parent classes reset method.
        """

        # if self.task_mode == 'reacher':
        #     target_values = []
        #     target_values.append(np.random.uniform(low=-0.2, high=0.2))
        #     target_values.append(np.random.uniform(low=-0.2, high=0.05))
        #     target_values.append(np.random.uniform(low=-0.15, high=0.15))
        #     qpos = self.data.qpos
        #     qvel = self.data.qvel
        #     qpos[-3:] = target_values
        #     # Option 1: Just set the desired starting joint angles
        #     qpos[self.controller.actuated_joint_ids] = [-1.57, -1.57, 1.57, -0.8, 0.5, 1.0, 0.2, 0.2, 0.0, -0.1]
        #     self.set_state(qpos, qvel)

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
