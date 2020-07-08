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
    def __init__(self, file='/UR5+gripper/UR5gripper_2_finger.xml', image_width=200, image_height=200, show_obs=True):
        self.initialized = False
        self.IMAGE_WIDTH = image_width
        self.IMAGE_HEIGHT = image_height
        self.action_space_type = 'discrete'
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
        self.grasp_counter = 0
        self.show_observations = show_obs
       

    def step(self, action, render=False, record_grasps=False, markers=False):
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

            if self.action_space_type == 'discrete':
                x = action % self.IMAGE_WIDTH
                y = action // self.IMAGE_WIDTH

            elif self.action_space_type == 'multidiscrete':
                x = action[0]
                y = action[1]

            coordinates = self.controller.pixel_2_world(pixel_x=x, pixel_y=y, depth=self.current_observation['depth'][y][x], height=self.IMAGE_HEIGHT, width=self.IMAGE_WIDTH)

            print(colored('Action: Pixel X: {}, Pixel Y: {}'.format(x, y), color='blue', attrs=['bold']))
            print(colored('Transformed into world coordinates: {}'.format(coordinates), color='blue', attrs=['bold']))
            
            # Check for coordinates we don't need to try
            if coordinates[2] < 0.8 or coordinates[2] > 1.0 or coordinates[1] > -0.3:
                print(colored('Skipping execution due to bad depth value!', color='red', attrs=['bold']))
                # Binary reward
                reward = 0
                # Old reward structure 
                # reward = -10

            else:
                grasped_something = self.move_and_grasp(coordinates, render=render, record_grasps=record_grasps, markers=markers)

                if grasped_something:
                    # Binary reward
                    reward = 1
                    # Old reward structure 
                    # reward = 100
                else:
                    reward = 0

            self.current_observation = self.get_observation(show=self.show_observations)

            print(colored('Reward received during step: {}'.format(reward), color='yellow', attrs=['bold']))

        self.step_called += 1

        # for _ in range(self.frame_skip):
            # self.sim.step()

        return self.current_observation, reward, done, info


    def _set_action_space(self):
        if self.action_space_type == 'discrete':
            size = self.IMAGE_WIDTH * self.IMAGE_HEIGHT
            self.action_space = spaces.Discrete(size)
        elif self.action_space_type == 'multidiscrete':
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


    def move_and_grasp(self, coordinates, render=False, record_grasps=False, markers=False, plot=False):

        # Try to move directly above target    
        coordinates_1 = copy.deepcopy(coordinates)
        coordinates_1[2] = 1.1
        result1 = self.controller.move_ee(coordinates_1, max_steps=1000, quiet=True, render=render, marker=markers, tolerance=0.05, plot=plot)
        steps1 = self.controller.last_steps
        result_pre = 'Failed'
        if result1 == 'success':
            result_pre = 'Above target'

        # If that's not possible, move to center as pre grasp position
        if result1[:2] == 'No':
            result1 = self.controller.move_ee([0.0, -0.6, 1.1], max_steps=1000, quiet=True, render=render, marker=markers, tolerance=0.05, plot=plot)
            steps1 = self.controller.last_steps
            if result1 == 'success':
                result_pre = 'Center'

        # Move to grasping height
        coordinates_2 = copy.deepcopy(coordinates)
        coordinates_2[2] = 0.91
        # coordinates_2[2] = 0.895

        result2 = self.controller.move_ee(coordinates_2, max_steps=1000, quiet=True, render=render, marker=markers, tolerance=0.01, plot=plot)
        steps2 = self.controller.last_steps
        
        # test = self.controller.get_image_data(show=True, height=1000, width=1000)

        self.controller.stay(100, render=render)

        result_grasp = self.controller.grasp(render=render, quiet=True, marker=markers, plot=plot)

        self.controller.actuators[0][4].Kp = 10.0

        result3 = self.controller.move_ee([0.0, -0.6, 1.1], max_steps=1000, quiet=True, render=render, plot=plot, marker=markers, tolerance=0.05)
        steps3 = self.controller.last_steps

        result4 = self.controller.move_ee([0.6, 0.0, 1.15], max_steps=1200, quiet=True, render=render, plot=plot, marker=markers, tolerance=0.01)
        steps4 = self.controller.last_steps

        # self.controller.stay(500)

        result_final = self.controller.close_gripper(max_steps=1000, render=render, quiet=True, marker=markers, plot=plot)

        if result_final == 'success':
            final_str = 'Nothing in the gripper'
        else:
            final_str = 'Object in the gripper'

        grasped_something = result_final[:3] == 'max' and result_grasp

        if grasped_something and record_grasps:
            capture_rgb, depth = self.controller.get_image_data(width=1000, height=1000, camera='side')
            self.grasp_counter += 1
            img_name = 'Grasp_{}.png'.format(self.grasp_counter)
            cv.imwrite(img_name, cv.cvtColor(capture_rgb, cv.COLOR_BGR2RGB))

        result_open = self.controller.open_gripper(render=render, quiet=True, plot=plot)
        steps_open = self.controller.last_steps

        self.controller.actuators[0][4].Kp = 20.0

        print('Results: ')
        print('Move to pre grasp position: '.ljust(40, ' '), result_pre, ',', steps1, 'steps')
        print('Move to grasping position: '.ljust(40, ' '), result2, ',', steps2, 'steps')
        print('Grasped anything?: '.ljust(40, ' '), result_grasp)
        print('Move to center: '.ljust(40, ' '), result3, ',', steps3, 'steps')
        print('Move to drop position: '.ljust(40, ' '), result4, ',', steps4, 'steps')
        print('Final finger check: '.ljust(40, ' '), final_str)
        print('Open gripper: '.ljust(40, ' '), result_open, ',', steps_open, 'steps')


        if result1 == result2 == result3 == result4 == result_open == 'success':
            print(colored('Executed all movements successfully.', color='green', attrs=['bold']))
        else:
            print(colored('Could not execute all movements successfully.', color='red', attrs=['bold']))

        if grasped_something:
            print(colored('Successful grasp!', color='green', attrs=['bold'])) 
            return True         
        else:
            print(colored('Did not grasp anything.', color='red', attrs=['bold']))
            return False   



    def get_observation(self, show=True):
        """
        Uses the controllers get_image_data method to return an top-down image (as a np-array).

        Args:
            show: If True, displays the observation in a cv2 window.
        """

        rgb, depth = self.controller.get_image_data(width=self.IMAGE_WIDTH, height=self.IMAGE_HEIGHT, show=show)
        # rgb, depth = self.controller.get_image_data(width=self.IMAGE_WIDTH, height=self.IMAGE_HEIGHT, show=show)
        depth = self.controller.depth_2_meters(depth)
        observation = defaultdict()
        observation['rgb'] = rgb
        observation['depth'] = depth

        return observation


    def reset_model(self, show_obs=True):
        """
        Method to perform additional reset steps and return an observation.
        Gets called in the parent classes reset method.
        """

        qpos = self.data.qpos
        qvel = self.data.qvel

        qpos[self.controller.actuated_joint_ids] = [0, -1.57, 1.57, -1.57, -1.57, 1.0, 0.3]

        n_boxes = 3
        n_balls = 3

        for j in ['rot', 'x', 'y', 'z']:
            for i in range(1,n_boxes+1):
                joint_name = 'box_' + str(i) + '_' + j
                q_adr = self.model.get_joint_qpos_addr(joint_name)
                if j == 'x':
                    qpos[q_adr] = np.random.uniform(low=-0.25, high=0.25)
                elif j == 'y':
                    qpos[q_adr] = np.random.uniform(low=-0.17, high=0.17)
                elif j == 'z':
                    qpos[q_adr] = 0.0
                elif j == 'rot':
                    start, end = q_adr
                    qpos[start:end] = [1., 0., 0., 0.]

            for i in range(1,n_balls+1):
                joint_name = 'ball_' + str(i) + '_' + j
                q_adr = self.model.get_joint_qpos_addr(joint_name)
                if j == 'x':
                    qpos[q_adr] = np.random.uniform(low=-0.25, high=0.25)
                elif j == 'y':
                    qpos[q_adr] = np.random.uniform(low=-0.17, high=0.17)
                elif j == 'z':
                    qpos[q_adr] = 0.0
                elif j == 'rot':
                    start, end = q_adr
                    qpos[start:end] = [1., 0., 0., 0.]

        self.set_state(qpos, qvel)

        self.controller.set_group_joint_target(group='All', target= qpos[self.controller.actuated_joint_ids])

        # return an observation image
        return self.get_observation(show=self.show_observations)


    def close(self):
        mujoco_env.MujocoEnv.close(self)
        cv.destroyAllWindows()


    def print_info(self):
        print('Model timestep:', self.model.opt.timestep)
        print('Set number of frames skipped: ', self.frame_skip)
        print('dt = timestep * frame_skip: ', self.dt)
        print('Frames per second = 1/dt: ', self.metadata['video.frames_per_second'])
        print('Actionspace: ', self.action_space)
        print('Observation space:', self.observation_space)
