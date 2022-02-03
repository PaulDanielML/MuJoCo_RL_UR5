#!/usr/bin/env python3

# Author: Paul Daniel (pdd@mp.aau.dk)
import sys

sys.path.insert(0, "..")
import os
import time
import math
import cv2 as cv
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
from decorators import *
from pyquaternion import Quaternion


class GraspEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    # def __init__(self, file='/UR5+gripper/UR5gripper_2_finger.xml', image_width=200, image_height=200, show_obs=True, demo=False, render=False):
    # def __init__(self, file='/UR5+gripper/UR5gripper_2_finger.xml', image_width=200, image_height=200, show_obs=True, demo=False, render=False):
    def __init__(
        self,
        file="/UR5+gripper/UR5gripper_2_finger_many_objects.xml",
        image_width=200,
        image_height=200,
        show_obs=True,
        demo=False,
        render=False,
    ):
        self.initialized = False
        self.IMAGE_WIDTH = image_width
        self.IMAGE_HEIGHT = image_height
        self.rotations = {0: 0, 1: 30, 2: 60, 3: 90, 4: -30, 5: -60}
        self.action_space_type = "multidiscrete"
        self.step_called = 0
        utils.EzPickle.__init__(self)
        path = os.path.realpath(__file__)
        path = str(Path(path).parent.parent.parent)
        full_path = path + file
        mujoco_env.MujocoEnv.__init__(self, full_path, 1)
        if render:
            # render once to initialize a viewer object
            self.render()
        self.controller = MJ_Controller(self.model, self.sim, self.viewer)
        self.initialized = True
        self.grasp_counter = 0
        self.show_observations = show_obs
        self.demo_mode = demo
        self.TABLE_HEIGHT = 0.91
        self.render = render

    def __repr__(self):
        return f"GraspEnv(obs height={self.IMAGE_HEIGHT}, obs_width={self.IMAGE_WIDTH}, AS={self.action_space_type})"

    def step(self, action, record_grasps=False, markers=False, action_info="no info"):
        """
        Lets the agent execute the action.

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
            # self.current_observation = np.zeros((200,200,4))
            self.current_observation = defaultdict()
            self.current_observation["rgb"] = np.zeros((self.IMAGE_WIDTH, self.IMAGE_HEIGHT, 3))
            self.current_observation["depth"] = np.zeros((self.IMAGE_WIDTH, self.IMAGE_HEIGHT))
            reward = 0
        else:
            if self.step_called == 1:
                self.current_observation = self.get_observation(show=False)

            if self.action_space_type == "discrete":
                x = action % self.IMAGE_WIDTH
                y = action // self.IMAGE_WIDTH

            elif self.action_space_type == "multidiscrete":
                x = action[0] % self.IMAGE_WIDTH
                y = action[0] // self.IMAGE_WIDTH
                rotation = action[1]

            # Depth value for the pixel corresponding to the action
            depth = self.current_observation["depth"][y][x]

            coordinates = self.controller.pixel_2_world(
                pixel_x=x, pixel_y=y, depth=depth, height=self.IMAGE_HEIGHT, width=self.IMAGE_WIDTH
            )

            print(
                colored(
                    "Action ({}): Pixel X: {}, Pixel Y: {}, Rotation: {} ({} deg)".format(
                        action_info, x, y, rotation, self.rotations[rotation]
                    ),
                    color="blue",
                    attrs=["bold"],
                )
            )
            print(
                colored(
                    "Transformed into world coordinates: {}".format(coordinates[:2]),
                    color="blue",
                    attrs=["bold"],
                )
            )

            # Check for coordinates we don't need to try to save some time
            if coordinates[2] < 0.8 or coordinates[1] > -0.3:
                print(
                    colored(
                        "Skipping execution due to bad depth value!", color="red", attrs=["bold"]
                    )
                )
                # Binary reward
                reward = 0

            else:
                grasped_something = self.move_and_grasp(
                    coordinates,
                    rotation,
                    render=self.render,
                    record_grasps=record_grasps,
                    markers=markers,
                )

                reward = 1 if grasped_something else 0
                if grasped_something != "demo":
                    print(
                        colored(
                            "Reward received during step: {}".format(reward),
                            color="yellow",
                            attrs=["bold"],
                        )
                    )

            self.current_observation = self.get_observation(show=self.show_observations)

        self.step_called += 1

        return self.current_observation, reward, done, info

    def _set_action_space(self):
        if self.action_space_type == "discrete":
            size = self.IMAGE_WIDTH * self.IMAGE_HEIGHT
            self.action_space = spaces.Discrete(size)
        elif self.action_space_type == "multidiscrete":
            self.action_space = spaces.MultiDiscrete(
                [self.IMAGE_HEIGHT * self.IMAGE_WIDTH, len(self.rotations)]
            )

        return self.action_space

    def set_grasp_position(self, position):
        """
        Legacy method, not used in the current setup. May be used to directly set the joint values to a desired position.
        """

        joint_angles = self.controller.ik(position)
        qpos = self.data.qpos
        idx = self.controller.actuated_joint_ids[self.controller.groups["Arm"]]
        for i, index in enumerate(idx):
            qpos[index] = joint_angles[i]

        self.controller.set_group_joint_target(group="Arm", target=joint_angles)

        idx_2 = self.controller.actuated_joint_ids[self.controller.groups["Gripper"]]

        open_gripper_values = [0.2, 0.2, 0.0, -0.1]

        for i, index in enumerate(idx_2):
            qpos[index] = open_gripper_values[i]

        qvel = np.zeros(len(self.data.qvel))
        self.set_state(qpos, qvel)
        self.data.ctrl[:] = 0

    def rotate_wrist_3_joint_to_value(self, degrees):
        self.controller.current_target_joint_values[5] = math.radians(degrees)
        return self.controller.move_group_to_joint_target(
            tolerance=0.05, max_steps=500, render=self.render, quiet=True
        )

    def transform_height(self, height_action, depth_height):
        return np.round(
            self.TABLE_HEIGHT + height_action * (0.1) / self.action_space.nvec[1], decimals=3
        )
        # return np.round(max(self.TABLE_HEIGHT, self.TABLE_HEIGHT + height_action * (depth_height - self.TABLE_HEIGHT)/self.action_space.nvec[1]), decimals=3)

    def move_and_grasp(
        self, coordinates, rotation, render=False, record_grasps=False, markers=False, plot=False
    ):

        # Try to move directly above target
        coordinates_1 = copy.deepcopy(coordinates)
        coordinates_1[2] = 1.1
        result1 = self.controller.move_ee(
            coordinates_1,
            max_steps=1000,
            quiet=True,
            render=render,
            marker=markers,
            tolerance=0.05,
            plot=plot,
        )
        steps1 = self.controller.last_steps
        result_pre = "Failed"
        if result1 == "success":
            result_pre = "Above target"

        # If that's not possible, move to center as pre grasp position
        if result1[:2] == "No":
            result1 = self.controller.move_ee(
                [0.0, -0.6, 1.1],
                max_steps=1000,
                quiet=True,
                render=render,
                marker=markers,
                tolerance=0.05,
                plot=plot,
            )
            steps1 = self.controller.last_steps
            if result1 == "success":
                result_pre = "Center"

        # Check result1 for max steps reached => if so it got stuck in a bad position
        if result1[:3] == "max":
            result_rotate = "Skipped"
            steps_rotate = 0
            result2 = "Skipped"
            coordinates_2 = None
            steps2 = 0
            result_grasp = False

        else:
            # Rotate gripper according to second action dimension
            result_rotate = self.rotate_wrist_3_joint_to_value(self.rotations[rotation])
            steps_rotate = self.controller.last_steps

            self.controller.open_gripper(half=True, render=render, quiet=True, plot=plot)

            # Move to grasping height
            coordinates_2 = copy.deepcopy(coordinates)
            coordinates_2[2] = max(self.TABLE_HEIGHT, coordinates_2[2] - 0.01)
            result2 = self.controller.move_ee(
                coordinates_2,
                max_steps=300,
                quiet=True,
                render=render,
                marker=markers,
                tolerance=0.01,
                plot=plot,
            )
            steps2 = self.controller.last_steps

            # If we can't reach the desired grasping position, don't grasp
            if result2[:3] == "max":
                result2 = "Could not reach target location"
                result_grasp = False

            else:
                self.controller.stay(100, render=render)
                result_grasp = self.controller.grasp(
                    render=render, quiet=True, marker=markers, plot=plot
                )

        self.controller.actuators[0][4].Kp = 10.0

        # Move back above center of table
        result3 = self.controller.move_ee(
            [0.0, -0.6, 1.1],
            max_steps=1000,
            quiet=True,
            render=render,
            plot=plot,
            marker=markers,
            tolerance=0.05,
        )
        steps3 = self.controller.last_steps

        # Move to drop position
        result4 = self.controller.move_ee(
            [0.6, 0.0, 1.15],
            max_steps=1200,
            quiet=True,
            render=render,
            plot=plot,
            marker=markers,
            tolerance=0.01,
        )
        steps4 = self.controller.last_steps

        # self.controller.stay(500)

        result_final = "Skipped"

        if result_grasp:
            if not self.demo_mode:
                # Perform check if object is in gripper
                result_final = self.controller.close_gripper(
                    max_steps=1000, render=render, quiet=True, marker=markers, plot=plot
                )
            else:
                result_final = self.controller.close_gripper(
                    max_steps=100, render=render, quiet=True, marker=markers, plot=plot
                )

        final_str = "Nothing in the gripper"
        if result_final[:3] == "max":
            final_str = "Object in the gripper"

        grasped_something = result_final[:3] == "max" and result_grasp

        if grasped_something and record_grasps:
            capture_rgb, depth = self.controller.get_image_data(
                width=1000, height=1000, camera="side"
            )
            self.grasp_counter += 1
            img_name = "Grasp_{}.png".format(self.grasp_counter)
            cv.imwrite(img_name, cv.cvtColor(capture_rgb, cv.COLOR_BGR2RGB))

        # Open gripper again
        result_open = self.controller.open_gripper(render=render, quiet=True, plot=plot)
        steps_open = self.controller.last_steps

        if grasped_something:
            self.controller.stay(200, render=render)

        # Move back to zero rotation
        result_rotate_back = self.rotate_wrist_3_joint_to_value(0)

        self.controller.actuators[0][4].Kp = 20.0

        # if self.demo_mode:
        #     self.controller.stay(200, render=render)
        #     return 'demo'

        # else:
        print("Results: ")
        print("Move to pre grasp position: ".ljust(40, " "), result_pre, ",", steps1, "steps")
        print("Rotate gripper: ".ljust(40, " "), result_rotate, ",", steps_rotate, "steps")
        print(
            f"Move to grasping position (z={np.round(coordinates_2[2], decimals=4) if isinstance(coordinates_2, np.ndarray) else 0}):".ljust(
                40, " "
            ),
            result2,
            ",",
            steps2,
            "steps",
        )
        print("Grasped anything?: ".ljust(40, " "), result_grasp)
        print("Move to center: ".ljust(40, " "), result3, ",", steps3, "steps")
        print("Move to drop position: ".ljust(40, " "), result4, ",", steps4, "steps")
        print("Final finger check: ".ljust(40, " "), final_str)
        print("Open gripper: ".ljust(40, " "), result_open, ",", steps_open, "steps")

        if result1 == result2 == result3 == result4 == result_open == "success":
            print(colored("Executed all movements successfully.", color="green", attrs=["bold"]))
        else:
            print(
                colored(
                    "Could not execute all movements successfully.", color="red", attrs=["bold"]
                )
            )

        if grasped_something:
            print(colored("Successful grasp!", color="green", attrs=["bold"]))
            return True
        else:
            print(colored("Did not grasp anything.", color="red", attrs=["bold"]))
            return False

    # @debug
    # @dict2list
    def get_observation(self, show=True):
        """
        Uses the controllers get_image_data method to return an top-down image (as a np-array).

        Args:
            show: If True, displays the observation in a cv2 window.
        """

        rgb, depth = self.controller.get_image_data(
            width=self.IMAGE_WIDTH, height=self.IMAGE_HEIGHT, show=show
        )
        depth = self.controller.depth_2_meters(depth)
        observation = defaultdict()
        observation["rgb"] = rgb
        observation["depth"] = depth

        return observation

    # @debug
    def reset_model(self, show_obs=True):
        """
        Method to perform additional reset steps and return an observation.
        Gets called in the parent classes reset method.
        """

        qpos = self.data.qpos
        qvel = self.data.qvel

        qpos[self.controller.actuated_joint_ids] = [0, -1.57, 1.57, -1.57, -1.57, 0.0, 0.3]

        n_objects = 40

        for i in range(n_objects):
            joint_name = f"free_joint_{i}"
            q_adr = self.model.get_joint_qpos_addr(joint_name)
            start, end = q_adr
            qpos[start] = np.random.uniform(low=-0.25, high=0.25)
            qpos[start + 1] = np.random.uniform(low=-0.77, high=-0.43)
            # qpos[start+2] = 1.0
            qpos[start + 2] = np.random.uniform(low=1.0, high=1.5)
            qpos[start + 3 : end] = Quaternion.random().unit.elements

        #########################################################################
        # Reset for IT4, older versions of IT5

        # n_boxes = 3
        # n_balls = 3

        # for j in ['rot', 'x', 'y', 'z']:
        #     for i in range(1,n_boxes+1):
        #         joint_name = 'box_' + str(i) + '_' + j
        #         q_adr = self.model.get_joint_qpos_addr(joint_name)
        #         if j == 'x':
        #             qpos[q_adr] = np.random.uniform(low=-0.25, high=0.25)
        #         elif j == 'y':
        #             qpos[q_adr] = np.random.uniform(low=-0.17, high=0.17)
        #         elif j == 'z':
        #             qpos[q_adr] = 0.0
        #         elif j == 'rot':
        #             start, end = q_adr
        #             qpos[start:end] = [1., 0., 0., 0.]

        #     for i in range(1,n_balls+1):
        #         joint_name = 'ball_' + str(i) + '_' + j
        #         q_adr = self.model.get_joint_qpos_addr(joint_name)
        #         if j == 'x':
        #             qpos[q_adr] = np.random.uniform(low=-0.25, high=0.25)
        #         elif j == 'y':
        #             qpos[q_adr] = np.random.uniform(low=-0.17, high=0.17)
        #         elif j == 'z':
        #             qpos[q_adr] = 0.0
        #         elif j == 'rot':
        #             start, end = q_adr
        #             qpos[start:end] = [1., 0., 0., 0.]
        #########################################################################

        self.set_state(qpos, qvel)

        self.controller.set_group_joint_target(
            group="All", target=qpos[self.controller.actuated_joint_ids]
        )

        # Turn this on for training, so the objects drop down before the observation
        self.controller.stay(1000, render=self.render)
        if self.demo_mode:
            self.controller.stay(5000, render=self.render)
        # return an observation image
        return self.get_observation(show=self.show_observations)

    def close(self):
        mujoco_env.MujocoEnv.close(self)
        cv.destroyAllWindows()

    def print_info(self):
        print("Model timestep:", self.model.opt.timestep)
        print("Set number of frames skipped: ", self.frame_skip)
        print("dt = timestep * frame_skip: ", self.dt)
        print("Frames per second = 1/dt: ", self.metadata["video.frames_per_second"])
        print("Actionspace: ", self.action_space)
        print("Observation space:", self.observation_space)
