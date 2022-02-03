#!/usr/bin/env python3

# Author: Paul Daniel (pdd@mp.aau.dk)

from collections import defaultdict
import os
from pathlib import Path
import mujoco_py as mp
import time
import numpy as np
from simple_pid import PID
from termcolor import colored
import ikpy
from pyquaternion import Quaternion
import cv2 as cv
import matplotlib.pyplot as plt
import copy
from decorators import debug


class MJ_Controller(object):
    """
    Class for control of an robotic arm in MuJoCo.
    It can be used on its own, in which case a new model, simulation and viewer will be created.
    It can also be passed these objects when creating an instance, in which case the class can be used
    to perform tasks on an already instantiated simulation.
    """

    def __init__(self, model=None, simulation=None, viewer=None):
        path = os.path.realpath(__file__)
        path = str(Path(path).parent.parent.parent)
        if model is None:
            self.model = mp.load_model_from_path(path + "/UR5+gripper/UR5gripper_2_finger.xml")
        else:
            self.model = model
        self.sim = mp.MjSim(self.model) if simulation is None else simulation
        self.viewer = mp.MjViewer(self.sim) if viewer is None else viewer
        self.create_lists()
        self.groups = defaultdict(list)
        self.groups["All"] = list(range(len(self.sim.data.ctrl)))
        self.create_group("Arm", list(range(5)))
        self.create_group("Gripper", [6])
        self.actuated_joint_ids = np.array([i[2] for i in self.actuators])
        self.reached_target = False
        self.current_output = np.zeros(len(self.sim.data.ctrl))
        self.image_counter = 0
        self.ee_chain = ikpy.chain.Chain.from_urdf_file(path + "/UR5+gripper/ur5_gripper.urdf")
        self.cam_matrix = None
        self.cam_init = False
        self.last_movement_steps = 0
        # self.move_group_to_joint_target()

    def create_group(self, group_name, idx_list):
        """
        Allows the user to create custom objects for controlling groups of joints.
        The method show_model_info can be used to get lists of joints and actuators.

        Args:
            group_name: String defining the désired name of the group.
            idx_list: List containing the IDs of the actuators that will belong to this group.
        """

        try:
            assert len(idx_list) <= len(self.sim.data.ctrl), "Too many joints specified!"
            assert (
                group_name not in self.groups.keys()
            ), "A group with name {} already exists!".format(group_name)
            assert np.max(idx_list) <= len(
                self.sim.data.ctrl
            ), "List contains invalid actuator ID (too high)"

            self.groups[group_name] = idx_list
            print("Created new control group '{}'.".format(group_name))

        except Exception as e:
            print(e)
            print("Could not create a new group.")

    def show_model_info(self):
        """
        Displays relevant model info for the user, namely bodies, joints, actuators, as well as their IDs and ranges.
        Also gives info on which actuators control which joints and which joints are included in the kinematic chain,
        as well as the PID controller info for each actuator.
        """

        print("\nNumber of bodies: {}".format(self.model.nbody))
        for i in range(self.model.nbody):
            print("Body ID: {}, Body Name: {}".format(i, self.model.body_id2name(i)))

        print("\nNumber of joints: {}".format(self.model.njnt))
        for i in range(self.model.njnt):
            print(
                "Joint ID: {}, Joint Name: {}, Limits: {}".format(
                    i, self.model.joint_id2name(i), self.model.jnt_range[i]
                )
            )

        print("\nNumber of Actuators: {}".format(len(self.sim.data.ctrl)))
        for i in range(len(self.sim.data.ctrl)):
            print(
                "Actuator ID: {}, Actuator Name: {}, Controlled Joint: {}, Control Range: {}".format(
                    i,
                    self.model.actuator_id2name(i),
                    self.actuators[i][3],
                    self.model.actuator_ctrlrange[i],
                )
            )

        print("\nJoints in kinematic chain: {}".format([i.name for i in self.ee_chain.links]))

        print("\nPID Info: \n")
        for i in range(len(self.actuators)):
            print(
                "{}: P: {}, I: {}, D: {}, setpoint: {}, sample_time: {}".format(
                    self.actuators[i][3],
                    self.actuators[i][4].tunings[0],
                    self.actuators[i][4].tunings[1],
                    self.actuators[i][4].tunings[2],
                    self.actuators[i][4].setpoint,
                    self.actuators[i][4].sample_time,
                )
            )

        print("\n Camera Info: \n")
        for i in range(self.model.ncam):
            print(
                "Camera ID: {}, Camera Name: {}, Camera FOV (y, degrees): {}, Position: {}, Orientation: {}".format(
                    i,
                    self.model.camera_id2name(i),
                    self.model.cam_fovy[i],
                    self.model.cam_pos0[i],
                    self.model.cam_mat0[i],
                )
            )

    def create_lists(self):
        """
        Creates some basic lists and fill them with initial values. This function is called in the class costructor.
        The following lists/dictionaries are created:

        - controller_list: Contains a controller for each of the actuated joints. This is done so that different gains may be
        specified for each controller.

        - current_joint_value_targets: Same as the current setpoints for all controllers, created for convenience.

        - current_output = A list containing the ouput values of all the controllers. This list is only initiated here, its
        values are overwritten at the first simulation step.

        - actuators: 2D list, each entry represents one actuator and contains:
            0 actuator ID
            1 actuator name
            2 joint ID of the joint controlled by this actuator
            3 joint name
            4 controller for controlling the actuator
        """

        self.controller_list = []

        # Values for training
        sample_time = 0.0001
        # p_scale = 1
        p_scale = 3
        i_scale = 0.0
        i_gripper = 0
        d_scale = 0.1
        self.controller_list.append(
            PID(
                7 * p_scale,
                0.0 * i_scale,
                1.1 * d_scale,
                setpoint=0,
                output_limits=(-2, 2),
                sample_time=sample_time,
            )
        )  # Shoulder Pan Joint
        self.controller_list.append(
            PID(
                10 * p_scale,
                0.0 * i_scale,
                1.0 * d_scale,
                setpoint=-1.57,
                output_limits=(-2, 2),
                sample_time=sample_time,
            )
        )  # Shoulder Lift Joint
        self.controller_list.append(
            PID(
                5 * p_scale,
                0.0 * i_scale,
                0.5 * d_scale,
                setpoint=1.57,
                output_limits=(-2, 2),
                sample_time=sample_time,
            )
        )  # Elbow Joint
        self.controller_list.append(
            PID(
                7 * p_scale,
                0.0 * i_scale,
                0.1 * d_scale,
                setpoint=-1.57,
                output_limits=(-1, 1),
                sample_time=sample_time,
            )
        )  # Wrist 1 Joint
        self.controller_list.append(
            PID(
                5 * p_scale,
                0.0 * i_scale,
                0.1 * d_scale,
                setpoint=-1.57,
                output_limits=(-1, 1),
                sample_time=sample_time,
            )
        )  # Wrist 2 Joint
        self.controller_list.append(
            PID(
                5 * p_scale,
                0.0 * i_scale,
                0.1 * d_scale,
                setpoint=0.0,
                output_limits=(-1, 1),
                sample_time=sample_time,
            )
        )  # Wrist 3 Joint
        self.controller_list.append(
            PID(
                2.5 * p_scale,
                i_gripper,
                0.00 * d_scale,
                setpoint=0.0,
                output_limits=(-1, 1),
                sample_time=sample_time,
            )
        )  # Gripper Joint
        # self.controller_list.append(PID(10.5*p_scale, 0.2, 0.1*d_scale, setpoint=0.0, output_limits=(-1, 1), sample_time=sample_time)) # Gripper Joint
        # self.controller_list.append(PID(2*p_scale, 0.1*i_scale, 0.05*d_scale, setpoint=0.2, output_limits=(-0.5, 0.8), sample_time=sample_time)) # Finger 2 Joint 1
        # self.controller_list.append(PID(1*p_scale, 0.1*i_scale, 0.05*d_scale, setpoint=0.0, output_limits=(-0.5, 0.8), sample_time=sample_time)) # Middle Finger Joint 1
        # self.controller_list.append(PID(1*p_scale, 0.1*i_scale, 0.05*d_scale, setpoint=-0.1, output_limits=(-0.8, 0.8), sample_time=sample_time)) # Gripperpalm Finger 1 Joint

        self.current_target_joint_values = [
            self.controller_list[i].setpoint for i in range(len(self.sim.data.ctrl))
        ]

        self.current_target_joint_values = np.array(self.current_target_joint_values)

        self.current_output = [controller(0) for controller in self.controller_list]
        self.actuators = []
        for i in range(len(self.sim.data.ctrl)):
            item = [i, self.model.actuator_id2name(i)]
            item.append(self.model.actuator_trnid[i][0])
            item.append(self.model.joint_id2name(self.model.actuator_trnid[i][0]))
            item.append(self.controller_list[i])
            self.actuators.append(item)

    def actuate_joint_group(self, group, motor_values):
        try:
            assert group in self.groups.keys(), "No group with name {} exists!".format(group)
            assert len(motor_values) == len(
                self.groups[group]
            ), "Invalid number of actuator values!"
            for i, v in enumerate(self.groups[group]):
                self.sim.data.ctrl[v] = motor_values[i]

        except Exception as e:
            print(e)
            print("Could not actuate requested joint group.")

    def move_group_to_joint_target(
        self,
        group="All",
        target=None,
        tolerance=0.1,
        max_steps=10000,
        plot=False,
        marker=False,
        render=True,
        quiet=False,
    ):
        """
        Moves the specified joint group to a joint target.

        Args:
            group: String specifying the group to move.
            target: List of target joint values for the group.
            tolerance: Threshold within which the error of each joint must be before the method finishes.
            max_steps: maximum number of steps to actuate before breaking
            plot: If True, a .png image of the group joint trajectories will be saved to the local directory.
                  This can be used for PID tuning in case of overshoot etc. The name of the file will be "Joint_angles_" + a number.
            marker: If True, a colored visual marker will be added into the scene to visualize the current
                    cartesian target.
        """

        try:
            assert group in self.groups.keys(), "No group with name {} exists!".format(group)
            if target is not None:
                assert len(target) == len(
                    self.groups[group]
                ), "Mismatching target dimensions for group {}!".format(group)
            ids = self.groups[group]
            steps = 1
            result = ""
            if plot:
                self.plot_list = defaultdict(list)
            self.reached_target = False
            deltas = np.zeros(len(self.sim.data.ctrl))

            if target is not None:
                for i, v in enumerate(ids):
                    self.current_target_joint_values[v] = target[i]
                    # print('Target joint value: {}: {}'.format(v, self.current_target_joint_values[v]))

            for j in range(len(self.sim.data.ctrl)):
                # Update the setpoints of the relevant controllers for the group
                self.actuators[j][4].setpoint = self.current_target_joint_values[j]
                # print('Setpoint {}: {}'.format(j, self.actuators[j][4].setpoint))

            while not self.reached_target:
                current_joint_values = self.sim.data.qpos[self.actuated_joint_ids]

                # self.get_image_data(width=200, height=200, show=True)

                # We still want to actuate all motors towards their targets, otherwise the joints of non-controlled
                # groups will start to drift
                for j in range(len(self.sim.data.ctrl)):
                    self.current_output[j] = self.actuators[j][4](current_joint_values[j])
                    self.sim.data.ctrl[j] = self.current_output[j]
                for i in ids:
                    deltas[i] = abs(self.current_target_joint_values[i] - current_joint_values[i])

                if steps % 1000 == 0 and target is not None and not quiet:
                    print(
                        "Moving group {} to joint target! Max. delta: {}, Joint: {}".format(
                            group, max(deltas), self.actuators[np.argmax(deltas)][3]
                        )
                    )

                if plot and steps % 2 == 0:
                    self.fill_plot_list(group, steps)

                temp = self.sim.data.body_xpos[self.model.body_name2id("ee_link")] - [
                    0,
                    -0.005,
                    0.16,
                ]

                if marker:
                    self.add_marker(self.current_carthesian_target)
                    self.add_marker(temp)

                if max(deltas) < tolerance:
                    if target is not None and not quiet:
                        print(
                            colored(
                                "Joint values for group {} within requested tolerance! ({} steps)".format(
                                    group, steps
                                ),
                                color="green",
                                attrs=["bold"],
                            )
                        )
                    result = "success"
                    self.reached_target = True
                    # break

                if steps > max_steps:
                    if not quiet:
                        print(
                            colored(
                                "Max number of steps reached: {}".format(max_steps),
                                color="red",
                                attrs=["bold"],
                            )
                        )
                        print("Deltas: ", deltas)
                    result = "max. steps reached: {}".format(max_steps)
                    break

                self.sim.step()
                if render:
                    self.viewer.render()
                steps += 1

            self.last_movement_steps = steps

            if plot:
                self.create_joint_angle_plot(group=group, tolerance=tolerance)

            return result

        except Exception as e:
            print(e)
            print("Could not move to requested joint target.")

    def set_group_joint_target(self, group, target):

        idx = self.groups[group]
        try:
            assert len(target) == len(
                idx
            ), "Length of the target must match the number of actuated joints in the group."
            self.current_target_joint_values[idx] = target

        except Exception as e:
            print(e)
            print(f"Could not set new group joint target for group {group}")

    def open_gripper(self, half=False, **kwargs):
        """
        Opens the gripper while keeping the arm in a steady position.
        """
        # print('Open: ', self.sim.data.qpos[self.actuated_joint_ids][self.groups['Gripper']])
        return (
            self.move_group_to_joint_target(
                group="Gripper", target=[0.0], max_steps=1000, tolerance=0.05, **kwargs
            )
            if half
            else self.move_group_to_joint_target(
                group="Gripper", target=[0.4], max_steps=1000, tolerance=0.05, **kwargs
            )
        )

    def close_gripper(self, **kwargs):
        # def close_gripper(self, render=True, max_steps=1000, plot=False, quiet=True):
        """
        Closes the gripper while keeping the arm in a steady position.
        """

        # result = self.move_group_to_joint_target(group='Gripper', target=[-0.4], tolerance=0.05, **kwargs)
        # print('Closed: ', self.sim.data.qpos[self.actuated_joint_ids][self.groups['Gripper']])
        # result = self.move_group_to_joint_target(group='Gripper', target=[0.45, 0.45, 0.55, -0.17], tolerance=0.05, max_steps=max_steps, render=render, marker=True, quiet=quiet, plot=plot)
        return self.move_group_to_joint_target(
            group="Gripper", target=[-0.4], tolerance=0.01, **kwargs
        )

    def grasp(self, **kwargs):
        # def grasp(self, render=True, plot=False):
        """
        Attempts a grasp at the current location and prints some feedback on weather it was successful
        """

        result = self.close_gripper(max_steps=300, **kwargs)

        return result != "success"

    def move_ee(self, ee_position, **kwargs):
        """
        Moves the robot arm so that the gripper center ends up at the requested XYZ-position,
        with a vertical gripper position.

        Args:
            ee_position: List of XYZ-coordinates of the end-effector (ee_link for UR5 setup).
            plot: If True, a .png image of the arm joint trajectories will be saved to the local directory.
                  This can be used for PID tuning in case of overshoot etc. The name of the file will be "Joint_angles_" + a number.
            marker: If True, a colored visual marker will be added into the scene to visualize the current
                    cartesian target.
        """
        joint_angles = self.ik(ee_position)
        if joint_angles is not None:
            result = self.move_group_to_joint_target(group="Arm", target=joint_angles, **kwargs)
            # result = self.move_group_to_joint_target(group='Arm', target=joint_angles, tolerance=0.05, plot=plot, marker=marker, max_steps=max_steps, quiet=quiet, render=render)
        else:
            result = "No valid joint angles received, could not move EE to position."
            self.last_movement_steps = 0
        return result

    def ik(self, ee_position):
        """
        Method for solving simple inverse kinematic problems.
        This was developed for top down graspig, therefore the solution will be one where the gripper is
        vertical. This might need adjustment for other gripper models.

        Args:
            ee_position: List of XYZ-coordinates of the end-effector (ee_link for UR5 setup).

        Returns:
            joint_angles: List of joint angles that will achieve the desired ee position.
        """

        try:
            assert (
                len(ee_position) == 3
            ), "Invalid EE target! Please specify XYZ-coordinates in a list of length 3."
            self.current_carthesian_target = ee_position.copy()
            # We want to be able to spedify the ee position in world coordinates, so subtract the position of the
            # base link. This is because the inverse kinematics solver chain starts at the base link.
            ee_position_base = (
                ee_position - self.sim.data.body_xpos[self.model.body_name2id("base_link")]
            )

            # By adding the appr. distance between ee_link and grasp center, we can now specify a world target position
            # for the grasp center instead of the ee_link
            gripper_center_position = ee_position_base + [0, -0.005, 0.16]
            # gripper_center_position = ee_position_base + [0, 0, 0.185]

            # initial_position=[0, *self.sim.data.qpos[self.actuated_joint_ids][self.groups['Arm']], 0]
            # joint_angles = self.ee_chain.inverse_kinematics(gripper_center_position, [0,0,-1], orientation_mode='X', initial_position=initial_position, regularization_parameter=0.05)
            joint_angles = self.ee_chain.inverse_kinematics(
                gripper_center_position, [0, 0, -1], orientation_mode="X"
            )

            prediction = (
                self.ee_chain.forward_kinematics(joint_angles)[:3, 3]
                + self.sim.data.body_xpos[self.model.body_name2id("base_link")]
                - [0, -0.005, 0.16]
            )
            diff = abs(prediction - ee_position)
            error = np.sqrt(diff.dot(diff))
            joint_angles = joint_angles[1:-2]
            if error <= 0.02:
                return joint_angles

            print("Failed to find IK solution.")
            return None
        except Exception as e:
            print(e)
            print("Could not find an inverse kinematics solution.")

    def ik_2(self, pose_target):
        """
        TODO: Implement orientation.
        """
        target_position = pose_target[:3]
        target_position -= self.sim.data.body_xpos[self.model.body_name2id("base_link")]
        orientation = Quaternion(pose_target[3:])
        target_orientation = orientation.rotation_matrix
        target_matrix = orientation.transformation_matrix
        target_matrix[0][-1] = target_position[0]
        target_matrix[1][-1] = target_position[1]
        target_matrix[2][-1] = target_position[2]
        print(target_matrix)
        self.current_carthesian_target = pose_target[:3]
        joint_angles = self.ee_chain.inverse_kinematics_frame(
            target_matrix, initial_position=initial_position, orientation_mode="all"
        )
        joint_angles = joint_angles[1:-1]
        current_finger_values = self.sim.data.qpos[self.actuated_joint_ids][6:]
        target = [*joint_angles, *current_finger_values]

    def display_current_values(self):
        """
        Debug method, simply displays some relevant data at the time of the call.
        """

        print("\n################################################")
        print("CURRENT JOINT POSITIONS (ACTUATED)")
        print("################################################")
        for i in range(len(self.actuated_joint_ids)):
            print(
                "Current angle for joint {}: {}".format(
                    self.actuators[i][3], self.sim.data.qpos[self.actuated_joint_ids][i]
                )
            )

        print("\n################################################")
        print("CURRENT JOINT POSITIONS (ALL)")
        print("################################################")
        for i in range(len(self.model.jnt_qposadr)):
            # for i in range(self.model.njnt):
            name = self.model.joint_id2name(i)
            print("Current angle for joint {}: {}".format(name, self.sim.data.get_joint_qpos(name)))
            # print('Current angle for joint {}: {}'.format(self.model.joint_id2name(i), self.sim.data.qpos[i]))

        print("\n################################################")
        print("CURRENT BODY POSITIONS")
        print("################################################")
        for i in range(self.model.nbody):
            print(
                "Current position for body {}: {}".format(
                    self.model.body_id2name(i), self.sim.data.body_xpos[i]
                )
            )

        print("\n################################################")
        print("CURRENT BODY ROTATION MATRIZES")
        print("################################################")
        for i in range(self.model.nbody):
            print(
                "Current rotation for body {}: {}".format(
                    self.model.body_id2name(i), self.sim.data.body_xmat[i]
                )
            )

        print("\n################################################")
        print("CURRENT BODY ROTATION QUATERNIONS (w,x,y,z)")
        print("################################################")
        for i in range(self.model.nbody):
            print(
                "Current rotation for body {}: {}".format(
                    self.model.body_id2name(i), self.sim.data.body_xquat[i]
                )
            )

        print("\n################################################")
        print("CURRENT ACTUATOR CONTROLS")
        print("################################################")
        for i in range(len(self.sim.data.ctrl)):
            print(
                "Current activation of actuator {}: {}".format(
                    self.actuators[i][1], self.sim.data.ctrl[i]
                )
            )

    def toss_it_from_the_ellbow(self):
        """
        Test method for trying out tossing of grasped objects.
        """
        for t in range(300):
            self.sim.data.ctrl[2] = -2.0
            self.sim.data.ctrl[0] = -2.0
            self.sim.step()
            self.viewer.render()

            if t > 200:
                self.sim.data.ctrl[6] = 1.0
                self.sim.data.ctrl[3] = -1.0

        self.sim.data.ctrl[:] = 0
        self.move_group_to_joint_target()

    def stay(self, duration, render=True):
        """
        Holds the current position by actuating the joints towards their current target position.

        Args:
            duration: Time in ms to hold the position.
        """

        # print('Holding position!')
        starting_time = time.time()
        elapsed = 0
        while elapsed < duration:
            self.move_group_to_joint_target(
                max_steps=10, tolerance=0.0000001, plot=False, quiet=True, render=render
            )
            elapsed = (time.time() - starting_time) * 1000
        # print('Moving on...')

    def fill_plot_list(self, group, step):
        """
        Creates a two dimensional list of joint angles for plotting.

        Args:
            group: The group involved in the movement.
            step: The step of the trajectory the values correspond to.
        """

        for i in self.groups[group]:
            self.plot_list[self.actuators[i][3]].append(
                self.sim.data.qpos[self.actuated_joint_ids][i]
            )
        self.plot_list["Steps"].append(step)

    def create_joint_angle_plot(self, group, tolerance):
        """
        Saves the recorded joint values as a .png-file. The values for each joint of the group are
        put in a seperate subplot.

        Args:
            group: The group the stored values belong to.
            tolerance: The tolerance value that the joints were required to be in.
        """

        self.image_counter += 1
        keys = list(self.plot_list.keys())
        number_subplots = len(self.plot_list) - 1
        columns = 3
        rows = (number_subplots // columns) + (number_subplots % columns)

        position = range(1, number_subplots + 1)
        fig = plt.figure(1, figsize=(15, 10))
        plt.subplots_adjust(hspace=0.4, left=0.05, right=0.95, top=0.95, bottom=0.05)

        for i in range(number_subplots):
            axis = fig.add_subplot(rows, columns, position[i])
            axis.plot(self.plot_list["Steps"], self.plot_list[keys[i]])
            axis.set_title(keys[i])
            axis.set_xlabel(keys[-1])
            axis.set_ylabel("Joint angle [rad]")
            axis.xaxis.set_label_coords(0.05, -0.1)
            axis.yaxis.set_label_coords(1.05, 0.5)
            axis.axhline(
                self.current_target_joint_values[self.groups[group][i]], color="g", linestyle="--"
            )
            axis.axhline(
                self.current_target_joint_values[self.groups[group][i]] + tolerance,
                color="r",
                linestyle="--",
            )
            axis.axhline(
                self.current_target_joint_values[self.groups[group][i]] - tolerance,
                color="r",
                linestyle="--",
            )

        filename = "Joint_values_{}.png".format(self.image_counter)
        plt.savefig(filename)
        print(
            colored(
                "Saved trajectory to {}.".format(filename),
                color="yellow",
                on_color="on_grey",
                attrs=["bold"],
            )
        )
        plt.clf()

    def get_image_data(self, show=False, camera="top_down", width=200, height=200):
        """
        Returns the RGB and depth images of the provided camera.

        Args:
            show: If True displays the images for five seconds or until a key is pressed.
            camera: String specifying the name of the camera to use.
        """

        rgb, depth = copy.deepcopy(
            self.sim.render(width=width, height=height, camera_name=camera, depth=True)
        )
        if show:
            cv.imshow("rbg", cv.cvtColor(rgb, cv.COLOR_BGR2RGB))
            # cv.imshow('depth', depth)
            cv.waitKey(1)
            # cv.waitKey(delay=5000)
            # cv.destroyAllWindows()

        return np.array(np.fliplr(np.flipud(rgb))), np.array(np.fliplr(np.flipud(depth)))

    def depth_2_meters(self, depth):
        """
        Converts the depth array delivered by MuJoCo (values between 0 and 1) into actual m values.

        Args:
            depth: The depth array to be converted.
        """

        extend = self.model.stat.extent
        near = self.model.vis.map.znear * extend
        far = self.model.vis.map.zfar * extend
        return near / (1 - depth * (1 - near / far))

    def create_camera_data(self, width, height, camera):
        """
        Initializes all camera parameters that only need to be calculated once.
        """

        cam_id = self.model.camera_name2id(camera)
        # Get field of view
        fovy = self.model.cam_fovy[cam_id]
        # Calculate focal length
        f = 0.5 * height / np.tan(fovy * np.pi / 360)
        # Construct camera matrix
        self.cam_matrix = np.array(((f, 0, width / 2), (0, f, height / 2), (0, 0, 1)))
        # Rotation of camera in world coordinates
        self.cam_rot_mat = self.model.cam_mat0[cam_id]
        self.cam_rot_mat = np.reshape(self.cam_rot_mat, (3, 3))
        # Position of camera in world coordinates
        self.cam_pos = self.model.cam_pos0[cam_id]
        self.cam_init = True

    def world_2_pixel(self, world_coordinate, width=200, height=200, camera="top_down"):
        """
        Takes a XYZ world position and transforms it into pixel coordinates.
        Mainly implemented for testing the correctness of the camera matrix, focal length etc.

        Args:
            world_coordinate: XYZ world coordinate to be transformed into pixel space.
            width: Width of the image (pixel).
            height: Height of the image (pixel).
            camera: Name of camera used to obtain the image.
        """

        if not self.cam_init:
            self.create_camera_data(width, height, camera)

        # Homogeneous image point
        hom_pixel = self.cam_matrix @ self.cam_rot_mat @ (world_coordinate - self.cam_pos)
        # Real image point
        pixel = hom_pixel[:2] / hom_pixel[2]

        return np.round(pixel[0]).astype(int), np.round(pixel[1]).astype(int)

    def pixel_2_world(self, pixel_x, pixel_y, depth, width=200, height=200, camera="top_down"):
        """
        Converts pixel coordinates into world coordinates.

        Args:
            pixel_x: X-coordinate in pixel space.
            pixel_y: Y-coordinate in pixel space.
            depth: Depth value corresponding to the pixel.
            width: Width of the image (pixel).
            height: Height of the image (pixel).
            camera: Name of camera used to obtain the image.
        """

        if not self.cam_init:
            self.create_camera_data(width, height, camera)

        # Create coordinate vector
        pixel_coord = np.array([pixel_x, pixel_y, 1]) * (-depth)
        # Get position relative to camera
        pos_c = np.linalg.inv(self.cam_matrix) @ pixel_coord
        # Get world position
        pos_w = np.linalg.inv(self.cam_rot_mat) @ (pos_c + self.cam_pos)

        return pos_w

    def add_marker(self, coordinates, label=True, size=None, color=None):
        """
        Adds a circular red marker at the coordinates, dislaying the coordinates as a label.

        Args:
            coordinates: List of XYZ-coordinates in m.
            label: If True, displays the target coordinates next to the marker
            size: List of floats specifying the radius in each direction
            color: List of floats between 0 and 1 specifying the RGB color parts
        """

        if size is None:
            size = [0.015, 0.015, 0.015]
        if color is None:
            color = [1, 0, 0]
        label_str = str(coordinates) if label else ""
        rgba = np.concatenate((color, np.ones(1)))
        self.viewer.add_marker(pos=coordinates, label=label_str, size=size, rgba=rgba, type=2)

    @property
    def last_steps(self):
        return self.last_movement_steps
