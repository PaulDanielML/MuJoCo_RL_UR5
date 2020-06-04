#!/usr/bin/env python3

# Author: Paul Daniel (pdd@mp.aau.dk)

from collections import defaultdict
import mujoco_py as mp
import time
import numpy as np
from simple_pid import PID
from termcolor import colored
import ikpy
from pyquaternion import Quaternion
import cv2 as cv
import matplotlib.pyplot as plt
import traceback


class MJ_Controller(object):
    def __init__(self):
        self.model = mp.load_model_from_path('UR5+gripper/UR5gripper_v2.xml')
        self.sim = mp.MjSim(self.model)
        self.viewer = mp.MjViewer(self.sim)
        self.create_lists()
        self.groups = defaultdict(list)
        self.groups['All'] = [i for i in range(len(self.sim.data.ctrl))]
        self.create_group('Arm', [i for i in range(6)])
        self.create_group('Gripper', [i for i in range(6,10)])
        self.actuated_joint_ids = [i[2] for i in self.actuators]
        self.reached_target = False
        self.current_output = np.zeros(len(self.sim.data.ctrl))
        self.image_counter = 0
        self.ee_chain = ikpy.chain.Chain.from_urdf_file('UR5+gripper/ur5_gripper.urdf')
        self.move_group_to_joint_target(plot=True, group='Arm')



        # rgb, depth = self.sim.render(width=800, height=800, camera_name='top_down', depth=True)
        # cv.imshow('rbg', cv.cvtColor(rgb, cv.COLOR_BGR2RGB))
        # cv.imshow('depth', depth)
        # cv.waitKey(delay=10000)
        # cv.destroyAllWindows()

    def create_group(self, group_name, idx_list):
        """
        Allows the user to create custom objects for controlling groups of joints.
        The method show_model_info can be used to get lists of joints and actuators.

        Args:
            group_name: String defining the désired name of the group.
            idx_list: List containing the IDs of the actuators that will belong to this group.
        """

        try:
            assert len(idx_list) <= len(self.sim.data.ctrl), 'Too many joints specified!'
            assert group_name not in self.groups.keys(), 'A group with name {} already exists!'.format(group_name)
            assert np.max(idx_list) <= len(self.sim.data.ctrl), 'List contains invalid actuator ID (too high)'

            self.groups[group_name] = idx_list
            print('Created new control group \'{}\'.'.format(group_name))

        except Exception as e:
            print(e)
            print('Could not create a new group.')

    def show_model_info(self):
        """
        Displays relevant model info for the user, namely bodies, joints, actuators, as well as their IDs and ranges.
        Also gives info on which actuators control which joints and which joints are included in the kinematic chain. 
        """
        print('\nNumber of bodies: {}'.format(self.model.nbody))
        for i in range(self.model.nbody):
            print('Body ID: {}, Body Name: {}'.format(i, self.model.body_id2name(i)))

        print('\nNumber of joints: {}'.format(self.model.njnt))
        for i in range(self.model.njnt):
            print('Joint ID: {}, Joint Name: {}, Limits: {}'.format(i, self.model.joint_id2name(i), self.model.jnt_range[i]))

        print('\nNumber of Actuators: {}'.format(len(self.sim.data.ctrl)))
        for i in range(len(self.sim.data.ctrl)):
            print('Actuator ID: {}, Actuator Name: {}, Controlled Joint: {}, Control Range: {}'.format(i, self.model.actuator_id2name(i), self.actuators[i][3], self.model.actuator_ctrlrange[i]))

        print('\nJoints in kinematic chain: {}'.format([i.name for i in self.ee_chain.links]))

    def create_lists(self):
        """
        Creates some basic lists and fill them with initial values. This function is called in the class costructor.
        The following lists/dictionaries are created:

        - controller_list: Contains a controller for each of the actuated joints. This is done so that different gains may be 
        specified for each controller.

        - current_joint_value_targets: Same as the current setpoints for all controllers, created for convenience

        - current_output = A list containing the ouput values of all the controllers. This list is only initiated here, its 
        values are overwritten at the first simulation step

        - actuators: 2D list, each entry represents one actuator and contains:
            0 actuator ID 
            1 actuator name 
            2 joint ID of the joint controlled by this actuator 
            3 joint name
            4 controller for controlling the actuator
        """

        self.controller_list = []
        self.controller_list.append(PID(10, 0.3, 0.05, setpoint=0.8, output_limits=(-2, 2))) # Shoulder Pan Joint
        self.controller_list.append(PID(5, 0.1, 0.05, setpoint=-1.57, output_limits=(-2, 2))) # Shoulder Lift Joint
        self.controller_list.append(PID(5, 2.0, 0.05, setpoint=1.57, output_limits=(-2, 2))) # Elbow Joint
        self.controller_list.append(PID(3, 0.5, 0.05, setpoint=-0.8, output_limits=(-1, 1))) # Wrist 1 Joint
        self.controller_list.append(PID(5, 1.0, 0.5, setpoint=0.5, output_limits=(-1, 1))) # Wrist 2 Joint
        self.controller_list.append(PID(2, 0.3, 0.05, setpoint=1.0, output_limits=(-1, 1))) # Wrist 3 Joint
        self.controller_list.append(PID(2, 0.1, 0.05, setpoint=0.2, output_limits=(-0.1, 0.8))) # Finger 1 Joint 1
        self.controller_list.append(PID(2, 0.1, 0.05, setpoint=0.2, output_limits=(-0.1, 0.8))) # Finger 2 Joint 1
        self.controller_list.append(PID(1, 0.1, 0.05, setpoint=0.0, output_limits=(-0.1, 0.8))) # Middle Finger Joint 1
        self.controller_list.append(PID(1, 0.1, 0.05, setpoint=0.0, output_limits=(-0.8, 0.8))) # Gripperpalm Finger 1 Joint

        self.current_target_joint_values = []
        for i in range(len(self.sim.data.ctrl)):
            self.current_target_joint_values.append(self.controller_list[i].setpoint)

        self.current_output = []
        for i in range(len(self.controller_list)):
            self.current_output.append(self.controller_list[i](0))


        self.actuators = []
        for i in range(len(self.sim.data.ctrl)):
            item = []
            item.append(i)
            item.append(self.model.actuator_id2name(i))
            item.append(self.model.actuator_trnid[i][0])
            item.append(self.model.joint_id2name(self.model.actuator_trnid[i][0]))
            item.append(self.controller_list[i])
            self.actuators.append(item)


    def actuate_joint_group(self, group, motor_values):
        try:
            assert group in self.groups.keys(), 'No group with name {} exists!'.format(group)
            assert len(motor_values) == len(self.groups[group]), 'Invalid number of actuator values!'
            for i,v in enumerate(self.groups[group]):
                self.sim.data.ctrl[v] = motor_values[i]
            print(self.sim.data.ctrl)

        except Exception as e:
            print(e)
            print('Could not actuate requested joint group.')

    def move_group_to_joint_target(self, group='All', target=None, tolerance=0.1, max_steps=10000, plot=False):
        """
        Moves the specified joint group to a joint target.

        Args:
            group: String specifying the group to move.
            target: List of target joint values for the group.
            tolerance: Threshold within which the error of each joint must be before the method finishes.
            max_steps: maximum number of steps to actuate before breaking
            plot: If True, a .png image of the group joint trajectories will be saved to the local directory.
                  This can be used for PID tuning in case of overshoot etc. The name of the file will be "Joint_angles_" + a number.
        """
        
        try:
            assert group in self.groups.keys(), 'No group with name {} exists!'.format(group)
            if target is not None:
                assert len(target) == len(self.groups[group]), 'Mismatching target dimensions for group {}!'.format(group)
            ids = self.groups[group]
            steps = 1
            self.plot_list = defaultdict(list)
            self.reached_target = False
            deltas = np.zeros(len(self.sim.data.ctrl))

            if target is not None:
                for i,v in enumerate(ids):
                    self.current_target_joint_values[v] = target[i]
                    # Update the setpoints of the relevant controllers for the group
                    self.actuators[v][4].setpoint = self.current_target_joint_values[v]
            while not self.reached_target:
                current_joint_values = self.sim.data.qpos[self.actuated_joint_ids]
                
                # We still want to actuate all motors towards their targets, otherwise the joints of non-controlled
                # groups will start to drift     
                for j in range(len(self.sim.data.ctrl)):
                    self.current_output[j] = self.actuators[j][4](current_joint_values[j])
                    self.sim.data.ctrl[j] = self.current_output[j]
                for i in ids:
                    deltas[i] = abs(self.current_target_joint_values[i] - current_joint_values[i])

                if steps%1000==0 and target is not None:
                    print('Moving group {} to joint target! Max. delta: {}, Joint: {}'.format(group, max(deltas), self.actuators[np.argmax(deltas)][3]))

                if plot and steps%20==0:
                    self.fill_plot_list(group, steps)

                if max(deltas) < tolerance:
                    if target is not None:
                        print(colored('Joint values for group {} within requested tolerance! ({} steps)'.format(group, steps), color='green', attrs=['bold']))
                    self.reached_target = True

                if steps > max_steps:
                    break

                self.sim.step()
                self.viewer.render()
                steps += 1

            if plot:
                self.create_joint_angle_plot(group)

        except Exception as e:
            print(e)
            print(traceback.format_exc())
            print('Could not move to requested joint target.')
       

    def open_gripper(self):
        """
        Opens the gripper while keeping the arm in a steady position.
        """

        print('Opening gripper...')
        self.move_group_to_joint_target(group='Gripper', target=[0.2, 0.2, 0.0, 0.0])


    def close_gripper(self):
        """
        Closes the gripper while keeping the arm in a steady position.
        """

        print('Closing gripper...')
        self.move_group_to_joint_target(group='Gripper', target=[0.45, 0.45, 0.55, -0.17], tolerance=0.05, max_steps=1000)
        print('Gripper joint positions:')
        print(self.sim.data.qpos[self.actuated_joint_ids][self.groups['Gripper']])


    def grasp(self):
        """
        Attempts a grasp at the current location and prints some feedback on weather it was successful 
        """
        self.close_gripper()
        if not self.reached_target:
            print(colored('Grasped something!', color='green', attrs=['bold', 'blink']))
        else:
            print(colored('Could not grasp anything!', color='red', attrs=['bold', 'blink']))


    def move_ee(self, ee_position, plot=False):
        """
        Moves the robot arm so that the end effector ends up at the requested XYZ-position,
        with a vertical gripper position.

        Args:
            ee_position: List of XYZ-coordinates of the end-effector (ee_link for UR5 setup).
            plot: If True, a .png image of the arm joint trajectories will be saved to the local directory.
                  This can be used for PID tuning in case of overshoot etc. The name of the file will be "Joint_angles_" + a number.
        """
        joint_angles = self.ik(ee_position)
        if joint_angles is not None:
            self.move_group_to_joint_target(group='Arm', target=joint_angles, tolerance=0.05, plot=plot)
        else:
            print('No valid joint angles received, could not move EE to position.')


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
            assert len(ee_position) == 3, 'Invalid EE target! Please specify XYZ-coordinates in a list of length 3.'
            self.current_carthesian_target = ee_position
            ee_position -= self.sim.data.body_xpos[self.model.body_name2id('base_link')]
            joint_angles = self.ee_chain.inverse_kinematics(ee_position, [0,0,-1], orientation_mode='X')
            joint_angles = joint_angles[1:-1]

            return joint_angles

        except Exception as e:
            print(e)
            print('Could not find an inverse kinematics solution.')

    def ik_2(self, pose_target):
        """
        TODO: Implement orientation.
        """
        target_position = pose_target[:3]
        target_position -= self.sim.data.body_xpos[self.model.body_name2id('base_link')]
        orientation = Quaternion(pose_target[3:])
        target_orientation = orientation.rotation_matrix
        target_matrix = orientation.transformation_matrix
        target_matrix[0][-1] = target_position[0]
        target_matrix[1][-1] = target_position[1]
        target_matrix[2][-1] = target_position[2]
        print(target_matrix)
        self.current_carthesian_target = pose_target[:3]
        initial_position=[0, *self.sim.data.qpos[:6], 0]
        joint_angles = self.ee_chain.inverse_kinematics_frame(target_matrix, initial_position=initial_position, orientation_mode='all')
        joint_angles = joint_angles[1:-1]
        current_finger_values = self.sim.data.qpos[self.actuated_joint_ids][6:]
        target = [*joint_angles, *current_finger_values]


    def display_current_values(self):
        """
        Debug method, simply displays some relevant data at the time of the call.
        """

        print('\n################################################')
        print('CURRENT JOINT POSITIONS')
        print('################################################')
        for i in range(len(self.actuated_joint_ids)):
            print('Current angle for joint {}: {}'.format(self.actuators[i][3], self.sim.data.qpos[self.actuated_joint_ids][i]))

        print('\n################################################')
        print('CURRENT BODY ROTATION MATRIZES')
        print('################################################')
        for i in range(self.model.nbody):
            print('Current rotation for body {}: {}'.format(self.model.body_id2name(i), self.sim.data.body_xmat[i]))

        print('\n################################################')
        print('CURRENT BODY ROTATION QUATERNIONS (w,x,y,z)')
        print('################################################')
        for i in range(self.model.nbody):
            print('Current rotation for body {}: {}'.format(self.model.body_id2name(i), self.sim.data.body_xquat[i]))

        print('\n################################################')
        print('CURRENT ACTUATOR CONTROLS')
        print('################################################') 
        for i in range(len(self.ee_chain)):
            print('Current activation of actuator {}: {}'.format(self.actuators[i][1], self.sim.data.ctrl[i]))


    def detect_collisions(self):
        """
        TODO
        """
        all_contacts = self.sim.data.contact
        collisions = [cont for cont in all_contacts if cont.dist < 0.0]
        # for coll in collisions:
            # print(self.model.body_id2name(coll.geom1))
            # print(self.model.body_id2name(coll.geom2))
        # print('\n', self.sim.data.ncon)
        # print(all_contacts[0].includemargin)


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
                self.sim.data.ctrl[6] = -1.0
                self.sim.data.ctrl[7] = -1.0
                self.sim.data.ctrl[8] = -1.0
                self.sim.data.ctrl[3] = -1.0

        self.sim.data.ctrl[:] = 0
        self.move_group_to_joint_target()




    def stay(self, duration):
        """
        Holds the current position by actuating the joints towards their current target position.

        Args:
            duration: time in ms to hold the position.
        """

        print('Holding position!')
        t = 0
        while t < duration:
            if t%100 == 0:
                self.move_group_to_joint_target(max_steps=1, plot=False)
            t += 1
            time.sleep(0.001)
        print('Moving on...')


    def fill_plot_list(self, group, step):
        for i in self.groups[group]:
            self.plot_list[self.actuators[i][3]].append(self.sim.data.qpos[self.actuated_joint_ids][i])
        self.plot_list['Steps'].append(step)


    def create_joint_angle_plot(self, group):
        self.image_counter += 1
        keys = list(self.plot_list.keys())
        number_subplots = len(self.plot_list) - 1
        columns = 3
        rows = (number_subplots // columns) + (number_subplots % columns)

        position = range(1, number_subplots+1)
        fig = plt.figure(1, figsize=(15,10))
        plt.subplots_adjust(hspace=0.4, left=0.05, right=0.95, top=0.95, bottom=0.05)

        for i in range(number_subplots):
            axis = fig.add_subplot(rows, columns, position[i])
            axis.plot(self.plot_list['Steps'], self.plot_list[keys[i]])
            axis.set_title(keys[i])
            axis.set_xlabel(keys[-1])
            axis.set_ylabel('Joint angle [rad]')
            axis.xaxis.set_label_coords(0.05, -0.15)
            axis.yaxis.set_label_coords(1.05, 0.5)
            axis.axhline(self.current_target_joint_values[self.groups[group][i]], color='g', linestyle='--')

        filename = 'Joint_values_{}.png'.format(self.image_counter)
        plt.savefig(filename)
        print(colored('Saved trajectory to {}.'.format(filename), color='yellow', on_color='on_grey', attrs=['bold']))
        plt.clf()

    
    def utils(self):
        self.viewer.add_marker(pos=self.current_carthesian_target, label='Target', size=np.ones(3) * 0.05)
