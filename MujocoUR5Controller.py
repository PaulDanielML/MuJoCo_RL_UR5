#!/usr/bin/env python3

# Author: Paul Daniel (pdd@mp.aau.dk)

import mujoco_py as mp
import time
import numpy as np
from simple_pid import PID
from termcolor import colored
import ikpy
from pyquaternion import Quaternion


class Mujoco_UR5_Controller(object):
    def __init__(self):
        self.model = mp.load_model_from_path('UR5+gripper/UR5gripper_v2.xml')
        self.sim = mp.MjSim(self.model)
        self.viewer = mp.MjViewer(self.sim)
        self.create_lists()
        self.actuated_joint_ids = [i[2] for i in self.actuators]
        self.reached_target = False
        self.counter = 0
        self.ee_chain = ikpy.chain.Chain.from_urdf_file('UR5+gripper/ur5_gripper.urdf')
        self.current_carthesian_target = [0, 0, 0]
        self.set_new_joint_target([0.8, -1.57, 1.57, -0.8, 0.5, 1, 0.1, 0.1, 0.1, 0.1])
        self.move_to_joint_target()
        self.open_gripper()


    def show_model_info(self):
        print('\nNumber of bodies: {}'.format(self.model.nbody))
        for i in range(self.model.nbody):
            print('Body ID: {}, Body Name: {}'.format(i, self.model.body_id2name(i)))

        print('\nNumber of joints: {}'.format(self.model.njnt))
        for i in range(self.model.njnt):
            print('Joint ID: {}, Joint Name: {}'.format(i, self.model.joint_id2name(i)))

        print('\nNumber of Actuators: {}'.format(len(self.sim.data.ctrl)))
        for i in range(len(self.sim.data.ctrl)):
            print('Actuator ID: {}, Actuator Name: {}, Control Range: {}'.format(i, self.model.actuator_id2name(i), self.model.actuator_ctrlrange[i]))

    def create_lists(self):
        """Create some basic lists:

        - controller_list: Contains a controller for each of the actuated joints. This is done so that different gains may be 
        specified for each controller.

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
        self.controller_list.append(PID(10, 0.3, 0.05, setpoint=0.1, output_limits=(-2, 2))) # Shoulder Pan Joint
        self.controller_list.append(PID(5, 0.1, 0.05, setpoint=0.1, output_limits=(-2, 2))) # Shoulder Lift Joint
        self.controller_list.append(PID(5, 2.0, 0.05, setpoint=0.1, output_limits=(-2, 2))) # Elbow Joint
        self.controller_list.append(PID(3, 0.5, 0.05, setpoint=0.1, output_limits=(-1, 1))) # Wrist 1 Joint
        self.controller_list.append(PID(5, 1.0, 0.5, setpoint=0.1, output_limits=(-1, 1))) # Wrist 2 Joint
        self.controller_list.append(PID(2, 0.3, 0.05, setpoint=0.1, output_limits=(-1, 1))) # Wrist 3 Joint
        self.controller_list.append(PID(1, 0.1, 0.05, setpoint=0.1, output_limits=(-0.1, 0.8))) # Finger 1 Joint 1
        self.controller_list.append(PID(1, 0.1, 0.05, setpoint=0.1, output_limits=(-0.1, 0.8))) # Finger 2 Joint 1
        self.controller_list.append(PID(1, 0.1, 0.05, setpoint=0.1, output_limits=(-0.1, 0.8))) # Middle Finger Joint 1
        self.controller_list.append(PID(1, 0.1, 0.05, setpoint=0.1, output_limits=(-0.8, 0.8))) # Gripperpalm Finger 1 Joint

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


    def move_to_joint_target(self, tolerance = 0.1):
        """Moves the joints to the specified target values.
        Args:
            - targets {list}: For the UR5 setup, this is a list of values of the 10 actuated joints
        """
        reached_target = False
        deltas = np.zeros(len(self.current_target_joint_values))
        while not self.reached_target:
        # while True:
            current_joint_values = self.sim.data.qpos[self.actuated_joint_ids]
            current_arm_joint_values = current_joint_values[:6]
            current_arm_joint_values_2 = [0.0, *current_arm_joint_values, 0.0]
            # print(self.sim.data.qpos[self.actuated_joint_ids])

            for i in range(len(current_joint_values)):
                self.actuators[i][4].setpoint = self.current_target_joint_values[i]
                self.current_output[i] = self.actuators[i][4](current_joint_values[i])
                self.sim.data.ctrl[i] = self.current_output[i]
                deltas[i] = abs(self.current_target_joint_values[i] - current_joint_values[i])

            if self.counter%1000==0 and self.counter > 0:
                print('Max. delta: {}, Joint: {}'.format(max(deltas), self.actuators[np.argmax(deltas)][3]))

                if np.argmax(deltas) > 5 and np.max(deltas) > 0.2: 
                    print(colored('Grasped something!', color='red', attrs=['bold', 'blink']))
                    self.current_target_joint_values = self.sim.data.qpos[self.actuated_joint_ids]
                    # self.sim.data.ctrl[6] = 0.1
                    # self.sim.data.ctrl[7] = 0.1
                    self.sim.data.ctrl[8] = 0.3
                    self.reached_target = True

                # print('Current angle for joint {}: {}'.format(self.actuators[np.argmax(deltas)][3], self.sim.data.qpos[self.actuated_joint_ids][np.argmax(deltas)]))
                # self.detect_collisions()
                # print('Current target angle for joint {}: {}'.format(self.actuators[np.argmax(deltas)][3], self.current_target_joint_values[np.argmax(deltas)]))
                # print('Current end effector position {}: {}'.format(self.ee_chain.forward_kinematics(current_arm_joint_values_2)))
                # print(self.sim.data.ctrl[np.argmax(deltas)])
                # print(self.model.body_id2name(self.sim.data.contact[0].geom1))
                # print(self.model.body_id2name(self.sim.data.contact[0].geom2))
            if max(deltas) < tolerance and not self.reached_target:
                print(colored('Joint values within requested tolerance! ({} steps)'.format(self.counter), color='green', attrs=['bold']))
                self.reached_target = True
                # break
            self.sim.step()
            # self.viewer.add_marker(pos=self.current_carthesian_target, label='Target', size=np.ones(3) * 0.05)
            self.viewer.render()
            self.counter += 1
       

    def set_new_joint_target(self, targets):
        try:
            assert (len(targets) == len(self.sim.data.ctrl)), "Length of the specified target does not match number of controllable joints."
            # print(colored('Received new joint space target: {}'.format(targets), color='red', attrs=['bold']))
            self.counter = 0
            self.reached_target = False
            self.current_target_joint_values = targets
        except Exception as e:
            print(e)
            print('Could not set new joint value target.')

    def update_control_outputs(self):
        current_joint_values = self.sim.data.qpos[self.actuated_joint_ids]
        for i in range(len(current_joint_values)):
            self.actuators[i][4].setpoint = self.current_target_joint_values[i]
            self.current_output[i] = self.actuators[i][4](current_joint_values[i])


    def actuate_joints(self):
        self.sim.data.ctrl[:] = self.current_output[:]

    def step_and_render(self):
        self.sim.step()
        self.viewer.render()

    def open_gripper(self):
        target_vector = self.sim.data.qpos[self.actuated_joint_ids]
        target_vector[6] = 0
        target_vector[7] = 0
        target_vector[8] = 0
        target_vector[9] = -0.15
        self.set_new_joint_target(target_vector)
        self.move_to_joint_target()
        # self.sim.data.ctrl[6:9] = -1


    def open_gripper_motors(self):
        for t in range(1500):
            self.sim.data.ctrl[6:9] = -1
            self.step_and_render()


    def close_gripper(self):
        target_vector = self.sim.data.qpos[self.actuated_joint_ids]
        target_vector[6] = 0.6
        target_vector[7] = 0.6
        target_vector[8] = 0.6
        target_vector[9] = -0.2
        self.set_new_joint_target(target_vector)
        self.move_to_joint_target()

    def close_gripper_motors(self):
        for t in range(200):
            self.sim.data.ctrl[6] = 1
            self.sim.data.ctrl[7] = 1
            self.sim.data.ctrl[8] = 1
            self.step_and_render()

    def hold_position(self, duration=1000):
        for t in range(duration*500):
            self.update_control_outputs()
            self.actuate_joints()
            self.step_and_render()
            time.sleep(0.002)

    def move_ee_to_position(self, x, y, z):
        for i in range(1000):
            J = self.sim.data.get_body_jacp('ee_link').reshape((3,self.model.njnt))
            vel = J.T @ np.linalg.inv(J@J.T + 1e-2*np.eye(3)) @ ([-x,-y,-z])
            act_vel = vel[self.actuated_joint_ids]
            self.set_new_joint_target(act_vel)
            self.update_control_outputs()
            self.actuate_joints()
            self.viewer.add_marker(pos=[x,y,z], label='Target', size=np.ones(3) * 0.05)
            self.step_and_render()
        for i in range(self.model.nbody):
            print(self.model.body_id2name(i))
            print(self.sim.data.body_xpos[i])
        for i in range(len(self.actuated_joint_ids)):
            print(self.actuators[i][3])
            print(self.current_target_joint_values[i])
            print(self.sim.data.qpos[self.actuated_joint_ids][i])


    def ik(self, pose_target):
        self.current_carthesian_target = pose_target[:3]
        target_position = pose_target[:3]
        target_position -= self.sim.data.body_xpos[self.model.body_name2id('base_link')]
        joint_angles = self.ee_chain.inverse_kinematics(target_position, [0,0,-1], orientation_mode='X')
        joint_angles = joint_angles[1:-1]
        current_finger_values = self.sim.data.qpos[self.actuated_joint_ids][6:]
        target = [*joint_angles, *current_finger_values]

        self.set_new_joint_target(target)
        self.move_to_joint_target(tolerance=0.05)


    def ik_2(self, pose_target):
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
        # target_matrix =   [[1, 0, 0, pose_target[0]],
                        # [0, 1, 0, pose_target[1]],
                        # [0, 0, 1, pose_target[2]-0.87],
                        # [0, 0, 0, 1]]
        initial_position=[0, *self.sim.data.qpos[:6], 0]
        # print(initial_position)
       
        # joint_angles = self.ee_chain.inverse_kinematics(target_position=target_position)

        joint_angles = self.ee_chain.inverse_kinematics_frame(target_matrix, initial_position=initial_position, orientation_mode='all')
        joint_angles = joint_angles[1:-1]
        current_finger_values = self.sim.data.qpos[self.actuated_joint_ids][6:]
        target = [*joint_angles, *current_finger_values]


    def display_current_values(self):
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
        print('JOINTS IN KINEMATIC CHAIN')
        print('################################################') 
        for i in range(len(self.ee_chain)):
            print(self.ee_chain.links[i])

        print('\n################################################')
        print('CURRENT ACTUATOR CONTROLS')
        print('################################################') 
        for i in range(len(self.ee_chain)):
            print('Current activation of actuator {}: {}'.format(self.actuators[i][1], self.sim.data.ctrl[i]))


    def detect_collisions(self):
        all_contacts = self.sim.data.contact
        collisions = [cont for cont in all_contacts if cont.dist < 0.0]
        # for coll in collisions:
            # print(self.model.body_id2name(coll.geom1))
            # print(self.model.body_id2name(coll.geom2))
        # print('\n', self.sim.data.ncon)
        # print(all_contacts[0].includemargin)


    def toss_it_from_the_ellbow(self):
        for t in range(300):
            self.sim.data.ctrl[2] = -2.0
            self.sim.data.ctrl[0] = -2.0
            self.step_and_render()

            if t > 200:
                self.sim.data.ctrl[6] = -1.0
                self.sim.data.ctrl[7] = -1.0
                self.sim.data.ctrl[8] = -1.0
                self.sim.data.ctrl[3] = -1.0
            # time.sleep(0.01)

        self.reached_target = False
        self.move_to_joint_target()

 
if __name__ == '__main__':

    UR5_controller = Mujoco_UR5_Controller()
    UR5_controller.close_gripper()
    UR5_controller.open_gripper()
    # UR5_controller.toss_it_from_the_ellbow()

    UR5_controller.ik([0.03, -0.6, 1.2, 0.707, 0, 0.707, 0])
    UR5_controller.ik([0.03, -0.6, 1.09, 0.707, 0, 0.707, 0])


    UR5_controller.close_gripper()

    UR5_controller.ik([0.03, -0.6, 1.3, 0.707, 0, 0.707, 0])
    UR5_controller.hold_position(1)

    UR5_controller.toss_it_from_the_ellbow()
    UR5_controller.hold_position()

# 