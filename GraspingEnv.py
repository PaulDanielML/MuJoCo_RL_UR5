#!/usr/bin/env python3

import mujoco_py
from gym.envs.mujoco import mujoco_env
from MujocoController import MJ_Controller

class GraspEnv(mujoco_env.MujocoEnv):
	def __init__(self):
		print('Hallo')



if __name__ == '__main__':
	env = GraspEnv()
