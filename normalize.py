#!/usr/bin/env python3

"""
Executing this file will save the mean and std values for all four channels, sampled from 100 images to a file.
These values can then be loaded to normalize the observations of the RL agent.
"""

import gym
import numpy as np
from termcolor import colored
import time
import pickle

env = gym.make("gym_grasper:Grasper-v0")

rgb_list = []
depth_list = []

# Get 100 images. Format of each observation: Dic containing 2 lists, rgb and height
for i in range(100):
    obs = env.reset()
    rgb_list.append(obs["rgb"])
    depth_list.append(obs["depth"])

print("Collected 100 images from the environment.")

rgb_arr = np.array(rgb_list)
# rgb_arr = np.array(rgb_list) / 255
depth_arr = np.array(depth_list)

red = rgb_arr[:, :, :, 0].flatten()
green = rgb_arr[:, :, :, 1].flatten()
blue = rgb_arr[:, :, :, 2].flatten()
depth = depth_arr.flatten()

# depth_min = np.min(depth)
# depth_max = np.max(depth)

# depth = (depth - depth_min) / (depth_max - depth_min)

print("Got {} pixel values.".format(red.shape[0]))

mean_red = np.mean(red)
print("Calculated mean value of {} for channel Red.".format(mean_red))
mean_green = np.mean(green)
print("Calculated mean value of {} for channel Green.".format(mean_green))
mean_blue = np.mean(blue)
print("Calculated mean value of {} for channel Blue.".format(mean_blue))
mean_depth = np.mean(depth)
print("Calculated mean value of {} for channel Depth.".format(mean_depth))

std_red = np.std(red)
print("Calculated standard deviation of {} for channel Red.".format(std_red))
std_green = np.std(green)
print("Calculated standard deviation of {} for channel Green.".format(std_green))
std_blue = np.std(blue)
print("Calculated standard deviation of {} for channel Blue.".format(std_blue))
std_depth = np.std(depth)
print("Calculated standard deviation of {} for channel Depth.".format(std_depth))

filename = "mean_and_std"

with open(filename, "wb") as file:
    pickle.dump(
        [mean_red, mean_green, mean_blue, mean_depth, std_red, std_green, std_blue, std_depth], file
    )

print("\nWrote values to file {}.".format(filename))

print(
    "\nYou can load the values as follows, this will return a list containing first the 4 mean values, then the 4 standard deviations.\n"
)

print(
    "with open('mean_and_std', 'rb') as file:\n    raw = file.read()\n    values = pickle.loads(raw)\n"
)
