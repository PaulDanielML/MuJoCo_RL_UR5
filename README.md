# MuJoCo Simulation Setup of a UR5 robot arm for Reinforcement Learning 

## Work in progress! Code cleanup, refactoring, more functionality and RL parts coming soon!

**Author:** Paul Daniel (pdd@mp.aau.dk)

This repository provides a python class for control of the UR5 robot arm in MuJoCo. 
The robot configuration used in this setup (Universal Robots UR5 + Robotiq S Model 3 Finger Gripper) is based on [this](http://www.mujoco.org/forum/index.php?resources/universal-robots-ur5-robotiq-s-model-3-finger-gripper.22/) resource.
The python bindings used come from [mujoco_py](https://github.com/openai/mujoco-py/tree/master/mujoco_py).
The PID controllers implemented are based on [simple_pid](https://github.com/m-lundberg/simple-pid).
A simple inverse kinematics solver for translating end-effector positions into joint angles has been implemented using [ikpy](https://github.com/Phylliade/ikpy).

The required modules can be installed either manually or using the provided requirements.txt - file. 

First clone this repo: 
```
git clone https://github.com/PaulDanielML/MuJoCo_RL_UR5.git
```
Then change into the newly created directory:
```
cd MuJoCo_RL_UR5/
```
If desired, activate a virtual environment, then run 
```
pip install -r requirements.txt
```
This will install all required packages using pip. The first time you run a script that uses the *Mujoco_UR5_controller* class some more setup might happen, which can take a few moments.
This is all the setup required to use this repo.


![img1](https://github.com/PaulDanielML/MuJoCo_RL_UR5/tree/master/media/pic_1.png "UR5 MuJoCo Setup")
