

# MuJoCo Simulation Setup of a UR5 robot arm for Reinforcement Learning 

## Work in progress! More functionality and RL parts coming soon. Currently implementing gym interface. 

**Author:** Paul Daniel (pdd@mp.aau.dk)

This repository provides a python class for control of the UR5 robot arm in MuJoCo. 
The robot configuration used in this setup (Universal Robots UR5 + Robotiq S Model 3 Finger Gripper) is based on [this](http://www.mujoco.org/forum/index.php?resources/universal-robots-ur5-robotiq-s-model-3-finger-gripper.22/) resource.  
The python bindings used come from [mujoco_py](https://github.com/openai/mujoco-py/tree/master/mujoco_py).  
The PID controllers implemented are based on [simple_pid](https://github.com/m-lundberg/simple-pid).  
A simple inverse kinematics solver for translating end-effector positions into joint angles has been implemented using [ikpy](https://github.com/Phylliade/ikpy).

The required modules can be installed either manually or using the provided requirements.txt - file.

## **Setup**

Download and install MuJoCo from [here](https://www.roboti.us/index.html). Set up a license and activate it [here](https://www.roboti.us/license.html).

Then clone this repo: 
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

![gif1](/media/gif_1.gif "Simple Grasp and Toss")

## **Usage**

Example usage of some of the class methods is demonstrated in the file [*example.py*](example.py).

The class *Mujoco_UR5_controller* offers high and low level methods for controlling the robot in MuJoCo. 

* **move_ee** : High level, moves the endeffector of the arm to the desired XYZ position (in world 					coordinates). This is done using very simple inverse kinematics, just obeying the joint limits. Currently there is not collision avoidance implemented. Since this whole repo is created with grasping in mind, the delivered pose will always be so that the gripper is oriented in a vertical way (for top down grasps).
* **actuate_joint_group** :  Low level, lets the user specify motor activations for a specified group
* **grasp** : Uses the specified gripper group to attempt a grasp. A simple check is done to determine weather the grasp was successful or not and the result will be output blinking in the console. 

## **Updates**

**New feature:** The methods *move_ee* and *move_group_to_joint_target* now have an optional *plot* parameter. If set to True, a .png-file will be created in the local directory. It will show plots for each joint involved in the trajectory, containing the joint angles over time, as well as the target values. This can be used to determine which joints overshoot, oscillate etc. and adjust the controller gains based on that.  
The tolerance used for the trajectory are plotted in red, so it can easily be determined how many steps each of the joints needs to reach a value within tolerance. 

![plot1](/media/plot_1.png "Example plot")
