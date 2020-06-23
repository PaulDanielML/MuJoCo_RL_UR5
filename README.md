# MuJoCo Simulation Setup of a UR5 robot arm for Reinforcement Learning 

## Work in progress! Current grasping gym environment version: 1.0 <br/> Next TO-DOs: add multiple objects, respawn them at random positions when resetting the environment

**Author:** Paul Daniel (pdd@mp.aau.dk)

This repository provides several python classes for control of robotic arms in MuJoCo: 

 * **MJ_Controller:** This class can be used as a standalone class for basic robot control in MuJoCo. This can be useful for trying out models and their grasping capabilities. 
 Alternatively, its methods can also be used by any other class (like a Gym environment) to provide some more functionality. One example of this might be to move the robot back into a certain position after every episode of training, which might be preferable compared to just resetting all the joint angles and velocities. 
 The controller currently also holds the methods for image transformations, which might be put into another separate class at some point. 

* **GraspEnv:** A Gym environment for training reinforcement learning agents. Currently a basic lifting task is implemented. 
The difference to most other MuJoCo Gym environments is that the observation returned is a camera image instead of a state vector of the simulation. This is meant to resemble a real world setup more closely. 

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



## **Usage**

### **GraspEnv - class:**

The file [*example_agent.py*](example_agent.py) demonstrates the use of a random agent for this environment.
The created environment has an associated controller object, which provides all the functionality of the *MJ_Controller* - class to it. 
* **Action space**: Pixel space, can be specified by setting height and width. Current defaults: 200x200. This means there are 40.000 possible actions. This resolution translates to a picking accuracy of ~ 4mm.
* **State space**: The states / observations provided are dictionaries containing two arrays: An RGB-image and a depth-image, both of the same resolution as the action space
* **Reward function**: Currently the environment can yield three different rewards: 
    * -10 if a pixel is chosen that is outside certain limits (on the floor or the robot itself).
    * 0 for choosing a pixel within the reachable space of the robot that does not lead to a successful grasp.
    * +100 for choosing a pixel that leads to a successful grasp.

The user gets a summary of each step performed in the console. It is recommended to train agents without rendering, as this will speed up training significantly. 
    
![console](/media/console.png "Example console output during training")

The rgb part of the last captured observation will be shown and updated in an extra window.

![observation](/media/observation_rgb.png "Example observation")

### **MJ_Controller - class:**

Example usage of some of the class methods is demonstrated in the file [*example.py*](example.py).

The class *MJ_Controller* offers high and low level methods for controlling the robot in MuJoCo. 

* **move_ee** : High level, moves the endeffector of the arm to the desired XYZ position (in world 					coordinates). This is done using very simple inverse kinematics, just obeying the joint limits. Currently there is not collision avoidance implemented. Since this whole repo is created with grasping in mind, the delivered pose will always be so that the gripper is oriented in a vertical way (for top down grasps).
* **actuate_joint_group** :  Low level, lets the user specify motor activations for a specified group
* **grasp** : Uses the specified gripper group to attempt a grasp. A simple check is done to determine weather the grasp was successful or not and the result will be output blinking in the console. 

![gif1](/media/gif_1.gif "Simple Grasp and Toss")

## **Updates**

**Record grasps:** The step method of the *GraspingEnv* now has the optional parameter *record_grasps*. If set to True, it will capture a side camera image every time a grasp is made that is deemed successful by the environment. This allows for "quality control" of the grasps, without having to watch all the failed attempts. The captured images can also be useful for fine tuning grasping parameters. 

![grasp](/media/grasp.png "Example grasp")

**Point clouds:** The controller class was provided with new methods for image transformations. 
* depth_2_meters: Converts the normalized depth values returned by mujoco_py into m.
* create_camera_data: Constructs a camera matrix, focal length and sets the camera's position and rotation based on a provided camera name and desired image width and depth. 
* world_2_pixel: Accepts a XYZ world position and returns the corresponding x-y pixel coordinates 
* pixel_2_world: Accepts x-y pixel coordinates and a depth value, returns the XYZ world position. This method can be used to construct point clouds out of the data returned by the controllers *get_image_data* method.

![cloud](/media/point_cloud.png "Example point cloud")

**Joint plots:** The methods *move_ee* and *move_group_to_joint_target* now have an optional *plot* parameter. If set to True, a .png-file will be created in the local directory. It will show plots for each joint involved in the trajectory, containing the joint angles over time, as well as the target values. This can be used to determine which joints overshoot, oscillate etc. and adjust the controller gains based on that.  
The tolerance used for the trajectory are plotted in red, so it can easily be determined how many steps each of the joints needs to reach a value within tolerance. 

![plot1](/media/plot_1.png "Example plot")
