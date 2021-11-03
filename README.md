## Accompanying repository of Master's thesis at TU Berlin / Aalborg University. No longer under active development. Developed in my earlier Python days, please forgive the unformatted spaghetti code. 

# Deep Reinforcement Learning for robotic pick and place applications using purely visual observations 
**Author:** Paul Daniel (paudan22@gmail.com)

### Traits of this environment: Very large and multi-discrete actionspace, very high sample-cost, visual observations, binary reward.


Trained agent in action            |  Example of predicted grasp chances 
:---------------------------------:|:-------------------------:
![](/media/gif_3.gif "Trained Agent")  |  ![](/media/gif_4.gif "Overlay")


Setup iteration            |  Relevant changes 
:-------------------------:|:-------------------------:
  IT5 | - Many more objects, randomly piled <br> - Actionspace now multi-discrete, with second dimension being a rotation action
  IT4 | - Z-coordinate for grasping now calculated using depth data <br> - Objects now vary in size
  IT3 | - New two-finger gripper implemented 
  IT2 | - Grasp success check now after moving to drop location (1000 steps)
  IT1 (Baseline)|- Grasp success check after moving straight up (500 steps of trying to close the gripper) <br> - Fixed z-coordinate for grasping <br> - Objects of equal size


This repository provides several python classes for control of robotic arms in MuJoCo: 

 * **MJ_Controller:** This class can be used as a standalone class for basic robot control in MuJoCo. This can be useful for trying out models and their grasping capabilities. 
 Alternatively, its methods can also be used by any other class (like a Gym environment) to provide some more functionality. One example of this might be to move the robot back into a certain position after every episode of training, which might be preferable compared to just resetting all the joint angles and velocities. 
 The controller currently also holds the methods for image transformations, which might be put into another separate class at some point. 

* **GraspEnv:** A Gym environment for training reinforcement learning agents. The task to master is a pick & place task. 
The difference to most other MuJoCo Gym environments is that the observation returned is a camera image instead of a state vector of the simulation. This is meant to resemble a real world setup more closely. 

The robot configuration used in this setup (Universal Robots UR5 + Robotiq S Model 3 Finger Gripper) is based on [this](http://www.mujoco.org/forum/index.php?resources/universal-robots-ur5-robotiq-s-model-3-finger-gripper.22/) resource.  It has since been heavily modified. Most current XML-file: *UR5gripper_2_finger.xml*  
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

Gym-environment for training agents to use RGB-D data for predicting pixel-wise grasp success chances.  
The file [*example_agent.py*](example_agent.py) demonstrates the use of a random agent for this environment.  
The file [*Grasping_Agent.py*](Grasping_Agent.py) gives an example of training a shortsighted DQN-agent in the environment to predict pixel-wise grasping success (PyTorch).
The created environment has an associated controller object, which provides all the functionality of the *MJ_Controller* - class to it. 
* **Action space**: Pixel space, can be specified by setting height and width. Current defaults: 200x200. This means there are 40.000 possible actions. This resolution translates to a picking accuracy of ~ 4mm.
* **State space**: The states / observations provided are dictionaries containing two arrays: An RGB-image and a depth-image, both of the same resolution as the action space
* **Reward function**: The environment has been updated to a binary reward structure:
    * 0 for choosing a pixel that does not lead to a successful grasp.
    * +1 for choosing a pixel that leads to a successful grasp.

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

**Trials for Offline RL:** The folder *Offline RL* contains scripts for generating and learning from a dataset of (state, action, reward)-transitions. *generate_data.py* can be used to generate as many files as required, each file containing 12 transitions.

**New gripper model available:** A new, less bulky, 2-finger gripper was implemented in the model in training setup iteration 3. 

![new_gripper](/media/new_gripper.png "new gripper")

**Image normalization:** Added script *normalize.py*, which samples 100 images from the environment and writes the mean values and standard deviations of all channels to a file. 

**Reset shuffle:** Calling the environments *step* method now rearranges all the pickable objects to random positions on the table. 

![gif2](/media/gif_2.gif "Respawning")

**Record grasps:** The step method of the *GraspingEnv* now has the optional parameter *record_grasps*. If set to True, it will capture a side camera image every time a grasp is made that is deemed successful by the environment. This allows for "quality control" of the grasps, without having to watch all the failed attempts. The captured images can also be useful for fine tuning grasping parameters. 

![grasp](/media/grasp.png "Example grasp")

**Point clouds:** The controller class was provided with new methods for image transformations. 
* depth_2_meters: Converts the normalized depth values returned by mujoco_py into m.
* create_camera_data: Constructs a camera matrix, focal length and sets the camera's position and rotation based on a provided camera name and desired image width and depth. 
* world_2_pixel: Accepts a XYZ world position and returns the corresponding x-y pixel coordinates 
* pixel_2_world: Accepts x-y pixel coordinates and a depth value, returns the XYZ world position. This method can be used to construct point clouds out of the data returned by the controllers *get_image_data* method.

![cloud](/media/point_cloud.png "Example point cloud")

**Joint plots:** All methods that move joints now have an optional *plot* parameter. If set to True, a .png-file will be created in the local directory. It will show plots for each joint involved in the trajectory, containing the joint angles over time, as well as the target values. This can be used to determine which joints overshoot, oscillate etc. and adjust the controller gains based on that.  
The tolerance used for the trajectory are plotted in red, so it can easily be determined how many steps each of the joints needs to reach a value within tolerance. 

![plot1](/media/plot_1.png "Example plot")
