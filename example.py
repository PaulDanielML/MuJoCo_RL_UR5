from gym_grasper.controller.MujocoController import MJ_Controller

# create controller instance
controller = MJ_Controller()

# Display robot information
controller.show_model_info()

# Move ee to position above the object, plot the trajectory to an image file, show a marker at the target location
controller.move_ee([0.0, -0.6, 0.95], plot=True, marker=True)

# Move down to object
controller.move_ee([0.0, -0.6, 0.895])

# Wait a second
controller.stay(1000)

# Attempt grasp
controller.grasp()

# Move up again
controller.move_ee([0.0, -0.6, 1.0])

# Throw the object away
controller.toss_it_from_the_ellbow()

# Wait before finishing
controller.stay(2000)
