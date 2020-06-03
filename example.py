from MujocoController import MJ_Controller

controller = MJ_Controller()

controller.move_ee([0.03, -0.6 , 1.2])
controller.move_ee([0.03, -0.6 , 1.1])
controller.stay(1000)
controller.grasp()
controller.move_ee([0.03, -0.6 , 1.2])