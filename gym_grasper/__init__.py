vfrom gym.envs.registration import register

register(
    id='Grasper-v0',
    entry_point='gym_grasper.envs:GraspEnv',
)