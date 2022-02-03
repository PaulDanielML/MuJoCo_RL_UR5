from gym.envs.registration import register
from gym_grasper.version import VERSION as __version__

register(
    id="Grasper-v0",
    entry_point="gym_grasper.envs:GraspEnv",
)
