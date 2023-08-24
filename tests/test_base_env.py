from physilearning.envs import BaseEnv
from gym.spaces import Box

def test_construct_base_env():
    env = BaseEnv()
    assert env


def test_observation_space():
    obs_space = 'number'
    env = BaseEnv(observation_type=obs_space, normalize_to=20)
    box = Box(low=0, high=20, shape=(3,))
    assert (env.observation_space == box)

