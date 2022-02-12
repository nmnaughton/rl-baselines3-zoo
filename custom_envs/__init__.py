import gym
from gym.envs.registration import registry, make, spec
from .RC_env import RC_env

def register(id, *args, **kvargs):
  if id in registry.env_specs:
    return
  else:
    return gym.envs.registration.register(id, *args, **kvargs)

print('Trying to register custom envs')


register(
    id='RC_env-v1',
    entry_point=RC_env,
    # max_episode_steps=2000,
    # reward_threshold=2000.0,
)