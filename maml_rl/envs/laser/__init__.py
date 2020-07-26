import os
import sys
import numpy as np
from .wrappers import GymEnv, LaserEnv, RobosuiteEnv, gym, rbs, laser
from collections import OrderedDict

rbs_extension_map = {"PathFollow": ["-EP-R1", "-EP-R2", "-R1", "-R2", "-3D"]}
rbs_envnames = [f"{rbs_name}{ext}" for rbs_name in rbs.ALL_ENVIRONMENTS for ext in rbs_extension_map.get(rbs_name, [''])]

lsr_extension_map = {"Particle": ["-R1", "-R2"], "Arm": ["-R1", "-R2"]}
lsr_envnames = [f"{lsr_name}{ext}" for lsr_name in laser.ENV_NAMES for ext in lsr_extension_map.get(lsr_name, [''])]

gym_types = ["classic_control", "box2d", "_atari", "mujoco", "robotics"]
env_grps = OrderedDict(
	gym_cct = tuple([env_spec.id for env_spec in gym.envs.registry.all() if type(env_spec.entry_point)==str and any(x in env_spec.entry_point for x in [gym_types[0]])]),
	gym_b2d = tuple([env_spec.id for env_spec in gym.envs.registry.all() if type(env_spec.entry_point)==str and any(x in env_spec.entry_point for x in [gym_types[1]])]),
	gym_atr = tuple([env_spec.id for env_spec in gym.envs.registry.all() if type(env_spec.entry_point)==str and any(x in env_spec.entry_point for x in [gym_types[2]])]),
	gym_mjc = tuple([env_spec.id for env_spec in gym.envs.registry.all() if type(env_spec.entry_point)==str and any(x in env_spec.entry_point for x in gym_types[3:4])]),
	gym = tuple([env_spec.id for env_spec in gym.envs.registry.all() if type(env_spec.entry_point)==str and any(x in env_spec.entry_point for x in gym_types)]),
	rbs = tuple(sorted([f"{r}" for r in rbs_envnames]) if rbs else []),
	lsr = tuple(sorted([f"{l}" for l in lsr_envnames])),
)

def get_group(env_name):
	for group, envs in reversed(env_grps.items()):
		if env_name in envs: return group
	return None

def get_names(groups):
	names = []
	for group in groups:
		names.extend(env_grps.get(group, []))
	return names

def make_env(cls, env_name):
	return lambda **kwargs: cls(env_name, **kwargs)

all_envs = get_names(["rbs", "lsr"])

for lsr_name in env_grps.get("lsr", []):
	gym.register(id=f"{lsr_name}-v0", entry_point=make_env(LaserEnv, lsr_name), max_episode_steps=250)

for rbs_name in env_grps.get("rbs", []):
	gym.register(id=f"{rbs_name}-v0", entry_point=make_env(RobosuiteEnv, rbs_name), max_episode_steps=250)

def get_env(env_name, render=False):
	kwargs = {} if get_group(env_name) in ["gym", "lsr"] else {"render":render}
	if get_group(env_name) in ["rbs", "lsr"]: env_name += "-v0"
	return GymEnv(gym.make(env_name, **kwargs))
