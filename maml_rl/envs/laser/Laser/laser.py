import os
from .envs import REGISTERED_ENVS, Reacher

ENV_NAMES = REGISTERED_ENVS.keys()

def make(env_name):
	return REGISTERED_ENVS[env_name]()