import os
import numpy as np
import matplotlib.pyplot as plt
gym, rbs = None, None
os.system("export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/shawn/.mujoco/mujoco200/bin")
os.system("export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so")

from .Laser import laser

try: from .Gym import gym
except ImportError as e: print(e)

try: from .Robosuite import robosuite as rbs
except ImportError as e: print(e)

class RawPreprocess():
	def __init__(self, env):
		self.observation_space = env.observation_space

	def __call__(self, state):
		return state

class AtariPreprocess():
	def __init__(self, env):
		self.observation_space = gym.spaces.Box(0, 255, shape=(105, 80, 3))

	def __call__(self, state):
		assert state.shape == (210,160,3)
		state = state[::2,::2] # downsample by factor of 2
		return state

def get_preprocess(env):
	return RawPreprocess(env)

class GymEnv(gym.Wrapper):
	def __init__(self, env, **kwargs):
		super().__init__(env)
		self.unwrapped.verbose = 0
		self.preprocess = get_preprocess(env)
		self.observation_space = self.preprocess.observation_space

	def reset(self, **kwargs):
		self.time = 0
		state = self.env.reset()
		return self.preprocess(state)

	def step(self, action, train=False):
		self.time += 1
		state, reward, done, info = super().step(action)
		return self.preprocess(state), reward, done, info

class CustomEnv(gym.Env):
	def __init__(self, env_name, max_steps):
		self.new_spec = gym.envs.registration.EnvSpec(env_name, max_episode_steps=max_steps)

	@property
	def spec(self):
		return self.new_spec

	@spec.setter
	def spec(self, v):
		max_steps = self.new_spec.max_episode_steps
		self.new_spec = v
		self.new_spec.max_episode_steps = max_steps

class LaserEnv(gym.Wrapper):
	def __init__(self, env_name):
		fields = env_name.split('-')
		env_name = fields[0]
		self.env = laser.make(env_name)
		self.use_region1 = not fields[1:] or "R1" in fields[1:]
		self.dynamics_size = self.env.robot_state_size
		super().__init__(self.env)

	def reset(self, **kwargs):
		kwargs["xoffset"] = -1 if self.use_region1 else 1
		state = self.env.reset(**kwargs)
		return state

	def step(self, action):
		state, reward, done, info = self.env.step(action)
		return state, reward, done, info

	def render(self, mode="human", **kwargs):
		return self.env.render(mode=mode, **kwargs)

class RobosuiteEnv(CustomEnv):
	def __init__(self, env_name, render=False, pixels=False):
		self.pixels = pixels
		fields = env_name.split('-')
		task_name = fields[0]
		self.threeD = "3D" in fields[1:]
		self.ee_pos = "EP" in fields[1:]
		self.use_region1 = "R1" in fields[1:]
		controller = list(rbs.ALL_CONTROLLERS)[3 if self.ee_pos else 1]
		controller_config = rbs.controllers.load_controller_config(default_controller=controller)
		self.env = rbs.make(task_name, ["Panda"], controller_configs=controller_config, has_offscreen_renderer=False, ignore_done=True, horizon=250, use_camera_obs=pixels, height=0.2*int(self.threeD))
		observation_spec = self.env.observation_spec()
		self.robot_state_size = len(observation_spec["robot0_robot-state"])
		self.observation_size = self.observation(observation_spec).shape
		low, high = (0,255) if pixels else (-np.inf,np.inf)
		self.observation_space = gym.spaces.Box(low=low, high=high, shape=self.observation_size)
		self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,) if self.ee_pos else (self.env.action_dim,))
		super().__init__(env_name+"-v0", self.env.horizon if self.env.horizon else 250)

	def reset(self, **kwargs):
		kwargs["xoffset"] = 0 if self.threeD else -1 if self.use_region1 else 1
		obs_spec = self.env.reset(**kwargs)
		return self.observation(obs_spec)

	def step(self, action):
		if self.ee_pos: action = np.concatenate([action, np.zeros(self.env.action_dim-len(action))])
		obs_spec, reward, done, info = self.env.step(action)
		return self.observation(obs_spec), reward, done, info

	def render(self, mode="human", **kwargs):
		return self.env.render(mode=mode, **kwargs)

	def observation(self, obs_spec):
		robot_state = obs_spec["robot0_robot-state"]
		task_state = obs_spec["task_state"]
		return  obs_spec["image"] if self.pixels else np.concatenate([robot_state, task_state])

	def sample_tasks(self, num_tasks):
		tasks = [{} for i in range(num_tasks)]
		return tasks

	def reset_task(self, task):
		self.reset(**task)