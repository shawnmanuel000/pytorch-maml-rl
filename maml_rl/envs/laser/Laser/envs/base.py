import io
import os
import mujoco_py
import numpy as np
import pybullet as pb
import xml.etree.ElementTree as ET
from collections import OrderedDict
from ...Gym import gym

DEFAULT_SIZE = 500

REGISTERED_ENVS = {}

class EnvMeta(type):
    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)
        if cls.__name__ not in ["MujocoEnv"]:
            REGISTERED_ENVS[cls.__name__] = cls
        return cls

class MujocoEnv(gym.Env, metaclass=EnvMeta):
    def __init__(self, model_path, frame_skip):
        self.fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        self.model = self.load_model(self.fullpath)
        self.sim = mujoco_py.MjSim(self.model)
        self.init_qpos = np.copy(self.sim.data.qpos)
        self.init_qvel = np.copy(self.sim.data.qvel)
        action_low, action_high = self.model.actuator_ctrlrange.astype(np.float32).T
        self.action_space = gym.spaces.Box(low=action_low, high=action_high, dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.Inf, np.Inf, self.observation().shape)
        self.robot_state_size = self.observation_space.shape[-1] - len(self.task_state())
        self.frame_skip = frame_skip
        self.viewers = {}
        self.viewer = None

    def load_model(self, path):
        self.tree = ET.parse(path)
        root = self.tree.getroot()
        root = self.init_task(root)
        with io.StringIO() as string:
            string.write(ET.tostring(root, encoding="unicode"))
            model = mujoco_py.load_model_from_xml(string.getvalue())
        return model

    def init_task(self, root):
        return root

    def reset_task(self):
        raise NotImplementedError()

    def task_state(self):
        raise NotImplementedError()

    def task_reward(self):
        raise NotImplementedError()

    def task_done(self):
        raise NotImplementedError()

    def observation(self):
        qvel = self.sim.data.qvel
        qpos = self.sim.data.qpos
        qpos = np.concatenate([np.sin(qpos), np.cos(qpos)])
        return np.concatenate([qpos, qvel, self.task_state()])

    def reset(self, **kwargs):
        self.sim.reset()
        self.reset_task(**kwargs)
        state = self.observation()
        return state

    def step(self, action):
        self.sim.data.ctrl[:] = action
        for _ in range(self.frame_skip):
            try:
                self.sim.step()
            except:
                print("Sim error")
                self.reset()
                return self.step(action)
        state = self.observation()
        reward = self.task_reward()
        done = self.task_done()
        return state, reward, done, {}

    def render(self, mode='human', width=DEFAULT_SIZE, height=DEFAULT_SIZE, camera_name=None):
        if mode == 'rgb_array':
            camera_id = self.model.camera_name2id(camera_name) if camera_name in self.model._camera_name2id else None
            self.get_viewer(mode).render(width, height, camera_id=camera_id)
            data = self.get_viewer(mode).read_pixels(width, height, depth=False)
            return data[::-1, :, :]
        elif mode == 'depth_array':
            self.get_viewer(mode).render(width, height)
            data = self.get_viewer(mode).read_pixels(width, height, depth=True)[1]
            return data[::-1, :]
        elif mode == 'human':
            self.get_viewer(mode).render()

    def get_viewer(self, mode):
        self.viewer = self.viewers.get(mode)
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim) if mode in ["human"] else mujoco_py.MjRenderContextOffscreen(self.sim, -1) if mode in ["rgb_array", "depth_array"] else None
            self.viewer._hide_overlay = True
            self.viewer._render_every_frame = True
            self.viewer.cam.trackbodyid = 0
            # self.viewer.cam.azimuth = 180
            # self.viewer.cam.elevation = -15
            self.viewers[mode] = self.viewer
        return self.viewer

    def get_body_pos(self, body_name):
        pos = self.sim.data.get_body_xpos(body_name)
        return pos

    def close(self):
        if self.viewer:
            self.viewer = None
            self.viewers = {}

    def set_state(self, qpos, qvel=None):
        qpos, qvel = map(np.array, [qpos, qvel])
        assert qpos.shape == (self.model.nq,) and (qvel is None or qvel.shape == (self.model.nv,))
        old_state = self.sim.get_state()
        qvel = old_state.qvel if qvel is None else qvel
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel, old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()
        self.sim.step()
