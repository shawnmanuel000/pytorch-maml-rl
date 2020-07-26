import re
import time
import copy
import random
import mujoco_py
import numpy as np
from .base import MujocoEnv
import xml.etree.ElementTree as ET
import pybullet as pb

class PybulletServer(object):
    def __init__(self):
        self.server_id = None
        self.is_active = False
        self.bodies = {}
        self.connect()

    def connect(self):
        if not self.is_active:
            self.server_id = pb.connect(pb.GUI)
            pb.resetSimulation(physicsClientId=self.server_id)
            self.is_active = True

    def disconnect(self):
        if self.is_active:
            pb.disconnect(physicsClientId=self.server_id)
            self.bodies = {}
            self.is_active = False

pybullet_server = None

class Arm(MujocoEnv):
    def __init__(self):
        super().__init__('Arm.mjcf', frame_skip=2)
        
    def init_task(self, root):
        option = root.find("option")
        option.set("gravity", "0 0 0")
        worldbody = root.find("worldbody")
        self.path_names = []
        path = ET.Element("body")
        path.set("name", "path")
        path.set("pos", "0 0 0")
        for i in range(10):
            name = f"path{i}"
            point = ET.Element("body")
            point.set("name", name)
            point.set("pos", "0 0 0")
            point.append(ET.fromstring("<geom conaffinity='0' contype='0' pos='0 0 0' rgba='0.8 0.2 0.4 0.8' size='.002' type='sphere'/>"))
            path.append(point)
            self.path_names.append(name)
        worldbody.append(path)
        self.range = 1.0
        self.origin = np.array([0, -0.2, 0.3])
        self.size = np.array([0.1, 0.2, 0])
        self.space = "box"
        size = self.size[0] if self.space == "sphere" else np.maximum(self.size, 0.001)
        size_str = lambda x: ' '.join([f'{p}' for p in x])
        space = f"<body name='space' pos='{size_str(self.origin)}'><geom conaffinity='0' group='1' contype='0' name='space' rgba='0.9 0.9 0.9 0.4' size='{size_str(size)}' type='{self.space}'/></body>"
        worldbody.append(ET.fromstring(space))
        target = "<body name='target' pos='0 -0.20 .2'><geom conaffinity='0' group='1' contype='0' name='target' pos='0 0 0' rgba='0.4 0.8 0.2 1' size='.005' type='sphere'/></body>"
        worldbody.append(ET.fromstring(target))
        return root

    def reset_task(self, xoffset=-1, **kwargs):
        origin = self.origin + xoffset*np.array([0.11, 0, 0.0])
        target_pos = origin + self.range*self.size*np.random.uniform(-1, 1, size=self.size.shape)
        ef_pos = origin + self.range*self.size*np.random.uniform(-1, 1, size=self.size.shape)
        self.model.body_pos[self.model.body_names.index("target")] = target_pos
        self.model.body_pos[self.model.body_names.index("space")] = origin
        dirn = ef_pos-self.model.body_pos[self.model.body_names.index("body0")]
        dist = np.linalg.norm(dirn)
        angle = np.pi/2 - np.arccos(dist/(2*0.155))
        rot = np.arctan2(-dirn[1],-dirn[0]) + int(dirn[0]>0)*np.pi
        sign = -1 if dirn[0]<0 else 1
        qpos = [rot, angle*sign, (np.pi-2*angle)*sign]
        qvel = self.init_qvel + np.random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        ef_pos = self.get_body_pos("fingertip")
        target_pos = self.get_body_pos("target")
        points = np.linspace(ef_pos, target_pos, len(self.path_names))
        path_indices = [self.model.body_names.index(name) for name in self.path_names]
        for i,point in zip(path_indices, points):
            self.model.body_pos[i] = point

    def task_reward(self):
        ef_pos = self.get_body_pos("fingertip")
        target_pos = self.get_body_pos("target")
        path = [self.get_body_pos(name) for name in self.path_names]
        target_dist = ef_pos-target_pos
        path_dists = [ef_pos-path_pos for path_pos in path]
        reward_goal = -np.linalg.norm(target_dist)*2
        reward_path = -(np.min(np.linalg.norm(path_dists, axis=-1))**2)
        reward = reward_goal + reward_path
        return reward

    def task_state(self):
        path = [self.get_body_pos(name) for name in self.path_names]
        return np.concatenate([*path])

    def task_done(self):
        ef_pos = self.get_body_pos("fingertip")
        target_pos = self.get_body_pos("target")
        return np.linalg.norm(ef_pos-target_pos)<0.005

    def observation(self):
        pos = self.get_body_pos("fingertip")
        return np.concatenate([pos, super().observation()])

    def sample_tasks(self, num_tasks):
        tasks = [-1 for i in range(num_tasks)]
        return tasks

