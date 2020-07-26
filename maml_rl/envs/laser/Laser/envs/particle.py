import re
import copy
import random
import numpy as np
from .base import MujocoEnv
import xml.etree.ElementTree as ET

class Particle(MujocoEnv):
    def __init__(self):
        super().__init__('Particle.xml', frame_skip=2)

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
            point.append(ET.fromstring("<geom conaffinity='0' group='1' contype='0' pos='0 0 0' rgba='0.8 0.2 0.4 0.8' size='.002' type='sphere'/>"))
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

    def reset_task(self, xoffset=-1):
        origin = self.origin + xoffset*np.array([0.1, 0, 0.05])
        target_pos = origin + self.range*self.size*np.random.uniform(-1, 1, size=self.size.shape)
        ef_pos = origin + self.range*self.size*np.random.uniform(-1, 1, size=self.size.shape)
        self.model.body_pos[self.model.body_names.index("target")] = target_pos
        self.model.body_pos[self.model.body_names.index("space")] = origin
        points = np.linspace(ef_pos, target_pos, len(self.path_names))
        path_indices = [self.model.body_names.index(name) for name in self.path_names]
        for i,point in zip(path_indices, points):
            self.model.body_pos[i] = point
        qvel = self.init_qvel + np.random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(ef_pos, qvel)

    def task_reward(self):
        ef_pos = self.get_body_pos("fingertip")
        target_pos = self.get_body_pos("target")
        path = [self.get_body_pos(name) for name in self.path_names]
        target_dist = ef_pos-target_pos
        path_dists = [ef_pos-path_pos for path_pos in path]
        reward_goal = -np.linalg.norm(target_dist)*2
        reward_path = -np.min(np.linalg.norm(path_dists, axis=-1))
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
        return np.concatenate([super().observation(), pos])

    def sample_tasks(self, num_tasks):
        tasks = [-1 for i in range(num_tasks)]
        return tasks
