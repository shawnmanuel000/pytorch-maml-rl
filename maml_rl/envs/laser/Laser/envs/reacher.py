import re
import copy
import random
import numpy as np
from .base import MujocoEnv
import xml.etree.ElementTree as ET

class Reacher(MujocoEnv):
    def __init__(self):
        super().__init__('Reacher.xml', frame_skip=2)

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
        self.origin = np.array([0, 0, 0.3])
        self.size = np.maximum([0.25, 0.25, 0], 0.001)
        space = f"<geom conaffinity='0' contype='0' name='space' pos='{' '.join([f'{p}' for p in self.origin])}' rgba='0.2 0.2 0.2 0.1' size='{self.size[0]}' type='sphere'/>"
        el = ET.fromstring(space)
        worldbody.append(el)
        return root

    def reset_task(self, task):
        rand = np.random.uniform(-1, 1, size=self.size.shape)
        while np.linalg.norm(rand) > 1 or np.linalg.norm(rand) < 0.1 or rand[1]>0:
            rand = np.random.uniform(-1, 1, size=self.size.shape)
        target_pos = self.origin + self.range*self.size*rand
        self.model.body_pos[self.model.body_names.index("target")] = target_pos
        qpos = 0.1*np.random.uniform(low=-1, high=1, size=self.model.nq) + self.init_qpos
        qpos[0] = 0.5*np.random.uniform(-3.14, 3.14)
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
        reward_path = -np.min(np.linalg.norm(path_dists, axis=-1))
        reward = reward_goal + reward_path
        return reward

    def task_state(self):
        path = [self.get_body_pos(name) for name in self.path_names]
        return np.concatenate([*path])

    def task_done(self):
        return False

    def observation(self):
        pos = self.get_body_pos("fingertip")
        return np.concatenate([super().observation(), pos])

    def sample_tasks(self, num_tasks):
        tasks = [{'id': i} for i in range(num_tasks)]
        return tasks
