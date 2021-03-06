import os
import numpy as np
import pybullet as pb
import xml.etree.ElementTree as ET
from collections import OrderedDict

from ..utils.transform_utils import convert_quat
from ..robots import SingleArm
from ..models import assets_root
from ..models.arenas import TableArena, WipeForceTableArena
from ..models.objects import BoxObject
from ..models.tasks import TableTopTask, UniformRandomSampler, WipeForceTableTask
from ..controllers import get_pybullet_server, load_controller_config, controller_factory
from ..controllers.ee_ik import PybulletServer
from .robot_env import RobotEnv


class SurfaceWipe(RobotEnv):
	"""
	This class corresponds to the surface wiping task for a single robot arm.
	"""
	def __init__(
		self,
		robots,
		controller_configs=None,
		gripper_types="WipingGripper",
		gripper_visualizations=False,
		initialization_noise=0.02,
		table_full_size=[0.5, 0.5, 0.8],
		table_friction=[0.00001, 0.005, 0.0001],
		use_camera_obs=True,
		use_object_obs=True,
		reward_scale=2.25,
		reward_shaping=False,
		placement_initializer=None,
		use_indicator_object=False,
		has_renderer=False,
		has_offscreen_renderer=True,
		render_camera="frontview",
		render_collision_mesh=False,
		render_visual_mesh=True,
		control_freq=100,
		horizon=250,
		ignore_done=False,
		camera_names="agentview",
		camera_heights=256,
		camera_widths=256,
		camera_depths=False,
		height=0
		):
		"""
		Args:
			robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms) Note: Must be a single single-arm robot!
			controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a custom controller. Else, uses the default controller for this specific task. Should either be single dict if same controller is to be used for all robots or else it should be a list of the same length as "robots" param
			gripper_types (str or list of str): type of gripper, used to instantiate gripper models from gripper factory. Default is "default", which is the default grippers(s) associated with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model overrides the default gripper. Should either be single str if same gripper type is to be used for all robots or else it should be a list of the same length as "robots" param
			gripper_visualizations (bool or list of bool): True if using gripper visualization. Useful for teleoperation. Should either be single bool if gripper visualization is to be used for all robots or else it should be a list of the same length as "robots" param
			initialization_noise (float or list of floats): The scale factor of uni-variate Gaussian random noise applied to each of a robot's given initial joint positions. Setting this value to "None" or 0.0 results in no noise being applied. Should either be single float if same noise value is to be used for all robots or else it should be a list of the same length as "robots" param
			table_full_size (3-tuple): x, y, and z dimensions of the table.
			table_friction (3-tuple): the three mujoco friction parameters for the table.
			use_camera_obs (bool): if True, every observation includes rendered image(s)
			use_object_obs (bool): if True, include object (cube) information in the observation.
			reward_scale (float): Scales the normalized reward function by the amount specified
			reward_shaping (bool): if True, use dense rewards.
			placement_initializer (ObjectPositionSampler instance): if provided, will be used to place objects on every reset, else a UniformRandomSampler is used by default.
			use_indicator_object (bool): if True, sets up an indicator object that is useful for debugging.
			has_renderer (bool): If true, render the simulation state in a viewer instead of headless mode.
			has_offscreen_renderer (bool): True if using off-screen rendering
			render_camera (str): Name of camera to render if `has_renderer` is True.
			render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.
			render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.
			control_freq (float): how many control signals to receive in every second. This sets the amount of simulation time that passes between every action input.
			horizon (int): Every episode lasts for exactly @horizon timesteps.
			ignore_done (bool): True if never terminating the environment (ignore @horizon).
			camera_names (str or list of str): name of camera to be rendered. Should either be single str if same name is to be used for all cameras' rendering or else it should be a list of cameras to render.
				Note: At least one camera must be specified if @use_camera_obs is True.
				Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each robot's camera list).
			camera_heights (int or list of int): height of camera frame. Should either be single int if same height is to be used for all cameras' frames or else it should be a list of the same length as "camera names" param.
			camera_widths (int or list of int): width of camera frame. Should either be single int if same width is to be used for all cameras' frames or else it should be a list of the same length as "camera names" param.
			camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single bool if same depth setting is to be used for all cameras or else it should be a list of the same length as "camera names" param.
		"""
		# First, verify that only one robot is being inputted
		self.region_height = height
		self._check_robot_configuration(robots)
		# settings for table top
		self.table_full_size = table_full_size
		self.table_friction = table_friction
		# reward configuration
		self.reward_scale = reward_scale
		self.reward_shaping = reward_shaping
		# whether to use ground-truth object states
		self.use_object_obs = use_object_obs
		self.placement_initializer = UniformRandomSampler(x_range=[0, 0.2], y_range=[0, 0.2],ensure_object_boundary_in_range=False,z_rotation=True)
		super().__init__(robots=robots, controller_configs=controller_configs, gripper_types=gripper_types, gripper_visualizations=gripper_visualizations, initialization_noise=initialization_noise, use_camera_obs=use_camera_obs, use_indicator_object=use_indicator_object, has_renderer=has_renderer, has_offscreen_renderer=has_offscreen_renderer, render_camera=render_camera, render_collision_mesh=render_collision_mesh, render_visual_mesh=render_visual_mesh, control_freq=control_freq, horizon=horizon, ignore_done=ignore_done, camera_names=camera_names, camera_heights=camera_heights, camera_widths=camera_widths, camera_depths=camera_depths)
		self.init_qpos = self.robots[0].inverse_controller.inverse_kinematics([0.6,0,0.88], None)

	def _load_model(self):
		"""
		Loads an xml model, puts it in self.model
		"""
		super()._load_model()
		# Verify the correct robot has been loaded
		assert isinstance(self.robots[0], SingleArm), "Error: Expected one single-armed robot! Got {} type instead.".format(type(self.robots[0]))
		
		self.robots[0].robot_model._init_qpos = np.array([0.0,0.2,-0.0615045,-2.17912,-0.0257932,2.5,0.798416])
		self.robots[0].robot_model.set_base_xpos([0,0,0])

		self.table_origin = [0.36 + self.table_full_size[0] / 2, 0, 0]
		self.table_friction_std = 0
		self.table_height_std = 0.0
		self.num_squares = [4,4]
		self.real_robot = False
		self.prob_sensor = 1.0
		self.table_rot_x = 0.0
		self.table_rot_y = 0.0
		self.line_width = 0.01
		self.two_clusters = False
		self.draw_line = True  # whether you want a line for sensors
		self.num_sensors = 10
		self.wiped_sensors = []
		self.collisions = 0
		self.f_excess = 0
		self.unit_wiped_reward = 10
		self.distance_multiplier = 0.1
		self.distance_th_multiplier = 0.1
		self.pressure_threshold = 5
		self.pressure_threshold_max = 30
		self.wipe_contact_reward = 0.01
		self.ee_accel_penalty = 0
		self.excess_force_penalty_mul = 0.01

		delta_height = min(0, np.random.normal(0.0, self.table_height_std))
		table_full_size_sampled = (self.table_full_size[0], self.table_full_size[1], self.table_full_size[2] + delta_height)
		self.mujoco_arena = WipeForceTableArena(table_full_size=table_full_size_sampled, table_friction=self.table_friction, table_friction_std=self.table_friction_std, num_squares=self.num_squares if not self.real_robot else 0, prob_sensor=self.prob_sensor, rotation_x=np.random.normal(0, self.table_rot_x), rotation_y=np.random.normal(0, self.table_rot_y), draw_line=self.draw_line, num_sensors=self.num_sensors if not self.real_robot else 0, line_width=self.line_width, two_clusters=self.two_clusters)
		self.mujoco_arena.set_origin(self.table_origin)
		self.mujoco_objects = {k:v for k,v in self.mujoco_arena.squares.items() if "contact" in k}
		
		self.model = WipeForceTableTask(self.mujoco_arena, [robot.robot_model for robot in self.robots], initializer=self.placement_initializer)
		self.model.place_objects()

	def _reset_internal(self, random=True, **kwargs):
		"""
		Resets simulation internal configurations.
		"""
		super()._reset_internal()
		self.model.place_objects()
		# reset joint positions
		# Small randomization of the initial configuration
		noise = np.random.randn(7) * 0.02 if random else np.zeros(7)
		self.sim.data.qpos[self.robots[0]._ref_joint_pos_indexes] = np.array(self.init_qpos) + noise
		self.timestep = 0
		self.wiped_sensors = []
		self.collisions = 0
		self.f_excess = 0
		self.ee_acc = np.zeros(6)
		self.ee_force = np.zeros(3)
		self.ee_force_bias = np.zeros(3)
		self.ee_torque = np.zeros(3)
		self.ee_torque_bias = np.zeros(3)
		self.prev_ee_pos = np.zeros(self.robots[0].dof)
		self.ee_pos = np.zeros(self.robots[0].robot_model.dof)
		src = np.random.uniform([-0.1,-0.1],[0.1,0.1],[2])
		dst = np.random.uniform([-0.1,-0.1],[0.1,0.1],[2])
		points = np.linspace(src, dst, len(self.mujoco_objects.keys()))
		points = (src+dst)/2 + np.random.randn_like(points)
		for i,name in enumerate(self.mujoco_objects):
			pos = points[i]
			site_pos = self.sim.model.site_pos[self.sim.model.site_names.index(name)]
			body_pos = self.sim.model.body_pos[self.sim.model.body_names.index(name)]
			self.sim.model.body_pos[self.sim.model.body_names.index(name)] = np.array([*pos,body_pos[2]])
			self.sim.model.site_pos[self.sim.model.site_names.index(name)] = np.array([*pos,site_pos[2]])
			self.sim.model.site_rgba[self.sim.model.site_name2id(name)] = [0, 1, 0, 1]
		
	def _get_observation(self):
		"""
		Returns an OrderedDict containing observations [(name_string, np.array), ...].
		Important keys:
			robot-state: contains robot-centric information.
			object-state: requires @self.use_object_obs to be True.
				contains object-centric information.
			image: requires @self.use_camera_obs to be True.
				contains a rendered frame from the simulation.
			depth: requires @self.use_camera_obs and @self.camera_depth to be True.
				contains a rendered depth map from the simulation
		"""
		di = super()._get_observation()
		# low-level object information
		if self.use_object_obs:
			pr = self.robots[0].robot_model.naming_prefix
			# gripper_position = np.array(self.sim.data.body_xpos[self.sim.model.body_name2id(pr+'right_hand')])
			# position of objects to wipe
			acc = np.array([])
			for sensor_name in self.model.arena.sensor_names:
				parts = sensor_name.split('_')
				sensor_id = int(parts[1])
				sensor_pos = np.array(self.sim.data.site_xpos[self.sim.model.site_name2id(self.model.arena.sensor_site_names[sensor_name])])
				# di['sensor' + str(sensor_id) + '_pos'] = sensor_pos
				acc = np.concatenate([acc, sensor_pos])
				acc = np.concatenate([acc, [[0, 1][sensor_id in self.wiped_sensors]]])
				# proprioception
				# di['gripper_to_sensor' + str(sensor_id)] = gripper_position - sensor_pos
				# acc = np.concatenate([acc, gripper_position - sensor_pos])
			di["target_pos"] = [0,0,0]
			di["task_state"] = np.concatenate([acc, di["robot0_eef_pos"]])
		return di

	def _check_contact(self):
		"""
		Returns True if gripper is in contact with an object.
		"""
		collision = False
		for contact in self.sim.data.contact[:self.sim.data.ncon]:
			if self.sim.model.geom_id2name(contact.geom1) in self.robots[0].robot_model.gripper.contact_geoms() or self.sim.model.geom_id2name(contact.geom2) in self.robots[0].robot_model.gripper.contact_geoms():
				collision = True
				break
		return collision

	def _check_arm_contact(self):
		"""
		Returns True if the arm is in contact with another object.
		"""
		collision = False
		for contact in self.sim.data.contact[:self.sim.data.ncon]:
			if self.sim.model.geom_id2name(contact.geom1) in self.robots[0].robot_model.contact_geoms or self.sim.model.geom_id2name(contact.geom2) in self.robots[0].robot_model.contact_geoms:
				collision = True
				break
		return collision

	def _check_q_limits(self, debug=False):
		"""
		Returns True if the arm is in joint limits or very close to.
		"""
		joint_limits = False
		tolerance = 0.1
		for (idx, (q, q_limits)) in enumerate(
				zip(self.sim.data.qpos[self.robots[0]._ref_joint_pos_indexes], self.sim.model.jnt_range)):
			if not (q > q_limits[0] + tolerance and q < q_limits[1] - tolerance):
				if debug: print("Joint limit reached in joint " + str(idx))
				joint_limits = True
				self.robots[0].joint_limit_count += 1
		return joint_limits

	def _post_action(self, action):
		"""
		(Optional) does gripper visualization after actions.
		"""
		self.prev_ee_pos = self.ee_pos
		pr = self.robots[0].robot_model.naming_prefix
		self.ee_pos = np.array(self.sim.data.body_xpos[self.sim.model.body_name2id(pr+'right_hand')])

		pr = self.robots[0].gripper.naming_prefix
		force_sensor_id = self.sim.model.sensor_name2id(pr+"force_ee")
		self.ee_force = np.array(self.sim.data.sensordata[force_sensor_id * 3: force_sensor_id * 3 + 3])

		if np.linalg.norm(self.ee_force_bias) == 0:
			self.ee_force_bias = self.ee_force

		torque_sensor_id = self.sim.model.sensor_name2id(pr+"torque_ee")
		self.ee_torque = np.array(self.sim.data.sensordata[torque_sensor_id * 3: torque_sensor_id * 3 + 3])

		if np.linalg.norm(self.ee_torque_bias) == 0:
			self.ee_torque_bias = self.ee_torque

		return super()._post_action(action)

	def reward(self, action=None):
		"""
			Reward function for the task.
			The dense un-normalized reward has three components.
				Reaching: in [0, 1], to encourage the arm to reach the cube
				Grasping: in {0, 0.25}, non-zero if arm is grasping the cube
				Lifting: in {0, 1}, non-zero if arm has lifted the cube
			The sparse reward only consists of the lifting component.
			Note that the final reward is normalized and scaled by
			reward_scale / 2.25 as well so that the max score is equal to reward_scale
			Args:
				action (np array): unused for this task
			Returns:
				reward (float): the reward
		"""
		reward = 0
		total_force_ee = np.linalg.norm(np.array(self.ee_force))
		# Neg Reward from collisions of the arm with the table
		if self._check_arm_contact() or self._check_q_limits():
			self.collisions += 1
			reward -= 10
		else:
			# TODO: Use the sensed touch to shape reward
			# Only do computation if there are active sensors and they weren't active before
			sensors_active_ids = []
			# Current 3D location of the corners of the wiping tool in world frame
			pr = self.robots[0].gripper.naming_prefix
			corner1_id = self.sim.model.geom_name2id(pr+"wiping_corner1")
			corner1_pos = np.array(self.sim.data.geom_xpos[corner1_id])
			corner2_id = self.sim.model.geom_name2id(pr+"wiping_corner2")
			corner2_pos = np.array(self.sim.data.geom_xpos[corner2_id])
			corner3_id = self.sim.model.geom_name2id(pr+"wiping_corner3")
			corner3_pos = np.array(self.sim.data.geom_xpos[corner3_id])
			corner4_id = self.sim.model.geom_name2id(pr+"wiping_corner4")
			corner4_pos = np.array(self.sim.data.geom_xpos[corner4_id])
			touching_points = [np.array(corner1_pos), np.array(corner2_pos), np.array(corner3_pos),np.array(corner4_pos)]
			# Unit vectors on my plane
			v1 = corner1_pos - corner2_pos
			v1 /= np.linalg.norm(v1)
			v2 = corner4_pos - corner2_pos
			v2 /= np.linalg.norm(v2)
			# Corners of the tool in the coordinate frame of the plane
			t1 = np.array([np.dot(corner1_pos - corner2_pos, v1), np.dot(corner1_pos - corner2_pos, v2)])
			t2 = np.array([np.dot(corner2_pos - corner2_pos, v1), np.dot(corner2_pos - corner2_pos, v2)])
			t3 = np.array([np.dot(corner3_pos - corner2_pos, v1), np.dot(corner3_pos - corner2_pos, v2)])
			t4 = np.array([np.dot(corner4_pos - corner2_pos, v1), np.dot(corner4_pos - corner2_pos, v2)])
			pp = [t1, t2, t4, t3]
			# Normal of the plane defined by v1 and v2
			n = np.cross(v1, v2)
			n /= np.linalg.norm(n)

			def area(p1, p2, p3):
				return abs(0.5 * ((p1[0] - p3[0]) * (p2[1] - p1[1]) - (p1[0] - p2[0]) * (p3[1] - p1[1])))

			def isPinRectangle(r, P, printing=False):
				"""
					r: A list of four points, each has a x- and a y- coordinate
					P: A point
				"""
				areaRectangle = area(r[0], r[1], r[2]) + area(r[1], r[2], r[3])
				ABP = area(r[0], r[1], P)
				BCP = area(r[1], r[2], P)
				CDP = area(r[2], r[3], P)
				DAP = area(r[3], r[0], P)
				inside = abs(areaRectangle - (ABP + BCP + CDP + DAP)) < 1e-6
				if printing:
					print(areaRectangle)
					print((ABP + BCP + CDP + DAP))
				return inside

			def isLeft(P0, P1, P2):
				return ((P1[0] - P0[0]) * (P2[1] - P0[1]) - (P2[0] - P0[0]) * (P1[1] - P0[1]))

			def PointInRectangle(X, Y, Z, W, P):
				return (isLeft(X, Y, P) < 0 and isLeft(Y, Z, P) < 0 and isLeft(Z, W, P) < 0 and isLeft(W, X, P) < 0)

			if self.sim.data.ncon != 0:
				for sensor_name in self.model.arena.sensor_names:
					# Current sensor 3D location in world frame
					# sensor_pos = np.array(self.sim.data.body_xpos[self.sim.model.site_bodyid[self.sim.model.site_name2id(self.model.arena.sensor_site_names[sensor_name])]])
					sensor_pos = np.array(self.sim.data.site_xpos[self.sim.model.site_name2id(self.model.arena.sensor_site_names[sensor_name])])
					# We use the second tool corner as point on the plane and define the vector connecting
					# the sensor position to that point
					v = sensor_pos - corner2_pos
					# Shortest distance between the center of the sensor and the plane
					dist = np.dot(v, n)
					# Projection of the center of the sensor onto the plane
					projected_point = np.array(sensor_pos) - dist * n
					# Positive distances means the center of the sensor is over the plane
					# The plane is aligned with the bottom of the wiper and pointing up, so the sensor would be over it
					if dist > 0.0:
						# Distance smaller than this threshold means we are close to the plane on the upper part
						if dist < 0.02:
							# Write touching points and projected point in coordinates of the plane
							pp_2 = np.array([np.dot(projected_point - corner2_pos, v1), np.dot(projected_point - corner2_pos, v2)])
							# if isPinRectangle(pp, pp_2):
							if PointInRectangle(pp[0], pp[1], pp[2], pp[3], pp_2):
								parts = sensor_name.split('_')
								sensors_active_ids += [int(parts[1])]
			lall = np.where(np.isin(sensors_active_ids, self.wiped_sensors, invert=True))
			new_sensors_active_ids = np.array(sensors_active_ids)[lall]
			for new_sensor_active_id in new_sensors_active_ids:
				new_sensor_active_site_id = self.sim.model.site_name2id(self.model.arena.sensor_site_names['contact_' + str(new_sensor_active_id) + '_sensor'])
				self.sim.model.site_rgba[new_sensor_active_site_id] = [0, 0, 0, 0]
				self.wiped_sensors += [new_sensor_active_id]
				reward += self.unit_wiped_reward
			mean_distance_to_things_to_wipe = 0
			num_non_wiped_sensors = 0
			for sensor_name in self.model.arena.sensor_names:
				parts = sensor_name.split('_')
				sensor_id = int(parts[1])
				if sensor_id not in self.wiped_sensors:
					pr = self.robots[0].robot_model.naming_prefix
					sensor_pos = np.array(self.sim.data.site_xpos[self.sim.model.site_name2id(self.model.arena.sensor_site_names[sensor_name])])
					gripper_position = np.array(self.sim.data.body_xpos[self.sim.model.body_name2id(pr+'right_hand')])
					mean_distance_to_things_to_wipe += np.linalg.norm(gripper_position - sensor_pos)
					num_non_wiped_sensors += 1
			mean_distance_to_things_to_wipe /= max(1, num_non_wiped_sensors)
			reward += self.distance_multiplier * (1 - np.tanh(self.distance_th_multiplier * mean_distance_to_things_to_wipe))
			# Reward for keeping contact
			if self.sim.data.ncon != 0:
				reward += 0.001
			if total_force_ee > self.pressure_threshold_max:
				reward -= self.excess_force_penalty_mul * total_force_ee
				self.f_excess += 1
			elif total_force_ee > self.pressure_threshold and self.sim.data.ncon > 1:
				reward += self.wipe_contact_reward + 0.01 * total_force_ee
				if self.sim.data.ncon > 50:
					reward += 10 * self.wipe_contact_reward
			# Final reward if all wiped
			if len(self.wiped_sensors) == len(self.model.arena.sensor_names):
				reward += 50 * (self.wipe_contact_reward + 0.5)  # So that is better to finish that to stay touching the table for 100 steps
				# The 0.5 comes from continuous_distance_reward at 0. If something changes, this may change as well
		# Penalize large accelerations
		reward -= self.ee_accel_penalty * np.mean(abs(self.ee_acc))
		return reward * self.reward_scale / 2.25

	def done(self, debug=False):
		"""
		Returns True if task is successfully completed
		"""
		terminated = False
		# Prematurely terminate if contacting the table with the arm
		if self._check_arm_contact():
			if debug: print(40 * '-' + " COLLIDED " + 40 * '-')
			terminated = True
		# Prematurely terminate if finished
		if len(self.wiped_sensors) == len(self.model.arena.sensor_names):
			if debug: print(40 * '+' + " FINISHED WIPING " + 40 * '+')
			terminated = True
		# force_sensor_id = self.sim.model.sensor_name2id("force_ee")
		# force_ee = self.sensor_data[force_sensor_id*3: force_sensor_id*3+3]
		# if np.linalg.norm(np.array(force_ee)) > 3*self.pressure_threshold_max:
		#     print(35*'*' + " TOO MUCH FORCE " + str(np.linalg.norm(np.array(force_ee))) + 35*'*')
		#     terminated = True
		# Prematurely terminate if contacting the table with the arm
		if self._check_q_limits():
			if debug: print(40 * '-' + " JOINT LIMIT " + 40 * '-')
			terminated = True
		return terminated

	def _visualization(self):
		"""
		Do any needed visualization here. Overrides superclass implementations.
		"""

	def _check_robot_configuration(self, robots):
		"""
		Sanity check to make sure the inputted robots and configuration is acceptable
		"""
		if type(robots) is list:
			assert len(robots) == 1, "Error: Only one robot should be inputted for this task!"

