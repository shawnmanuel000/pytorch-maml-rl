import mujoco_py
import numpy as np
from collections import OrderedDict
from mujoco_py import MjSim, MjRenderContextOffscreen
from mujoco_py import load_model_from_xml

from ..utils import SimulationError, XMLError, MujocoPyRenderer

REGISTERED_ENVS = {}
DEFAULT_SIZE = 500


def register_env(target_class):
    REGISTERED_ENVS[target_class.__name__] = target_class


def make(env_name, *args, **kwargs):
    """Try to get the equivalent functionality of gym.make in a sloppy way."""
    if env_name not in REGISTERED_ENVS:
        raise Exception(
            "Environment {} not found. Make sure it is a registered environment among: {}".format(
                env_name, ", ".join(REGISTERED_ENVS)
            )
        )
    return REGISTERED_ENVS[env_name](*args, **kwargs)


class EnvMeta(type):
    """Metaclass for registering environments"""

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)

        # List all environments that should not be registered here.
        _unregistered_envs = ["MujocoEnv", "RobotEnv"]

        if cls.__name__ not in _unregistered_envs:
            register_env(cls)
        return cls


class MujocoEnv(metaclass=EnvMeta):
    """Initializes a Mujoco Environment."""
    def __init__(
        self,
        has_renderer=False,
        has_offscreen_renderer=False,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        control_freq=10,
        horizon=1000,
        ignore_done=False,
    ):
        """
        Args:
            has_renderer (bool): If true, render the simulation state in a viewer instead of headless mode.
            has_offscreen_renderer (bool): True if using off-screen rendering.
            render_camera (str): Name of camera to render if `has_renderer` is True.
            render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.
            render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.
            control_freq (float): how many control signals to receive in every simulated second. This sets the amount of simulation time that passes between every action input.
            horizon (int): Every episode lasts for exactly @horizon timesteps.
            ignore_done (bool): True if never terminating the environment (ignore @horizon).
        """
        # Rendering-specific attributes
        self.has_renderer = has_renderer
        self.has_offscreen_renderer = has_offscreen_renderer
        self.render_camera = render_camera
        self.render_collision_mesh = render_collision_mesh
        self.render_visual_mesh = render_visual_mesh
        self.viewer = None
        self.viewers = {}

        # Simulation-specific attributes
        self.control_freq = control_freq
        self.horizon = horizon
        self.ignore_done = ignore_done
        self.model = None
        self.cur_time = None
        self.model_timestep = None
        self.control_timestep = None
        self.deterministic_reset = False            # Whether to add randomized resetting of objects / robot joints

        # Load the model
        self._load_model()

        # Initialize the simulation
        self._initialize_sim()

        # Run all further internal (re-)initialization required
        self._reset_internal()

    def initialize_time(self, control_freq):
        """
        Initializes the time constants used for simulation.
        """
        self.cur_time = 0
        self.model_timestep = self.sim.model.opt.timestep
        if self.model_timestep <= 0:
            raise XMLError("xml model defined non-positive time step")
        self.control_freq = control_freq
        if control_freq <= 0:
            raise SimulationError("control frequency {} is invalid".format(control_freq))
        self.control_timestep = 1. / control_freq

    def _load_model(self):
        """Loads an xml model, puts it in self.model"""
        pass

    def _get_reference(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        pass

    def _initialize_sim(self, xml_string=None):
        """
        Creates a MjSim object and stores it in self.sim. If @xml_string is specified, the MjSim object will be created
        from the specified xml_string. Else, it will pull from self.model to instantiate the simulation
        """
        # if we have an xml string, use that to create the sim. Otherwise, use the local model
        self.mjpy_model = load_model_from_xml(xml_string) if xml_string else self.model.get_model(mode="mujoco_py")

        # Create the simulation instance and run a single step to make sure changes have propagated through sim state
        self.sim = MjSim(self.mjpy_model)
        self.sim.step()

        self.init_qpos = np.copy(self.sim.data.qpos)
        self.init_qvel = np.copy(self.sim.data.qvel)

        # Setup sim time based on control frequency
        self.initialize_time(self.control_freq)

    def reset(self, **kwargs):
        """Resets simulation."""
        # TODO(yukez): investigate black screen of death
        self._reset_internal(**kwargs)
        self.sim.forward()
        return self._get_observation()

    def _reset_internal(self):
        """Resets simulation internal configurations."""

        # create visualization screen or renderer
        if self.has_renderer and self.viewer is None:
            self.viewer = MujocoPyRenderer(self.sim)
            self.viewer.viewer.vopt.geomgroup[0] = (1 if self.render_collision_mesh else 0)
            self.viewer.viewer.vopt.geomgroup[1] = (1 if self.render_visual_mesh else 0)

            # hiding the overlay speeds up rendering significantly
            self.viewer.viewer._hide_overlay = True

            # make sure mujoco-py doesn't block rendering frames
            # (see https://github.com/StanfordVL/robosuite/issues/39)
            self.viewer.viewer._render_every_frame = True

            # Set the camera angle for viewing
            self.viewer.set_camera(camera_id=self.sim.model.camera_name2id(self.render_camera))

        elif self.has_offscreen_renderer:
            if self.sim._render_context_offscreen is None:
                render_context = MjRenderContextOffscreen(self.sim)
                self.sim.add_render_context(render_context)
            self.sim._render_context_offscreen.vopt.geomgroup[0] = (1 if self.render_collision_mesh else 0)
            self.sim._render_context_offscreen.vopt.geomgroup[1] = (1 if self.render_visual_mesh else 0)

        # additional housekeeping
        self.sim_state_initial = self.sim.get_state()
        self._get_reference()
        self.cur_time = 0
        self.timestep = 0

    def _get_observation(self):
        """Returns an OrderedDict containing observations [(name_string, np.array), ...]."""
        return OrderedDict()

    def step(self, action):
        """Takes a step in simulation with control command @action."""
        self.timestep += 1

        # Since the env.step frequency is slower than the mjsim timestep frequency, the internal controller will output
        # multiple torque commands in between new high level action commands. Therefore, we need to denote via
        # 'policy_step' whether the current step we're taking is simply an internal update of the controller,
        # or an actual policy update
        policy_step = True

        # Loop through the simulation at the model timestep rate until we're ready to take the next policy step
        # (as defined by the control frequency specified at the environment level)
        try:
            for i in range(int(self.control_timestep / self.model_timestep)):
                self._pre_action(action, policy_step)
                self.sim.step()
                policy_step = False
        except:
            print("error. resetting")
            self.reset()
            return self.step(action)

        # Note: this is done all at once to avoid floating point inaccuracies
        self.cur_time += self.control_timestep
        next_state = self._get_observation()
        reward, done, info = self._post_action(action)
        return next_state, reward, done, info

    def _pre_action(self, action, policy_step=False):
        """Do any preprocessing before taking an action."""
        self.sim.data.ctrl[:] = action

    def _post_action(self, action):
        """Do any housekeeping after taking an action."""
        reward = self.reward(action)
        done = self.done()
        return reward, done, {}

    def reward(self, action):
        """Reward should be a function of state and action."""
        return 0

    def done(self):
        # done if number of elapsed timesteps is greater than horizon
        return (self.timestep >= self.horizon) and not self.ignore_done

    def render(self, mode='human', width=DEFAULT_SIZE, height=DEFAULT_SIZE, camera_name=None):
        if mode == 'rgb_array':
            camera_id = self.mjpy_model.camera_name2id(camera_name) if camera_name in self.mjpy_model._camera_name2id else None
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
            self.viewer.vopt.geomgroup[0] = (1 if self.render_collision_mesh else 0)
            self.viewer.vopt.geomgroup[1] = (1 if self.render_visual_mesh else 0)
            self.viewer._hide_overlay = True
            self.viewer._render_every_frame = True
            self.viewer.cam.trackbodyid = 0
            self.viewer.cam.azimuth = 180
            self.viewer.cam.elevation = -15
            self.viewers[mode] = self.viewer
        return self.viewer

    def observation_spec(self):
        """
        Returns an observation as observation specification.

        An alternative design is to return an OrderedDict where the keys
        are the observation names and the values are the shapes of observations.
        We leave this alternative implementation commented out, as we find the
        current design is easier to use in practice.
        """
        observation = self._get_observation()
        return observation

        # observation_spec = OrderedDict()
        # for k, v in observation.items():
        #     observation_spec[k] = v.shape
        # return observation_spec

    @property
    def action_spec(self):
        """
        Action specification should be implemented in subclasses.

        Action space is represented by a tuple of (low, high), which are two numpy
        vectors that specify the min/max action limits per dimension.
        """
        raise NotImplementedError

    def reset_from_xml_string(self, xml_string):
        """Reloads the environment from an XML description of the environment."""

        # if there is an active viewer window, destroy it
        self.close()

        # Since we are reloading from an xml_string, we are deterministically resetting
        self.deterministic_reset = True

        # initialize sim from xml
        self._initialize_sim(xml_string=xml_string)

        # Now reset as normal
        self.reset()

        # Turn off deterministic reset
        self.deterministic_reset = False

    def find_contacts(self, geoms_1, geoms_2):
        """
        Finds contact between two geom groups.

        Args:
            geoms_1: a list of geom names (string)
            geoms_2: another list of geom names (string)

        Returns:
            iterator of all contacts between @geoms_1 and @geoms_2
        """
        for contact in self.sim.data.contact[0 : self.sim.data.ncon]:
            # check contact geom in geoms
            c1_in_g1 = self.sim.model.geom_id2name(contact.geom1) in geoms_1
            c2_in_g2 = self.sim.model.geom_id2name(contact.geom2) in geoms_2
            # check contact geom in geoms (flipped)
            c2_in_g1 = self.sim.model.geom_id2name(contact.geom2) in geoms_1
            c1_in_g2 = self.sim.model.geom_id2name(contact.geom1) in geoms_2
            if (c1_in_g1 and c2_in_g2) or (c1_in_g2 and c2_in_g1):
                yield contact

    def _check_contact(self):
        """Returns True if gripper is in contact with an object."""
        return False

    def _check_success(self):
        """
        Returns True if task has been completed.
        """
        return False

    def _destroy_viewer(self):
        # if there is an active viewer window, destroy it
        if self.viewer is not None:
            self.viewer.close()  # change this to viewer.finish()?
            self.viewer = None

    def get_body_pos(self, body_name):
        # pos = self.sim.data.get_body_xpos(body_name)
        pos = self.sim.model.body_pos[self.sim.model.body_names.index(body_name)]
        return pos

    def set_state(self, qpos=None, qvel=None):
        old_state = self.sim.get_state()
        qpos = old_state.qpos if qpos is None else qpos
        qvel = old_state.qvel if qvel is None else qvel
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel, old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()
        self.sim.step()

    def close(self):
        """Do any cleanup necessary here."""
        self._destroy_viewer()
