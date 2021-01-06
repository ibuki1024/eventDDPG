import gym2
from gym2 import spaces
from gym2.utils import seeding
import numpy as np
from os import path

import sys
sys.path.append('../../../')
from rl2.barrier_certificate import h, set_alpha


class LinearEnv(gym2.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        """
        write me
        """
        self.A = np.array([[-1, 4], [2, -3]])
        self.B = np.array([2, 4])

        """
        high = np.array([1., 1., self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque, shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )
        """

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u, dt, tau):
        """Struct discretized system suitable for any tau."""
        x = self.state  # th := theta
        u = u[0]
        
        self.last_u = u  # for rendering
        costs = .01*(x[0] ** 2 + x[1] ** 2)

        Ad, Bd = discretized_system(self.A, self.B, dt)

        x_prime = np.dot(Ad, x) + np.dot(Bd, u)
        x_prime = np.clip(x_prime, -7, 7)

        self.state = np.array(x_prime)
        return self._get_obs(), -costs, False, {}

    # modify to change start position
    def reset(self):
        high = np.array([np.pi, np.pi]) # start with inverted point
        self.state = self.np_random.uniform(low=-high, high=high) # th=0, -1<thd<1
        self.last_u = None
        return self._get_obs()

    def set_state(self, x):
        self.state = x

    def _get_obs(self):
        x = self.state
        # return np.array([np.cos(theta), np.sin(theta), thetadot])
        return np.array(x)

    def render(self, mode='human'):
        if self.viewer is None:
            from gym2.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def discretized_system(A, B, dt):
    Ad = np.eye(A.shape[0]) + dt * A
    Bd = dt * B
    return Ad, Bd
