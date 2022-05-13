import gym
from gym import spaces
import numpy as np
from env.SimulationGraph import SimulationGraph
from env.uav import UAV

LOOKBACK_WINDOW_SIZE = 40


def get_na(uav1, uav2):
    der_pu = np.array([uav1.der_x, uav1.der_y, uav1.der_z])
    der_pt = np.array([uav2.der_x, uav2.der_y, uav2.der_z])
    pu = np.array(uav1.position)
    pt = np.array(uav2.position)

    temp1 = (der_pu @ (pt - pu)) / np.linalg.norm(pt - pu)
    temp2 = (der_pt @ (pu - pt)) / np.linalg.norm(pu - pt)

    na = temp1 - temp2

    return na


def get_nd(uav1, uav2, beta1=8, beta2=0.7, d_max=30, d_min=0):
    distance = np.linalg.norm(np.array(uav1.position) - np.array(uav2.position))

    f_t = (d_max - abs(distance)) / (d_max - d_min)
    s_t = 1 - np.exp(-(abs(distance) - d_min) ** beta2)
    return_val = beta1 * f_t * s_t
    return return_val


class SimulationEnv(gym.Env):
    """A 3D flight simulation environment for OpenAI gym"""
    metadata = {'render.modes': ['live', 'file', 'none']}
    visualization = None

    def __init__(self):
        super(SimulationEnv, self).__init__()
        self.uav1 = UAV()
        self.uav2 = UAV()
        self.uav_list = [self.uav1, self.uav2]
        self.num_states = 11
        self.num_actions = 27
        self.current_reward = [0, 0]  # first is orientation reward, second is distance reward
        self.dt = 0.1  # each step time
        self.g = 9.81
        self.episode = 0
        self.current_step = 0
        # Set actionSpace
        # first variable is roll change: 0 -> left; 1 -> steady; 2 -> right
        # second variable is pitch change: 0 -> down; 1 -> steady; 2 -> up
        # first variable is speed change: 0 -> decelerate; 1 -> steady; 2 -> accelerate
        self.action_space = spaces.Box(low=0, high=1, shape=(3, 1), dtype=np.float16)

        # Set Observation space
        self.observation_space = spaces.Box(low=0, high=1, shape=(11, 1), dtype=np.float16)

    def calculate_advantage(self, w1=0.5, w2=0.5):
        self.current_reward[0], self.current_reward[1] = get_na(self.uav1, self.uav2), get_nd(self.uav1, self.uav2)
        adv = w1 * get_na(self.uav1, self.uav2) + w2 * get_nd(self.uav1, self.uav2) - self.current_step * 0.02, \
              w1 * get_na(self.uav2, self.uav1) + w2 * get_nd(self.uav2, self.uav1) - self.current_step * 0.02
        return adv

    def _next_observation(self, uav1, uav2):
        obs = np.zeros(self.observation_space.shape[0])

        # first uav
        obs[0] = 1 / (1 + np.exp(-(uav1.position[0] - uav2.position[0]) / 30))
        obs[1] = 1 / (1 + np.exp(-(uav1.position[1] - uav2.position[1]) / 30))
        obs[2] = 1 / (1 + np.exp(-(uav1.position[2] - uav2.position[2]) / 30))

        obs[3] = (uav1.roll + 23) / 46
        obs[4] = (uav1.pitch + 23) / 46
        obs[5] = uav1.yaw / 360

        obs[6] = (uav1.speed - 4) / 16

        # second uav
        obs[7] = (uav2.roll - 23) / 46
        obs[8] = (uav2.pitch - 23) / 46
        obs[9] = uav2.yaw / 360

        obs[10] = (uav2.speed - 4) / 16
        return obs

    def _take_action(self, action):
        for i, uav in enumerate(self.uav_list):
            uav.roll = uav.roll + max(-1, min(1, action[i * 3 + 0])) * self.dt * uav.der_roll
            uav.pitch = uav.pitch + max(-1, min(1, action[i * 3 + 1])) * self.dt * uav.der_pitch
            uav.speed = uav.speed + max(-1, min(1, action[i * 3 + 2])) * self.dt * uav.der_speed

            # check for boundaries
            if uav.roll > uav.max_roll:
                uav.roll = uav.max_roll
            elif uav.roll < -uav.max_roll:
                uav.roll = -uav.max_roll

            if uav.pitch > uav.max_pitch:
                uav.pitch = uav.max_pitch
            elif uav.pitch < -uav.max_pitch:
                uav.pitch = -uav.max_pitch

            if uav.speed > uav.max_speed:
                uav.speed = uav.max_speed
            elif uav.speed < uav.min_speed:
                uav.speed = uav.min_speed

            n = 1 / np.cos(np.radians(uav.roll))
            der_yaw = np.degrees((self.g * np.sqrt(n ** 2 - 1)) / uav.speed)
            uav.yaw += der_yaw * self.dt
            uav.yaw = uav.yaw % 360

            uav.der_x = uav.speed * np.cos(np.radians(uav.yaw)) * np.cos(np.radians(uav.pitch))
            uav.der_y = uav.speed * np.sin(np.radians(uav.yaw)) * np.cos(np.radians(uav.pitch))
            uav.der_z = uav.speed * np.sin(np.radians(uav.pitch))

            uav.position[0] += uav.der_x * self.dt
            uav.position[1] += uav.der_y * self.dt
            uav.position[2] += uav.der_z * self.dt

    def step(self, action):
        # Execute one time step within the environment
        # action[3] = 0
        # action[4] = 0
        # action[5] = -1
        self._take_action(action)

        # print("self.calculate_advantage():", self.calculate_advantage())

        self.current_step += 1

        thresh = 12 - self.current_step * 0.02

        reward, rival_reward = self.calculate_advantage()

        done = 1 if reward > thresh else 0
        done = -1 if rival_reward > thresh else done

        if done == 1:
            reward = 50 - self.current_step * 0.02
        #if done == -1:
        #    reward = -50

        obs = self._next_observation(self.uav1, self.uav2)
        obs_rival = self._next_observation(self.uav2, self.uav1)

        return obs, obs_rival, reward, done, rival_reward

    def reset(self):
        # Reset the state of the environment to an initial state
        self.uav1 = UAV()
        self.uav2 = UAV()
        self.uav_list = [self.uav1, self.uav2]
        self.current_step = 0

        return self._next_observation(self.uav1, self.uav2), self._next_observation(self.uav2, self.uav1)

    def render(self, mode='live', **kwargs):
        # Render the environment to the screen
        if mode == 'live':
            if self.visualization is None:
                self.visualization = SimulationGraph(
                    kwargs.get('title', None))
            if self.current_step is not None:
                self.visualization.render(self.uav1.position, self.uav2.position, self.current_reward)

    def close(self):
        if self.visualization is not None:
            self.episode += 1
            self.visualization.close(self.episode)
            self.visualization = None
