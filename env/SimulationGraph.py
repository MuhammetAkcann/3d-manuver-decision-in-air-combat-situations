import numpy as np

VOLUME_CHART_HEIGHT = 1


class SimulationGraph:
    """A simulation visualization using matplotlib made to render OpenAI gym environments"""

    def __init__(self, title=None):
        self.locations = []
        self.rewards = []

    def render(self, new_data1, new_data2, reward):
        self.locations.append([new_data1.copy(), new_data2.copy()])
        self.rewards.append(reward.copy())

    def close(self, episode):
        nparr = np.array(self.locations).T
        np_rewards = np.array(self.rewards)

        print("self.episode:", episode)

        f1 = "sims/2/" + "locations_" + str(episode) + ".npy"
        f2 = "sims/2/" + "rewards_" + str(episode) + ".npy"

        np.save(f1, nparr)
        np.save(f2, np_rewards)
