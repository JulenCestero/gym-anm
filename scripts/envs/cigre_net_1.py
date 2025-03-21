import json
import pickle

import numpy as np
import pandas as pd

from gym_anm import ANMEnv

# Load the network from a JSON file
with open("data/networks/cigre_network_1.json", "r") as f:
    network = json.load(f)

# Convert lists in the JSON to numpy arrays
for key, value in network.items():
    if isinstance(value, list):
        network[key] = np.array(value)


class CigreNetwork(ANMEnv):
    """An example of a simple 2-bus custom gym-anm environment."""

    def __init__(self, network=network, p_loads_loaded="", p_max_loaded=""):
        observation = "state"  # fully observable environment
        K = 1  # 1 auxiliary variable
        delta_t = 0.5  # 30min intervals
        gamma = 0.995  # discount factor
        lamb = 100  # penalty weighting hyperparameter
        aux_bounds = np.array([[0, 24 / delta_t - 1]])  # bounds on auxiliary variable
        costs_clipping = (1, 100)  # reward clipping parameters
        # seed = 1  # random seed
        self.network = network
        self.p_loads_loaded = p_loads_loaded  # TODO: Load the path to the P_loads.pkl file
        self.p_max_loaded = p_max_loaded

        super().__init__(self.network, observation, K, delta_t, gamma, lamb, aux_bounds, costs_clipping)

        self.P_loads = self._get_load_time_series()
        self.P_maxs = self._get_gen_time_series()

    def init_state(self):
        n_dev, n_gen, n_des = 22, 5, 1

        state = np.zeros(2 * n_dev + n_des + n_gen + self.K)

        t_0 = self.np_random.integers(0, int(24 / self.delta_t))
        state[-1] = t_0

        self.network_df = pd.DataFrame(self.network["device"])
        # Load (P, Q) injections.
        load_ids = self.network_df[self.network_df[2] == -1][0].values
        for load_id, p_load in zip(load_ids, self.P_loads):
            state[load_id] = p_load[t_0]
            state[n_dev + load_id] = p_load[t_0] * self.simulator.devices[load_id].qp_ratio

        gen_ids = self.network_df[self.network_df[2].isin([1, 2])][0].values
        # Non-slack generator (P, Q) injections.
        for idx, (dev_id, p_max) in enumerate(zip(gen_ids, self.P_maxs)):
            state[2 * n_dev + n_des + idx] = p_max[t_0]
            state[dev_id] = p_max[t_0]
            state[n_dev + dev_id] = self.np_random.uniform(
                self.simulator.devices[dev_id].q_min, self.simulator.devices[dev_id].q_max
            )

        # Energy storage unit.
        des_ids = self.network_df[self.network_df[2] == 3][0].values
        for idx, dev_id in enumerate(des_ids):
            state[2 * n_dev + idx] = self.np_random.uniform(
                self.simulator.devices[dev_id].soc_min, self.simulator.devices[dev_id].soc_max
            )

        return state

    def next_vars(self, s_t):
        aux = int((s_t[-1] + 1) % (24 / self.delta_t))

        vars = []
        for p_load in self.P_loads:
            vars.append(p_load[aux])
        for p_max in self.P_maxs:
            vars.append(p_max[aux])

        vars.append(aux)

        return np.array(vars)

    def _get_load_time_series(self):
        """Return the fixed 24-hour time-series for the load injections."""

        # Load pickle from ../data/preprocessed/P_loads.pkl
        with open(self.p_loads_loaded, "rb") as f:
            data = pickle.load(f)

        if data is None:
            P_loads = np.random.rand(len(self.network_df[self.network_df[2] == -1][0].values), 24 / self.delta_t) * -1
        else:
            P_loads = data

        return P_loads

    def _get_gen_time_series(self):
        """Return the fixed 24-hour time-series for the generator maximum production."""
        # Load pickle from ../data/preprocessed/P_maxs.pkl
        with open(self.p_max_loaded, "rb") as f:
            data = pickle.load(f)

        if data is None:
            P_maxs = np.random.rand(len(self.network_df[self.network_df[2].isin([1, 2])][0].values), 24 / self.delta_t)

        else:
            P_maxs = data

        return P_maxs


if __name__ == "__main__":
    env = CigreNetwork()
    env.reset()

    for t in range(10):
        a = env.action_space.sample()
        o, r, terminated, _, _ = env.step(a)
        print(f"t={t}, r_t={r:.3}")
