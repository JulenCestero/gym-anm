import json

import numpy as np
import pandas as pd

from gym_anm import ANMEnv


class CigreNetwork(ANMEnv):
    """An example of a custom gym-anm environment."""

    __name__ = "CigreNetwork"

    def __init__(self, network, p_loads_loaded=None, p_max_loaded=None):
        observation = "state"  # fully observable environment
        K = 1  # 1 auxiliary variable
        delta_t = 0.5  # 30min intervals
        gamma = 0.995  # discount factor
        lamb = 100  # penalty weighting hyperparameter
        aux_bounds = np.array([[0, 24 / delta_t - 1]])  # bounds on auxiliary variable
        costs_clipping = (1, 500)  # reward clipping parameters
        # seed = 1  # random seed
        self.network = network
        self.network_df = pd.DataFrame(self.network["device"])
        self.p_loads_loaded = p_loads_loaded
        self.p_max_loaded = p_max_loaded

        super().__init__(self.network, observation, K, delta_t, gamma, lamb, aux_bounds, costs_clipping)

        self.P_loads = self._get_load_time_series()
        self.P_maxs = self._get_gen_time_series()

    def init_state(self):
        n_dev, n_gen, n_des = self._get_network_devices_info(self.network_df)

        state = np.zeros(2 * n_dev + n_des + n_gen + self.K)

        t_0 = self.np_random.integers(0, int(24 / self.delta_t))
        state[-1] = t_0

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

    def _get_network_devices_info(self, network_df):
        """Return the number of devices, generators and energy storage units."""
        n_dev = len(network_df)
        n_gen = len(network_df[network_df[2].isin([1, 2])])
        n_des = len(network_df[network_df[2] == 3])
        return n_dev, n_gen, n_des

    def _get_load_time_series(self):
        """Return the fixed 24-hour time-series for the load injections."""
        data = None
        # Load pickle from ../data/preprocessed/P_loads.pkl
        if self.p_loads_loaded is not None:
            data = self.p_loads_loaded
        if data is None:
            # P_loads = (
            #     np.random.rand(len(self.network_df[self.network_df[2] == -1][0].values), int(24 / self.delta_t)) * -1
            # )
            s1 = -np.ones(25)
            s12 = np.linspace(-1.5, -4.5, 7)
            s2 = -5 * np.ones(13)
            s23 = np.linspace(-4.625, -2.375, 7)
            s3 = -2 * np.ones(13)
            P1 = np.concatenate((s1, s12, s2, s23, s3, s23[::-1], s2, s12[::-1], s1[:4]))
            P_loads = [P1] * len(self.network_df[self.network_df[2] == -1][0].values)
        else:
            P_loads = data

        return P_loads

    def _get_gen_time_series(self):
        """Return the fixed 24-hour time-series for the generator maximum production."""
        data = None
        # Load pickle from ../data/preprocessed/P_maxs.pkl
        if self.p_max_loaded is not None:
            data = self.p_max_loaded

        if data is None:
            # P_maxs = np.random.rand(
            #     len(self.network_df[self.network_df[2].isin([1, 2])][0].values), int(24 / self.delta_t)
            # )
            s1 = -np.ones(25)
            s12 = np.linspace(-1.5, -4.5, 7)
            s2 = -5 * np.ones(13)
            s23 = np.linspace(-4.625, -2.375, 7)
            s3 = -2 * np.ones(13)
            P1 = np.concatenate((s1, s12, s2, s23, s3, s23[::-1], s2, s12[::-1], s1[:4]))
            P_maxs = [P1] * len(self.network_df[self.network_df[2] == -1][0].values)
        else:
            P_maxs = data

        return P_maxs


def load_data():
    # Load the network from a JSON file
    with open("data/networks/cigre_5.json", "r") as f:
        network = json.load(f)

    # Convert lists in the JSON to numpy arrays
    for key, value in network.items():
        if isinstance(value, list):
            network[key] = np.array(value)
    # Load the network from a JSON file
    with open("data/demand_ts/cigre_5_ts.json", "r") as f:
        demand = json.load(f)

    # Convert lists in the JSON to numpy arrays
    for key, value in demand.items():
        if isinstance(value, list):
            demand[key] = np.array(value)
    return network, demand


if __name__ == "__main__":
    network, demand = load_data()
    env = CigreNetwork(network=network, p_loads_loaded=demand["P_loads"], p_max_loaded=demand["P_maxs"])
    env.reset()

    for t in range(10):
        a = env.action_space.sample()
        o, r, terminated, _, _ = env.step(a)
        print(f"t={t}, r_t={r}")
