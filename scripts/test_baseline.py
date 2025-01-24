from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

MAX_ANM6_STEPS = 3000


class TimeLimitWrapper(gym.Wrapper):
    """
    This class imposes a maximum number of timesteps that can be taken in the
    environment before a `env.reset()` is required (at which point
    `done=True` is returned).

    :param env: (gym.Env) Gym environment that will be wrapped
    :param max_steps: (int) Max number of steps per episode
    """

    def __init__(self, env, max_steps):
        # Call the parent constructor, so we can access self.env later
        super(TimeLimitWrapper, self).__init__(env)
        self.max_steps = max_steps
        # Counter of steps per episode
        self.current_step = 0

    def reset(self, seed=None):
        """
        Reset the environment
        """
        # Reset the counter
        self.current_step = 0
        return self.env.reset()

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
        """
        self.current_step += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Overwrite the done signal when
        if self.current_step >= self.max_steps:
            truncated = True
            # Update the info dict to signal that the limit was exceeded
            info["time_limit_reached"] = True
        return obs, reward, terminated, truncated, info


def evaluate_policy_deprecated(policy, env, n_eval_episodes, need_return=False):
    results = [0] * n_eval_episodes
    energy_loss = [0] * n_eval_episodes
    penalty = [0] * n_eval_episodes
    for ii in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        step = MAX_ANM6_STEPS
        while not done and step > 0:
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, te, tr, info = env.step(action)
            done = te or tr
            if need_return:
                results[ii] *= 0.9
            if len(info) > 0:
                energy_loss[ii] += info["energy loss"]
                penalty[ii] += info["penalty"]
            results[ii] += reward
            step -= 1
    rr = np.array(results)
    return rr.mean(), rr.std(), {"energy_loss": np.array(energy_loss).mean(), "penalty": np.array(penalty).mean()}


class MonitorANM6(BaseCallback):
    def __init__(self, log, verbose: int = 0, early_stopping_count=np.inf):
        super().__init__(verbose)
        self.env = gym.make("gym_anm:ANM6Easy-v0")
        self.env = TimeLimitWrapper(self.env, max_steps=3000)
        self.logfile = Path(log)
        self.logfile.parent.mkdir(parents=True, exist_ok=True)
        self.early_stopping_flag = True
        self.early_stopping_initial = early_stopping_count
        self.early_stopping_count = self.early_stopping_initial
        self.best_reward = -np.inf
        with open(self.logfile, "w") as f:
            f.write("step,time,mean_reward,std_reward,energy_loss,penalty\n")

    def _on_rollout_start(self) -> None:
        """
        This event is triggered before collecting new samples.
        """
        mean_rwd = std_rwd = 0
        mean_rwd, std_rwd, info = evaluate_policy_deprecated(self.model, self.env, n_eval_episodes=1)
        with open(self.logfile, "a") as f:
            f.write(
                f"{self.num_timesteps},{datetime.now()},{mean_rwd},{std_rwd},{info['energy_loss']},{info['penalty']}\n"
            )
        if mean_rwd > self.best_reward:
            self.early_stopping_count = self.early_stopping_initial
            self.best_reward = mean_rwd
        else:
            self.early_stopping_count -= 1
            if self.early_stopping_count <= 0:
                self.early_stopping_flag = False
                print(
                    f"Training aborted early, after no policy improvement over {self.early_stopping_initial} rollouts"
                )

    def _on_training_end(self) -> None:
        return super()._on_training_end()

    def _on_step(self) -> bool:
        return self.early_stopping_flag


def test_baseline(logs_folder):
    logger.info("Training baseline model")
    env = gym.make("gym_anm:ANM6Easy-v0")
    env = TimeLimitWrapper(env, 3000)  # Limit length of training episodes
    csv_file = f"{logs_folder}/baseline.csv"
    callback = MonitorANM6(log=csv_file, verbose=0, early_stopping_count=15)

    model = PPO(
        "MlpPolicy", env, gamma=0.995, verbose=0, tensorboard_log=f"{logs_folder}/bs_tensorboard", n_steps=10_000
    )
    model.learn(total_timesteps=2_000_000, progress_bar=True, callback=callback)
    test_env = gym.make("GymV21Environment-v0", env_id="gym_anm:ANM6Easy-v0")
    mean_rwd, std_rwd, info = evaluate_policy_deprecated(model, test_env, n_eval_episodes=1)
    with open(csv_file, "a") as f:
        f.write(f"2_000_000,{datetime.now()},{mean_rwd},{std_rwd},{info['energy_loss']},{info['penalty']}\n")
    logger.info(f"Mean reward: {mean_rwd}, std: {std_rwd}")
    logger.info("Saving baseline model")
    model.save(f"{logs_folder}/model_baseline")
    logger.info("Saving baseline model")
    model.save(f"{logs_folder}/model_baseline")


def main():
    test_baseline("logs")


if __name__ == "__main__":
    main()
