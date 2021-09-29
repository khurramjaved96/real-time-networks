import argparse
import glob
import importlib
import os
import sys

sys.path.append("rl-baselines3-zoo")

import numpy as np
import torch as th
import yaml
from stable_baselines3.common.utils import set_random_seed

import utils.import_envs  # noqa: F401 pylint: disable=unused-import
from utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams
from utils.exp_manager import ExperimentManager
from utils.utils import StoreDict
import matplotlib.pyplot as plt


class BaselinesExpert:
    def __init__(
        self,
        seed=0,
        env_id="PongNoFrameskip-v4",
        algo="dqn",
        folder="rl-baselines3-zoo/rl-trained-agents",
        no_render=True,
        deterministic=True,
    ):
        log_path = os.path.join(folder, algo, f"{env_id}_1")
        assert os.path.isdir(log_path), f"The {log_path} folder was not found"

        found = False
        for ext in ["zip"]:
            model_path = os.path.join(log_path, f"{env_id}.{ext}")
            found = os.path.isfile(model_path)
            if found:
                break

        if not found:
            raise ValueError(f"No model found for {algo} on {env_id}, path: {model_path}")
        print(f"Loading {model_path}")

        # Off-policy algorithm only support one env for now
        off_policy_algos = ["qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc"]
        set_random_seed(seed)
        is_atari = ExperimentManager.is_atari(env_id)
        stats_path = os.path.join(log_path, env_id)
        hyperparams, stats_path = get_saved_hyperparams(
            stats_path, norm_reward=False, test_mode=True
        )

        # load env_kwargs if existing
        env_kwargs = {}
        args_path = os.path.join(log_path, env_id, "args.yml")
        if os.path.isfile(args_path):
            with open(args_path, "r") as f:
                loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)  # pytype: disable=module-attr
                if loaded_args["env_kwargs"] is not None:
                    env_kwargs = loaded_args["env_kwargs"]

        env = create_test_env(
            env_id,
            n_envs=1,
            stats_path=stats_path,
            seed=seed,
            log_dir=None,
            should_render=not no_render,
            hyperparams=hyperparams,
            env_kwargs=env_kwargs,
        )

        kwargs = dict(seed=seed)
        if algo in off_policy_algos:
            # Dummy buffer size as we don't need memory to enjoy the trained agent
            kwargs.update(dict(buffer_size=1))

        # Check if we are running python 3.8+
        # we need to patch saved model under python 3.6/3.7 to load them
        newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8

        custom_objects = {}
        if newer_python_version:
            custom_objects = {
                "learning_rate": 0.0,
                "lr_schedule": lambda _: 0.0,
                "clip_range": lambda _: 0.0,
            }

        model = ALGOS[algo].load(model_path, env=env, custom_objects=custom_objects, **kwargs)

        self.model = model
        self.env = env
        self.deterministic = deterministic
        self.state = None

    def predict(self, obs):
        action, self.state = self.model.predict(obs, state=self.state, deterministic=self.deterministic)
        return action


if __name__ == "__main__":
	expert_policy = BaselinesExpert()
	env = expert_policy.env
	obs = env.reset()

	import random
	episode_reward = 0
	for timestep in range(20000):
		action = expert_policy.predict(obs)
		#action = [env.action_space.sample()]
		if timestep > 30:
			from IPython import embed; embed()
		obs, reward, done, infos = env.step(action)
		if True:
			env.render("human")

		episode_reward += reward[0]

		# For atari the return reward is not the atari score
		# so we have to get it from the infos dict
		if True and infos is not None:
			episode_infos = infos[0].get("episode")
			if episode_infos is not None:
				print(f"Atari Episode Score: {episode_infos['r']:.2f}")
				print("Atari Episode Length", episode_infos["l"])
				expert_policy.state = None
	env.close()
