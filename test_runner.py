"""The two-step game from QMIX: https://arxiv.org/pdf/1803.11485.pdf

Configurations you can try:
    - normal policy gradients (PG)
    - MADDPG
    - QMIX

See also: centralized_critic.py for centralized critic PPO on this game.
"""

import argparse
from gym.spaces import Dict, Discrete, Tuple, MultiDiscrete
import logging
import os

import ray
from ray import tune
from ray.tune import register_env
from ray.rllib.algorithms.qmix import QMixConfig
from ray.rllib.env.multi_agent_env import ENV_STATE
from ray.rllib.examples.env.two_step_game import TwoStepGame
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.test_utils import check_learning_achieved
from gym.spaces import Discrete, Box
import math

from ray.tune.registry import register_env
# import the pettingzoo environment
from test_env import g_env
# import rllib pettingzoo interface
from ray.rllib.env import PettingZooEnv
# define how to make the environment. This way takes an optional environment config, num_floors

# env_creator = lambda config: g_env(n_agent=config.get("n_agents", 10))
# register that way to make the environment under an rllib name

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run", type=str, default="QMIX", help="The RLlib-registered algorithm to use."
)
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "tfe", "torch"],
    default="torch",
    help="The DL framework specifier.",
)
parser.add_argument("--num-cpus", type=int, default=0)
parser.add_argument(
    "--mixer",
    type=str,
    default="qmix",
    choices=["qmix", "vdn", "none"],
    help="The mixer model to use.",
)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument(
    "--stop-iters", type=int, default=5, help="Number of iterations to train."
)
parser.add_argument(
    "--stop-timesteps", type=int, default=40, help="Number of timesteps to train."
)
parser.add_argument(
    "--stop-reward", type=float, default=1000, help="Reward at which we stop training."
)
parser.add_argument(
    "--local-mode",
    action="store_true",
    help="Init Ray in local mode for easier debugging.",
)


def obs_create(n_agents):
    return Dict({"obs": Discrete(2), ENV_STATE: MultiDiscrete([2 for i in range(n_agents)]),})

if __name__ == "__main__":
    args = parser.parse_args()
    print('start')
    ray.init(num_cpus=args.num_cpus or None, local_mode=args.local_mode)
    print('ray start---------------------------------------------------')
    if args.run == "contrib/MADDPG":
        logger.warning(
            "`contrib/MADDPG` is not longer a valid algorithm descriptor! "
            "Use `MADDPG` instead."
        )
        args.run = "MADDPG"
    
    n_agents = 5
    grouping = {
        "group_1": [i for i in range(n_agents)],
    }
    obs_space = Tuple(
        [
            obs_create(n_agents) for i in range(n_agents)
        ]
    )
    act_space = Tuple(
        [
            g_env.action_space for i in range(n_agents)
        ]
    )
    register_env(
        "g_env",
        lambda config: g_env(config).with_agent_groups(
            grouping, obs_space=obs_space, act_space=act_space
        ),
    )
    
    if args.run == "MADDPG":
        obs_space = Discrete(6)
        act_space = TwoStepGame.action_space
        config = {
            "env": TwoStepGame,
            "env_config": {
                "actions_are_logits": True,
            },
            "replay_buffer_config": {"learning_starts": 100},
            "multiagent": {
                "policies": {
                    "pol1": PolicySpec(
                        observation_space=obs_space,
                        action_space=act_space,
                        config={"agent_id": 0},
                    ),
                    "pol2": PolicySpec(
                        observation_space=obs_space,
                        action_space=act_space,
                        config={"agent_id": 1},
                    ),
                },
                "policy_mapping_fn": (lambda aid, **kwargs: "pol2" if aid else "pol1"),
            },
            "framework": args.framework,
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        }
    elif args.run == "QMIX":
        max_seq_len = 40
        config = (
            QMixConfig()
            .training(mixer=args.mixer, train_batch_size=40, model={"max_seq_len": max_seq_len})
            .rollouts(num_rollout_workers=0, rollout_fragment_length=40)
            .exploration(
                exploration_config={
                    "final_epsilon": 0.0,
                }
            )
            .environment(
                env="g_env",
                env_config={
                    "n_agents": 5,
                    "separate_state_space": True,
                    "one_hot_state_encoding": True,
                },
            )
            .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
        )
        config = config.to_dict()
    else:
        config = {
            "env": TwoStepGame,
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "framework": args.framework,
        }

    stop = {
        "timesteps_total": 70000,
    }

    results = tune.run(args.run, stop=stop, config=config, verbose=2)

    if args.as_test:
        check_learning_achieved(results, args.stop_reward)

    ray.shutdown()