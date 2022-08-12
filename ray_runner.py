"""The two-step game from QMIX: https://arxiv.org/pdf/1803.11485.pdf

Configurations you can try:
    - normal policy gradients (PG)
    - MADDPG
    - QMIX

See also: centralized_critic.py for centralized critic PPO on this game.
"""

import argparse
import logging
import os

import ray
from ray import tune
# from ray.tune import register_env
from gym.spaces import Discrete, Box, MultiDiscrete
from ray.rllib.algorithms.qmix import QMixConfig
from ray.rllib.algorithms.maddpg.maddpg import MADDPGConfig
from ray.rllib.env.multi_agent_env import ENV_STATE
from ray.rllib.examples.env.two_step_game import TwoStepGame
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.test_utils import check_learning_achieved
from gym.spaces import Discrete, Box, Tuple, Dict
import math

from ray.tune.registry import register_env
# import the pettingzoo environment
from final_env import g_env, env_obs, agent_obs, env_obs_one_hot, agent_obs_one_hot


logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run", type=str, default="QMIX", help="The RLlib-registered algorithm to use. Options: QMIX, MADDPG"
)

parser.add_argument(
    "--suffix", type=str
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
    "--as_test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument(
    "--stop_iters", type=int, default=1, help="Number of iterations to train."
)
parser.add_argument(
    "--steps_per_iters", type=int, default=2000, help="Number of timesteps to train."
)

parser.add_argument(
    "--local_mode",
    action="store_true",
    help="Init Ray in local mode for easier debugging.",
)

parser.add_argument(
    "--one_hot_encoding",
    action="store_true"
)

parser.add_argument(
    "--n_agents",
    type=int,
    default=100
)



if __name__ == "__main__":
    args = parser.parse_args()
    print('start')
    ray.init(num_cpus=args.num_cpus or None, local_mode=args.local_mode)
    print('ray start---------------------------------------------------')
    
    n_agents = args.n_agents

    register_env(
        "g_env",
        lambda config: g_env(config, args)
    )

    grouping = {
        "group_1": [i for i in range(n_agents)],
    }

    if args.one_hot_encoding:
        obs_space = Tuple(
            [
                Dict({"obs": agent_obs_one_hot(), ENV_STATE: env_obs_one_hot(n_agents)}) for i in range(n_agents)
            ]
        )
    else:
        obs_space = Tuple(
            [
                Dict({"obs": agent_obs(), ENV_STATE: env_obs(n_agents)}) for i in range(n_agents)
            ]
        )

    act_space = Tuple(
        [
            g_env.action_space for i in range(n_agents)
        ]
    )
    register_env(
        "grouped_g_env",
        lambda config: g_env(config, args).with_agent_groups(
            grouping, obs_space=obs_space, act_space=act_space
        )
    )

    
    if args.run == "MADDPG":

        policies = {f"policy_{i}": PolicySpec(None,
            observation_space=Box(0.0, 1.0, shape=agent_obs_one_hot().shape),
            action_space=g_env.action_space,
            config={'agent_id': i}
        ) for i in range(n_agents)}

        policy_ids = list(policies.keys())

        config =  (
            MADDPGConfig()
            .training(train_batch_size=64)
            .rollouts(
                num_rollout_workers=0,
                rollout_fragment_length=64
            )
            .environment(
                env="g_env",
                env_config={
                    "n_agents": n_agents,
                    "separate_state_space": False,
                    "actions_are_logits": True,
                    "one_hot_state_encoding": True,
                    "binary": False
                },
            )
            .multi_agent(
                policies=policies,
                policy_mapping_fn=(lambda i, episode, worker, **kwargs: policy_ids[i])
            )
            .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
        )
        config = config.to_dict()

    elif args.run == "QMIX":
        max_seq_len = 64
        config = (
            QMixConfig()
            .training(
                mixer=args.mixer,
                train_batch_size=64, model={"max_seq_len": max_seq_len})
            .rollouts(num_rollout_workers=0, rollout_fragment_length=64)
            .exploration(
                exploration_config={
                    "final_epsilon": 0.01,
                    "epsilon_timesteps": 1000
                }
            )
            .environment(
                env="grouped_g_env",
                env_config={
                    "n_agents": n_agents,
                    "separate_state_space": True,
                    "one_hot_state_encoding": args.one_hot_encoding,
                },
            )
            .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
        )
        config = config.to_dict()
    else:
        raise ValueError(f"{args.run} not implemented")

    config['min_sample_timesteps_per_iteration'] = args.steps_per_iters

    stop = {
        "timesteps_total": 3000,
        "training_iteration": 1
    }

    results = tune.run(args.run, stop=stop, config=config, verbose=2)

    if args.as_test:
        check_learning_achieved(results, args.stop_reward)

    ray.shutdown()