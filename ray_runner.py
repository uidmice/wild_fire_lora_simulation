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
import numpy as np


logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run", type=str, default="QMIX", help="The algorithm to use. Options: QMIX, MADDPG, RANDOM, HEURISTIC"
)

parser.add_argument(
    "--random-prob", type=float, default=0.1, help="The probability of sending of random policies (default: 0.5)"
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
    "--as-test",
    action="store_true"
)

parser.add_argument(
    "--test",
    action="store_true"
)

parser.add_argument(
    "--stop-iters", type=int, default=1, help="Number of iterations to train."
)
parser.add_argument(
    "--steps-per-iters", type=int, default=2500, help="Number of timesteps to train."
)

parser.add_argument(
    "--steps-per-episodes", type=int, default=50, help="Number of timesteps to train."
)

parser.add_argument(
    "--step-size", type=int, default=30, help="Number of timesteps to train."
)


parser.add_argument(
    "--local-mode",
    action="store_true",
    help="Init Ray in local mode for easier debugging.",
)

parser.add_argument(
    "--one-hot-encoding",
    action="store_true"
)

parser.add_argument(
    "--spotting",
    action="store_true"
)

parser.add_argument(
    "--disable-random-source",
    dest='random_source',
    action='store_false'
)

parser.add_argument(
    "--n-agents",
    type=int,
    default=100
)



if __name__ == "__main__":
    args = parser.parse_args()
    print('start')
    n_agents = args.n_agents

    register_env(
        "g_env",
        lambda config: g_env(config, args)
    )


    if args.as_test:
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from matplotlib.animation import FuncAnimation
        import datetime

        p = args.random_prob
        env = g_env(dict(), args)
        for i in range(2):
            if i==1:
                env.spotting = True
            env.reset()
            real = [env.environment.get_on_fire(k) for k in range(env.step_size, env.T+1, env.step_size)]
            pre = []
            points = []
            update_points = []

            for j in range(env.T // env.step_size + 1):
                action_dict = {i: np.random.choice([0, 1], p=[1 - p, p]) for i in range(n_agents)}
                obs, rewards, dones, infos = env.step(action_dict)
                pre.append(infos['predict'])
                points.append([[env.col_idx[i], env.row_idx[i]] for i in infos['sent']])
                update_points.append([[env.col_idx[i], env.row_idx[i]] for i in infos['received']])

            a = np.array(real)
            b = np.array(pre)

            fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)


            def animate(i):
                fig.suptitle(f'$\Delta t={env.step_size}, T = {(i+1) * env.step_size}$', fontsize=16)
                ax[0].clear()
                ax[1].clear()
                ax[0].imshow(a[i])
                ax[1].imshow(b[i])
                ax[1].scatter([p[0] for p in points[i]], [p[1] for p in points[i]], color='red')
                ax[1].scatter([p[0] for p in update_points[i]], [p[1] for p in update_points[i]], color='green')

                e = np.sum(np.absolute(a[i] - b[i]))
                ax[0].set_title('Propogation')
                ax[1].set_title('Prediction, e=%d' % (e))
                plt.tight_layout()


            writer = animation.writers['ffmpeg']
            writer = writer(fps=2, metadata=dict(artist='Me'), bitrate=900)
            ani = FuncAnimation(fig, animate, len(a), interval=1000)
            ani.save(f"results/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.mp4", writer=writer)

    
    elif args.run == "MADDPG":
        ray.init(num_cpus=args.num_cpus or None, local_mode=args.local_mode, include_dashboard=False)
        print('ray start MADDPG ---------------------------------------------------')

        if args.one_hot_encoding:
            policies = {f"policy_{i}": PolicySpec(None,
                observation_space=Box(0.0, 1.0, shape=agent_obs_one_hot().shape),
                action_space=g_env.action_space,
                config={'agent_id': i}
            ) for i in range(n_agents)}
        else:
            policies = {f"policy_{i}": PolicySpec(None,
                observation_space=Box(0.0, 1.0, shape=agent_obs().shape),
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
                    "one_hot_state_encoding": args.one_hot_encoding,
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

        config['min_sample_timesteps_per_iteration'] = args.steps_per_iters

        stop = {
            "timesteps_total": 3000,
            "training_iteration": args.stop_iters
        }

        results = tune.run(args.run, stop=stop, config=config, verbose=2)

        ray.shutdown()



    elif args.run == "QMIX":
        ray.init(num_cpus=args.num_cpus or None, local_mode=args.local_mode, include_dashboard=False)
        print('ray start QMIX ---------------------------------------------------')

        grouping = {
            "group_1": [i for i in range(n_agents)],
        }

        if args.one_hot_encoding:
            obs_space = Tuple([Dict({"obs": agent_obs_one_hot(), ENV_STATE: env_obs_one_hot(n_agents)}) for i in range(n_agents)])
        else:
            obs_space = Tuple([Dict({"obs": agent_obs(), ENV_STATE: env_obs(n_agents)}) for i in range(n_agents)])

        act_space = Tuple([g_env.action_space for i in range(n_agents)])

        register_env(
            "grouped_g_env",
            lambda config: g_env(config, args).with_agent_groups(
                grouping, obs_space=obs_space, act_space=act_space
            )
        )

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

        config['min_sample_timesteps_per_iteration'] = args.steps_per_iters

        stop = {
            "timesteps_total": 3000,
            "training_iteration": args.stop_iters
        }

        results = tune.run(args.run, stop=stop, config=config, verbose=2)

        ray.shutdown()

    elif args.run == "RANDOM":

        p = args.random_prob

        env = g_env(dict(), args)
        steps_per_episode = env.T // env.step_size + 1

        for iter in range(args.stop_iters):
            for ep in range(math.ceil(args.steps_per_iters/steps_per_episode)):
                env.reset()
                for i in range(steps_per_episode):
                    action_dict = {i: np.random.choice([0,1], p=[1-p, p]) for i in range(n_agents)}
                    obs, rewards, dones, infos = env.step(action_dict)
                    if dones['__all__']:
                        break

    elif args.run == 'HEURISTIC':

        env = g_env(dict(), args)
        steps_per_episode = env.T // env.step_size + 1

        for iter in range(args.stop_iters):
            for ep in range(math.ceil(args.steps_per_iters/steps_per_episode)):
                env.reset()
                action = [0 for _ in range(n_agents)]
                for i in range(steps_per_episode):
                    fb, done = env.step_heuristic(action)
                    print(fb)
                    action = [int(a==2) for a in fb]
                    if done:
                        break


    else:
        raise ValueError(f"{args.run} not implemented")

