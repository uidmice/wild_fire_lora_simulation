import sys
import pickle, argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import datetime

from algorithms.qmix  import QMIX_Agent
from algorithms.baselines import Base_Agent, Random_Agent, Heuristic_Agent, Heuristic_Agent2, Heuristic_Agent3
from train_env import g_env, test_env
from framework.GRASS import SOURCE_NAME

import os

def parse_args():
    parser = argparse.ArgumentParser("RL experiments Wildfires")

    # algorithms
    parser.add_argument(
        "--run",
        choices=['QMIX', 'BASE', 'RANDOM', 'HEURISTIC1', 'HEURISTIC2', 'HEURISTIC3', 'HEURISTIC4', 'GNNQmix'],
        default="QMIX",
        help="The algorithm to use. Options: QMIX, BASE, RANDOM, HEURISTIC, GNNQmix"
    )

    # wildfire env
    parser.add_argument("--n-agents", type=int, default=200)
    parser.add_argument("--alternating-wind", type=int, default=1)
    parser.add_argument("--disable-random-source", dest='random_source', action='store_false')
    parser.add_argument("--spotting", action="store_true")
    parser.add_argument("--wind-step-size", type=int, default=120)
    parser.add_argument("--disable-single-reward", dest='single_reward', action='store_false')
    parser.add_argument('--disable-prediction', action='store_true')
    parser.add_argument("--synchronous", action='store_true')
    parser.add_argument("--limit-observation", action='store_true')
    parser.add_argument("--uneven-wind", action='store_true')
    parser.add_argument("--mapset", default='grass')
    parser.add_argument('--unknown-source', action='store_true')
    parser.add_argument('--simplified-state', action='store_true')
    parser.add_argument("--reward-weights", type=list, default=[1, 0])
    parser.add_argument(
        "--data-reward",
        choices=['acc', 'track_acc', 'correction_eff'],
        default="acc"
    )
    parser.add_argument(
        "--mixer",
        choices=['QMIX', 'VDN', 'NONE', 'GraphMix'],
        default="QMIX"
    )

    # random
    parser.add_argument(
        "--random-prob", type=float, default=0.05, help="The probability of sending of random policies (default: 0.5)"
    )



    # environment
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--visualize", action='store_true')
    parser.add_argument("--save_dir", type=str, default="results_new/new_env_120_2ch", help="model should be saved")
    parser.add_argument("--suffix", type=str)
    parser.add_argument("--per_episode_max_len", type=int, default=30, help="maximum episode length")
    parser.add_argument("--max_episode", type=int, default=1000, help="maximum episode length")
    parser.add_argument("--num_epi4evaluation", type=int, default=10, help="the num for evaluation")
    parser.add_argument("--num_epi4enjoy", type=int, default=40, help="the num for evaluation")
    parser.add_argument("--fre_epi4evaluation", type=int, default=50, help="the num for evaluation")
    # parser.add_argument("--num-adversaries", type=int, default=1, help="number of adversaries")

    # core training parameters
    parser.add_argument("--device", choices=['cpu', 'cuda'], default='cpu', help="torch device ")
    parser.add_argument("--tao", type=int, default=0.01, help="how depth we exchange the par of the nn")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate for adam optimizer")
    parser.add_argument("--gamma", type=float, default=1, help="discount factor")
    parser.add_argument("--anneal_par", type=float, default=0.002, help="learning frequency")
    parser.add_argument("--epsilon", type=float, default=1.0, help="the init par for e-greedy")
    parser.add_argument("--max_grad_norm", type=float, default=6, help="max gradient norm for clip")
    parser.add_argument("--learning_start_episode", type=int, default=50, help="learning start episode")
    parser.add_argument("--learning_fre", type=int, default=10, help="learning frequency")
    parser.add_argument("--tar_net_update_fre", type=int, default=3, help="learning rounds for update target net")
    parser.add_argument("--memory_size", type=int, default=100, help="number of data stored in the memory")
    parser.add_argument("--batch_size", type=int, default=20, help="number of episodes to optimize at the same time")

    # qmix
    parser.add_argument("--q_net_out", type=list, default=[32, 32], help="size of layers feature in q_net")
    parser.add_argument("--mix_net_out", type=list, default=[32, 1], help="size of layers feature in q_net")
    parser.add_argument("--q_net_hidden_size", type=list, default=32, help="size of hidden feature in q_net")
    parser.add_argument("--shape_hyper_b2_hidden", type=int, default=32, help="size of hidden feature in q_net")
    parser.add_argument("--double_q", action='store_true')

    # GNN
    parser.add_argument("--gnn_hidden_dim", type=list, default=32)
    parser.add_argument("--gnn_out_dim", type=list, default=24)
    parser.add_argument("--lambda_local", type=float, default=1)


    # checkpointing
    parser.add_argument("--fre4save_model", type=int, default=6)
    parser.add_argument("--start_save_model", type=int, default=20, help="saving the model")

    return parser.parse_args()


def train(env, args, agent):
    """ step1: init the env and par """

    """ step2: init the QMIX agent """
    agent.init_trainers()

    """ step3: interact with the env and learn """
    step_cnt = 0
    if args.visualize:
        args.max_episode = 2

    for epi_cnt in range(args.max_episode):

        # init the episode data
        env.reset()
        episode_reward = np.zeros((args.n_agents, 2))
        actions_last = env.last_action
        agent.memory.create_new_episode()
        hidden_last = np.zeros((args.n_agents, args.q_net_hidden_size))
        fb = env.fb
        if args.visualize:
            real = []
            pre = []
            disregard = []
            points = []
            update_points = []
            acc = []


        for epi_step_cnt in range(args.per_episode_max_len):
            step_cnt += 1 # update the cnt every time

            # get obs state for select action
            state = env.get_state()
            obs = env.get_obs()
            avail_actions = env.get_avail_actions()

            # interact with the env and get new state obs
            actions, hidden = agent.select_actions(avail_actions, obs, actions_last, hidden_last, args=args,  eval_flag=False, fb=fb, burning=env.burning)
            rewards, dones, done, fb, infos = env.step(actions)

            if epi_step_cnt == args.per_episode_max_len-1:
                done = True # max len of episode
                dones[:] = 1


            state_new = env.get_state()

            obs_new = env.get_obs()
            avail_actions_new = env.get_avail_actions()
            actions_now_onehot = env.last_action # the env do the things for us

            # update the date and save experience to memory

            # concatenate the obs and actions_last for speed up the train
            if epi_step_cnt > 0:
                agent.save_memory(np.concatenate([obs, actions_last], axis=-1), state, actions.reshape(1, -1), avail_actions_new, np.concatenate([obs_new, actions_now_onehot], axis=-1),
                     state_new, rewards, dones, done)

            actions_last = env.last_action
            hidden_last = hidden

            # if done, end the episode
            episode_reward += rewards

            if args.visualize:
                real.append(infos['burning'])
                pre.append(infos['predict'])
                disregard.append(infos['disregard'])
                points.append([[env.col_idx[i], env.row_idx[i]] for i in infos['sent']])
                update_points.append([[env.col_idx[i], env.row_idx[i]] for i in infos['received']])
                acc.append(rewards[0,0])


            if done:
                # predata = json.load(open(env.logging, 'r'))
                # predata.append(env.records)
                # json.dump(predata, open(env.logging, 'w'))
                pickle.dump(env.records, open(env.logging, 'ab'))
                print(f' # of update: {env.records["received"]}')
                print(f' update time: {env.records["update_time"]}')
                print(f' predict time: {env.records["predict_time"]}')
                break
        # agents learn

        if args.visualize:
            a = np.array(real)
            b = np.array(pre)
            dis = np.array(disregard)

            fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)

            def animate(i):
                fig.suptitle(f'$\Delta t={env.step_size}, T = {(i + 1) * env.step_size}$', fontsize=16)
                ax[0].clear()
                ax[1].clear()
                ax[0].imshow(a[i] + dis[i])
                ax[1].imshow(b[i] + dis[i])
                ax[1].scatter([p[0] for p in points[i]], [p[1] for p in points[i]], color='red')
                ax[1].scatter([p[0] for p in update_points[i]], [p[1] for p in update_points[i]], color='green')

                e = acc[i]
                ax[0].set_title('Propogation')
                ax[1].set_title(f'Track acc ={e:.4f}')
                plt.tight_layout()

            writer = animation.writers['ffmpeg']
            writer = writer(fps=2, metadata=dict(artist='Me'), bitrate=900)
            ani = FuncAnimation(fig, animate, len(b), interval=1000)
            ani.save(f"test_log/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args.run}.mp4", writer=writer)

        loss = agent.learn(epi_cnt, args, env.logdir)
        print(' ' * 80, 'loss is', loss, end='\r')


def run_baselines(env, args, agent):
    args.max_episode = args.n_agents
    if args.visualize:
        args.max_episode = 2

    for epi_cnt in range(args.max_episode):
        env.reset(epi_cnt)
        fb = env.fb
        if args.visualize:
            real = []
            pre = []
            disregard = []
            points = []
            update_points = []
            acc = []

        for epi_step_cnt in range(args.per_episode_max_len):
            # get obs state for select action
            avail_actions = env.get_avail_actions()

            # interact with the env and get new state obs
            actions, hidden = agent.select_actions(avail_actions, args=args, fb=fb, burning=env.burning)
            rewards, dones, done, fb, infos = env.step(actions)

            if epi_step_cnt == args.per_episode_max_len-1:
                done = True # max len of episode
                dones[:] = 1

            if args.visualize:
                real.append(infos['burning'])
                pre.append(infos['predict'])
                disregard.append(infos['disregard'])
                points.append([[env.col_idx[i], env.row_idx[i]] for i in infos['sent']])
                update_points.append([[env.col_idx[i], env.row_idx[i]] for i in infos['received']])
                acc.append(rewards[0,0])

            if done:
                pickle.dump(env.records, open(env.logging, 'ab'))
                print(f' # of update: {env.records["received"]}')
                print(f' update time: {env.records["update_time"]}')
                print(f' predict time: {env.records["predict_time"]}')
                break
        if args.visualize:
            a = np.array(real)
            b = np.array(pre)
            dis = np.array(disregard)

            fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)

            def animate(i):
                fig.suptitle(f'$\Delta t={env.step_size}, T = {(i + 1) * env.step_size}$', fontsize=16)
                ax[0].clear()
                ax[1].clear()
                ax[0].imshow(a[i] + dis[i])
                ax[1].imshow(b[i] + dis[i])
                ax[1].scatter([p[0] for p in points[i]], [p[1] for p in points[i]], color='red')
                ax[1].scatter([p[0] for p in update_points[i]], [p[1] for p in update_points[i]], color='green')

                e = acc[i]
                ax[0].set_title('Propogation')
                ax[1].set_title(f'Track acc ={e:.4f}')
                plt.tight_layout()

            writer = animation.writers['ffmpeg']
            writer = writer(fps=2, metadata=dict(artist='Me'), bitrate=900)
            ani = FuncAnimation(fig, animate, len(b), interval=1000)
            ani.save(f"test_log/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args.run}.mp4", writer=writer)

if __name__ == '__main__':
    args = parse_args()
    # args.n_agents = 5
    # args.visualize = True
    # args.wind_step_size = 30
    args.learning_start_episode = 1
    # args.per_episode_max_len = 3
    # args.learning_fre = 2
    # args.mixer = 'GraphMix'
    # args.double_q = True
    # args.run= "HEURISTIC2"
    # args.batch_size = 2
    # args.fre4save_model = 3

    if args.device == 'cuda' and not torch.cuda.is_available():
        print('CUDA is not available. Use CPU instead.')
        args.device = 'cpu'

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    env = g_env(args)
    # env = test_env(args)
    print(args)


    args.device = torch.device(args.device)
    agent = None
    if args.run == 'BASE':
        agent = Base_Agent()
        run_baselines(env, args, agent)

    elif args.run == "QMIX" or args.run == "GNNQmix":
        agent = QMIX_Agent(env.obs_dim, env.state_dim, env.n_sensors, [2], args)
        train(env, args, agent)

    elif args.run == 'RANDOM':
        agent = Random_Agent([1-args.random_prob, args.random_prob], env.num_actions, env.n_sensors)
        train(env, args, agent)

    elif 'HEURISTIC' in args.run:
        agent = Heuristic_Agent()
        train(env, args, agent)


