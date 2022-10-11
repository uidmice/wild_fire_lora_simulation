import sys
import time, argparse

import torch
import numpy as np

from algorithms.qmix  import QMIX_Agent
from algorithms.baselines import Base_Agent, Random_Agent, Heuristic_Agent
from train_env import g_env, test_env

import os

def parse_args():
    parser = argparse.ArgumentParser("RL experiments Wildfires")

    # algorithms
    parser.add_argument(
        "--run",
        choices=['QMIX', 'BASE', 'RANDOM', 'HEURISTIC', 'GNNQmix'],
        default="QMIX",
        help="The algorithm to use. Options: QMIX, BASE, RANDOM, HEURISTIC, GNNQmix"
    )

    # wildfire env
    parser.add_argument("--n-agents", type=int, default=200)
    parser.add_argument("--alternating-wind", type=int, default=10)
    parser.add_argument("--disable-random-source", dest='random_source', action='store_false')
    parser.add_argument("--spotting", action="store_true")
    parser.add_argument("--wind-step-size", type=int, default=120)
    parser.add_argument("--disable_single_reward", dest='single_reward', action='store_false')

    # environment
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--save_dir", type=str, default="results_new", help="model should be saved")
    parser.add_argument("--suffix", type=str)
    parser.add_argument('--enable-store', dest='store', action='store_true')
    parser.add_argument("--per_episode_max_len", type=int, default=25, help="maximum episode length")
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
    parser.add_argument("--learning_start_episode", type=int, default=45, help="learning start episode")
    parser.add_argument("--learning_fre", type=int, default=5, help="learning frequency")
    parser.add_argument("--tar_net_update_fre", type=int, default=4, help="learning rounds for update target net")
    parser.add_argument("--memory_size", type=int, default=100, help="number of data stored in the memory")
    parser.add_argument("--batch_size", type=int, default=40, help="number of episodes to optimize at the same time")
    parser.add_argument(
        "--mixer",
        choices=['QMIX', 'VDN', 'NONE'],
        default="QMIX"
    )
    # qmix
    parser.add_argument("--q_net_out", type=list, default=[32, 32], help="size of layers feature in q_net")
    parser.add_argument("--mix_net_out", type=list, default=[32, 1], help="size of layers feature in q_net")
    parser.add_argument("--q_net_hidden_size", type=list, default=32, help="size of hidden feature in q_net")
    parser.add_argument("--shape_hyper_b2_hidden", type=int, default=32, help="size of hidden feature in q_net")

    # GNN
    parser.add_argument("--gnn_hidden_dim", type=list, default=32)
    parser.add_argument("--gnn_out_dim", type=list, default=16)

    # random
    parser.add_argument(
        "--random-prob", type=float, default=0.05, help="The probability of sending of random policies (default: 0.5)"
    )
    # checkpointing
    parser.add_argument("--fre4save_model", type=int, default=12)
    parser.add_argument("--start_save_model", type=int, default=20, help="saving the model")

    return parser.parse_args()


def train(env, args, agent):
    """ step1: init the env and par """

    """ step2: init the QMIX agent """
    # qmix_agent = QMIX_Agent(shape_obs, shape_state, num_agents, num_actions_set, args)
    agent.init_trainers()

    """ step3: interact with the env and learn """
    step_cnt = 0
    for epi_cnt in range(args.max_episode):

        # init the episode data
        env.reset()
        episode_reward = np.zeros(args.n_agents)
        actions_last = env.last_action
        agent.memory.create_new_episode()
        hidden_last = np.zeros((args.n_agents, args.q_net_hidden_size))
        fb = [0 for _ in range(args.n_agents)]

        for epi_step_cnt in range(args.per_episode_max_len):
            step_cnt += 1 # update the cnt every time

            # get obs state for select action
            state = env.get_state()
            obs = env.get_obs()
            avail_actions = env.get_avail_actions()

            # interact with the env and get new state obs
            actions, hidden = agent.select_actions(avail_actions, obs, actions_last, hidden_last, args)
            rewards, dones, done, fb = env.step(actions)

            if epi_step_cnt == args.per_episode_max_len-1:
                done = True # max len of episode
                dones[:] = 1
            state_new = env.get_state()

            obs_new = env.get_obs()
            avail_actions_new = env.get_avail_actions()
            actions_now_onehot = env.last_action # the env do the things for us

            # update the date and save experience to memory

            # concatenate the obs and actions_last for speed up the train
            agent.save_memory(np.concatenate([obs, actions_last], axis=-1), state, actions.reshape(1, -1), avail_actions_new, np.concatenate([obs_new, actions_now_onehot], axis=-1),
                     state_new, rewards, dones, done)

            actions_last = env.last_action
            hidden_last = hidden

            # if done, end the episode
            episode_reward += rewards
            if done:
                break
        # agents learn

        loss = agent.learn(epi_cnt, args, env.logdir)
        print(' ' * 80, 'loss is', loss, end='\r')

if __name__ == '__main__':
    args = parse_args()
    # args.n_agents = 20
    # args.learning_start_episode = 4
    # args.per_episode_max_len = 2
    # args.learning_fre = 2
    # args.mixer = 'QMIX'
    # args.run= "GNNQmix"
    # args.fre4save_model = 3

    if args.device == 'cuda' and not torch.cuda.is_available():
        print('CUDA is not available. Use CPU instead.')
        args.device = 'cpu'

    env = g_env(args)
    print(args)


    args.device = torch.device(args.device)

    if args.run == 'BASE':
        agent = Base_Agent()
    elif args.run == "QMIX" or args.run == "GNNQmix":
        agent = QMIX_Agent(env.obs_dim, env.state_dim, env.n_sensors, [2], args)
    elif args.run == 'RANDOM':
        agent = Random_Agent([1-args.random_prob, args.random_prob], env.num_actions, env.n_sensors)
    elif args.run == 'HEURISTIC':
        agent = Heuristic_Agent()

    train(env, args, agent)
