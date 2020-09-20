import gym
import numpy as np
import matplotlib.pyplot as plt
import random
import argparse
import os
import time

from qlearning import QLearningAgent

np.random.seed(0)
random.seed(0)

env = gym.make('Taxi-v3')
env.seed(0)
max_episode_steps = 100
env._max_episode_steps = max_episode_steps

class TaxiEnv():
    def __init__(self, env):
        self.env = env

    def getLegalActions(self, state):
        return list(range(int(env.action_space.n)))

    def getStateSpaceSize(self):
        return int(self.env.observation_space.n)

    def getActionSpaceSize(self):
        return int(self.env.action_space.n)

def decode(s):
        out = []
        out.append(s % 4)
        s = s // 4
        out.append(s % 5)
        s = s // 5
        out.append(s % 5)
        s = s // 5
        out.append(s)
        assert 0 <= s < 5
        return tuple(reversed(out))

def pos_id2char(id):
    if   id == 0:
        return "R"
    elif id == 1:
        return "G"
    elif id == 2:
        return "Y"
    elif id == 3:
        return "B"
    elif id == 4:
        return "Taxi"

def action_id2char(id):
    if   id == 0:
        return "south"
    elif id == 1:
        return "north"
    elif id == 2:
        return "east"
    elif id == 3:
        return "west"
    elif id == 4:
        return "pickup"
    elif id == 5:
        return "dropoff"

def report_state(s):
    taxi_x, taxi_y, pass_loc, dest_loc = decode(s)
    print("Taxi position: [{}, {}]".format(taxi_x, taxi_y))
    print("Passenger position: {}".format(pos_id2char(pass_loc)))
    print("Delivery position: {}".format(pos_id2char(dest_loc)))


def plotHistory(steps_per_episode, w=10, title='', plt_show=True):
    plt.figure()
    plt.plot([np.mean(steps_per_episode[i:i+w]) for i in
        range(0,len(steps_per_episode),w)])
    plt.xlabel("Episodes (average by each {} episodes)".format(w))
    plt.ylabel("Average Steps")
    plt.title(title)
    if plt_show:
        plt.show()

def checkStep(args):
    cmd = input()
    if cmd == 's':
        args.silent = True
    elif cmd == 'p':
        os.system('clear')
    elif cmd == '\'':
        print(bytes.fromhex('512d4c6f7563757261').decode('utf-8'))
    return cmd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-tr", "--train_episodes", default=1000,
                        help="maximum of trainning episodes",
                        required=False, dest="train_episodes", type=int)
    parser.add_argument("-vl", "--val_episodes", default=100,
                        help="total of episodes simulated without training",
                        required=False, dest="val_episodes", type=int)
    parser.add_argument("-e", "--epsilon", default=0.01, required=False,
                        dest="epsilon", type=float)
    parser.add_argument("-a", "--alpha", default=0.8, required=False,
                        dest="alpha", type=float)
    parser.add_argument("-g", "--gamma", default=0.4, required=False,
                        dest="gamma", type=float)
    parser.add_argument("-p", "--plot", dest="plot", default=False,
                        action='store_true')
    parser.add_argument('-v','--verbose', dest="verbose", action='store_true',
                        default=False)
    parser.add_argument('-s','--silent', dest="silent", action='store_true',
                        default=False)
    args = parser.parse_args()

    agent = QLearningAgent(TaxiEnv(env), epsilon=args.epsilon, alpha=args.alpha,
                                         gamma=args.gamma)

    total_episodes = args.train_episodes+args.val_episodes + 1
    episode_start_val = args.train_episodes + 1
    train_timesteps_list = []
    val_timesteps_list = []

    cmd = None
    for i_episode in range(1, total_episodes):
        state = env.reset()
        done = 0
        t = 0
        reward = 0

        if i_episode >= episode_start_val:
            agent.epsilon = 0.0

        if i_episode >= episode_start_val and not args.silent:
            print("Testing: use return key to skip each step, 'p' to play"+
                  "all steps, 's' to stop printing or 'q' to quit the program")
            cmd = checkStep(args)
            if cmd == 'q':
                break

        while not done:
            if i_episode >= episode_start_val and not args.silent:
                if cmd == 'p':
                    time.sleep(.1)
                    os.system('clear')
                elif t != 0:
                    print("Testing: use return key to skip each step, 'p'"+
                           "to play all steps, 's' to stop printing")
                    cmd = checkStep(args)
                env.render()
                print("Last Reward: {}".format(reward))
                report_state(state)

            action = agent.getAction(state)

            last_state = state
            state, reward, done, info = env.step(action)
            taxi_x, taxi_y, pass_loc, dest_loc = decode(state)

            if i_episode < episode_start_val:
                agent.update(last_state, action, reward, state)
            t += 1
        if i_episode >= episode_start_val:
            val_timesteps_list.append(t)
        else:
            train_timesteps_list.append(t)

        # Prevent flood log information when training
        if i_episode >= episode_start_val:
            if not args.silent:
                print("Episode {} stops after {} timesteps".format(
                    i_episode, t))
                if cmd == 'p':
                    time.sleep(1)
        elif args.verbose:
            print("\rEpisode {} stops after {} timesteps".format(
                i_episode, t), end='\r')
    env.close()
    print("")
    if args.plot:
        plotHistory(train_timesteps_list, title='Train - Episodes {} - Epsilon {}'.format(args.train_episodes, args.epsilon),
                plt_show=False)
        plotHistory(val_timesteps_list, 1, title='Test - Episodes {} - Epsilon {}'.format(args.val_episodes, args.epsilon))
