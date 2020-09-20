import numpy as np
import pickle
import argparse
import gym
from qlearning_aprox import QLearningAgent

np.random.seed(0)

env = gym.make('LunarLander-v2')
env.seed(0)
# max_episode_steps = 400
# env._max_episode_steps = max_episode_steps

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", help="visualzation mode", choices=["view", "train"],
                        dest="mode", required=False, default='train')
    parser.add_argument("-ms", "--max_episodes", help="maximum of episodes simulated",
                        required=False, dest="max_episodes", type=int,
                        default='100')
    parser.add_argument("-e", "--episode", help="episode to view or start training from", default=0,
                        required=False, dest="episode", type=int)
    parser.add_argument('-r','--render', dest="render", action='store_true',
                        default=False)

    parser.add_argument('--save_episodes', nargs='+', required=False, type=int,
                        default=[], help='Episodes where agent will be saved')

    parser.add_argument('-l','--list', dest="list", action='store_true',
                        default=False, help="list saved episodes")
    args = parser.parse_args()


    agent = QLearningAgent(env, possible_actions=4, alpha=0.001, epsilon=0.01,
                      gamma=0.999)
    if(args.list):
        agent.load_snapshots_from_file()
        print(agent.snapshots.keys())
        env.close()
        exit()

    use_snapshots = (args.mode == "view")

    render_simulation = args.render

    if (use_snapshots or args.episode != 0):
        agent.load_snapshots_from_file()
        if not args.episode in agent.snapshots:
            print("Snapshot not found. Available episode: "+
                    "{}".format(list(agent.snapshots.keys())))
            exit()

        for x in agent.snapshots[args.episode]:
            agent.networks[x] = pickle.loads(agent.snapshots[args.episode][x])

    total_episodes = args.max_episodes
    if args.mode == "view":
        total_episodes += args.episode
        episode_start_test = 0
    else:
        episode_start_test = total_episodes+1

    interval_render = total_episodes+1

    episode_to_save = args.save_episodes
    if(len(episode_to_save) == 0):
        episode_to_save = [total_episodes-1]

    actions = []
    for i_episode in range(total_episodes+1):
        observation = env.reset()
        if(i_episode < args.episode):
            continue

        done = 0
        t=0
        if i_episode >= episode_start_test or args.mode == "view":
            agent.epsilon = 0.0

        # agent.epsilon=0.8 - float(i_episode)/total_episodes
        rewards = []
        actions = []

        if args.mode == "train" and (i_episode in episode_to_save) and args.episode != i_episode:
            print("\nSnapshot at episode {}\n".format(i_episode))
            agent.save_snapshot(i_episode)
            agent.save_snapshots_to_file()

        while not (done):
            t+=1
            if render_simulation and (i_episode % interval_render == 0 or i_episode >= episode_start_test):
                env.render()

            action = agent.getAction(observation)
            # action = env.action_space.sample()

            last_obs = observation
            observation, reward, done, info = env.step(action)
            
            if i_episode < episode_start_test and args.mode == "train":
                agent.update(last_obs, action, observation, reward)

            rewards.append(reward)

            if done:
                print(" "*120, end='')
                print("\rEpisode {} stops after {} timesteps\t\t|\t\tMean Reward: {:f}\t\t|\t\tLast Reward: {:f}".format(
                      i_episode,t+1, np.mean(rewards), reward), end='')
                break

    env.close()
    if not use_snapshots:
        agent.save_snapshots_to_file()

    print("")
