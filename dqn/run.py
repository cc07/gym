import gym
import sys
import numpy as np

from collections import deque
from dqn import DeepQNetwork
from gym import wrappers

def rgb2gray(rgb):
    return np.expand_dims(np.dot(rgb[...,:3], [0.299, 0.587, 0.114]), axis=3)

if __name__ == '__main__':

    env = gym.make('Pendulum-v0')
    env = env.unwrapped
    env.seed(1)
    # env = wrappers.Monitor(env, '/tmp/FrozenLake-v0-experiment-15')

    n_action = 11
    n_width = 3
    n_height = 1
    n_channel = 1

    n_episode = 1000
    e_greedy_increment = 0.001
    learning_rate = 0.005
    memory_size = 3000
    dueling = True
    prioritized = True
    double_q = True

    dqn = DeepQNetwork(n_action, n_width, n_height, n_channel, e_greedy_increment=e_greedy_increment, \
        memory_size=memory_size, \
        learning_rate=learning_rate, dueling=dueling, prioritized=prioritized, double_q=double_q)
    # dqn.load(372)

    counter = 0
    state = deque([], maxlen=n_width)
    state_ = deque([], maxlen=n_width)

    for i in range(n_episode):

        observation = env.reset()
        state_.append(observation)
        # observation = np.identity(16)[observation:observation + 1]
        # observation = np.expand_dims(observation, axis=2)
        # observation = np.expand_dims(observation, axis=3)

        # observation = rgb2gray(observation)
        score = 0

        while True:

            env.render()

            # action = dqn.choose_action(state, True if counter > n_width else False)
            action = dqn.choose_action(observation)
            f_action = (action-(n_action-1)/2)/((n_action-1)/4)

            observation_, reward, done, info = env.step(np.array([f_action]))

            reward = reward / 10
            # observation_ = np.identity(16)[observation_:observation_ + 1]
            # observation_ = np.expand_dims(observation_, axis=2)
            # observation_ = np.expand_dims(observation_, axis=3)
            # observation_ = rgb2gray(observation_)

            score += reward

            state_.append(observation_)

            # if counter > n_width:
            #     dqn.store_transition(np.expand_dims(np.array(list(state)), axis=3), action, reward, np.expand_dims(np.array(list(state_)), axis=3))
            dqn.store_transition(observation, action, reward, observation_)
            # dqn.store_transition(np.reshape(observation, (1, 3, 1)), action, reward, np.reshape(observation_, (1, 3, 1)))
            counter += 1

            if counter > memory_size:
                dqn.learn()
                # dqn.finish_episode(counter, reward)

            if done:
                print ('Ep: {} Score: {}'.format(i, score))
                # dqn.finish_episode(i, score)
                dqn.save()
                break;
            # print(counter)

            observation = observation_
            state = state_
