"""
Deep Q network,

"""


import gym
from RL_brain_DDQN import DeepQNetwork
import time
import math
from pd_control import pd_control

env = gym.make('CartPole_SwingUp-v0')
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = DeepQNetwork(n_actions=env.action_space.n,
                  n_features=env.observation_space.shape[0],
                  learning_rate=0.01, e_greedy=0.5,
                  replace_target_iter=100, memory_size=2000,
                  e_greedy_increment=0.001,)

total_steps = 0


for i_episode in range(100):

    observation = env.reset()
    ep_r = 0
    while True:
        env.render()

        action = RL.choose_action(observation)
        print(action)

        observation_, reward, done, info = env.step(action)

        # the smaller theta and closer to center the better
        x, x_dot, theta, theta_dot = observation_
        r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
        # r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
        # r1 = 0 if abs(x) < 1.5 else -0.5
        r2 = abs(theta)
        r3 = 0 if 0.02*abs(theta_dot) < (math.pi - abs(theta)) else (math.pi - abs(theta)) - 0.02*abs(theta_dot)  # 控制角速度
        reward = 4*r1 + r2 + r3

        RL.store_transition(observation, action, reward, observation_)

        ep_r += reward
        if total_steps > 1000:
            RL.learn()

        if done:
            print('episode: ', i_episode,
                  'ep_r: ', round(ep_r, 2),
                  ' epsilon: ', round(RL.epsilon, 2))
            break

        observation = observation_
        total_steps += 1


# 测试过程
for i in range(1):
    observation = env.reset()
    count = 0
    done = False
    while True:
        # 采样动作，探索环境
        env.render()

        if done:
            # action = pd_control(observation)
            # print('pd：', action)
            break
        else:
            action = RL.greedy(observation)
        # action = RL.choose_action(observation)
        # action = RL.sample_action(observation)
        # print (action)
        # print(action1)
        observation_, reward, done, info = env.step(action)

        string = input("stop")
        if string == 's':
            break
        observation = observation_
        count += 1
        # time.sleep(0.2)
        print(count)
