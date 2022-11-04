import gym
import numpy as np
import plotly.express as px
from network import QNetwork
from memory import Buffermem
import torch as T


def agent(param):
    gamma = param['gamma']
    epsilon = param['epsilon']
    lr = param['learning_rate']
    num_actions = param['action_space']
    input_dims = param['observation_space']
    batch_size = param['batch_size']
    eps_min = param['epsilon_min']
    eps_dec = param['epsilon_dec']
    replace = param['replace']
    mem_size = param['buffer_mem_size']
    episode = param['num_games']
    learn_step = 0
    action_space = []
    for i in range(num_actions):
        action_space.append(i)
    memory = Buffermem(mem_size, input_dims)
    q_cur = QNetwork(lr, num_actions,
                     input_dims=input_dims)
    q_p = QNetwork(lr, num_actions,
                   input_dims=input_dims)
    num_step = 0
    best_sum_rewards = -np.inf
    sum_rewards = []
    eps_in_step = []
    steps = []
    for i in range(episode):
        donne = False
        observ = env.reset()
        observ = observ[0]

        sum_reward = 0
        while not donne:
            if np.random.random() > epsilon:
                state = T.tensor([observ], dtype=T.float).to(q_cur.device)
                actions = q_cur.forward(state)
                action = T.argmax(actions).item()
            else:
                action = np.random.choice(action_space)
            observation, reward, donne, info, _ = env.step(action)
            sum_reward += reward
            memory.store_step_param(observ, action,
                                    reward, observation, donne)
            q_cur.optimizer.zero_grad()

            if learn_step % replace == 0:
                q_p.load_state_dict(q_cur.state_dict())

            state, action, reward, new_state, done = memory.sample(batch_size)

            states = T.tensor(state).to(q_cur.device)
            rewards = T.tensor(reward).to(q_cur.device)
            dones = T.tensor(done).to(q_cur.device)
            actions = T.tensor(action).to(q_cur.device)
            states_ = T.tensor(new_state).to(q_cur.device)

            indices = np.arange(batch_size)

            q_predict = q_cur.forward(states)[indices, actions]
            q_n = q_p.forward(states_).max(dim=1)[0]

            q_n[dones] = 0.0
            q_target = rewards + gamma * q_n

            loss = q_cur.loss(q_target, q_predict).to(q_cur.device)
            loss.backward()
            q_cur.optimizer.step()
            learn_step += 1

            if epsilon > eps_min:
                epsilon = epsilon - eps_dec
            else:
                epsilon = eps_min
            observ = observation
            num_step += 1
        sum_rewards.append(sum_reward)
        steps.append(num_step)

        avg_rewards = np.mean(sum_rewards[-100:])
        print('episode: ', i, 'sum_reward: ', sum_reward)

        if avg_rewards > best_sum_rewards:
            best_sum_rewards = avg_rewards

        eps_in_step.append(epsilon)
    return steps, sum_rewards, eps_in_step


if __name__ == '__main__':
    env = gym.make('CartPole-v1')

    params = {
        'gamma': 0.99,
        'epsilon': 1,
        'learning_rate': 0.0001,
        'observation_space': env.observation_space.shape,
        'action_space': env.action_space.n,
        'buffer_mem_size': 50000,
        'epsilon_min': 0.1,
        'batch_size': 32,
        'replace': 1000,
        'epsilon_dec': 1e-5,
        'num_games': 200
    }

    steps, sum_rewards, eps_in_step = agent(params)
    fig = px.scatter(x=steps, y=sum_rewards).update_layout(
        xaxis_title="Step", yaxis_title="Score")
    fig.show()
    fig1 = px.line(x=steps, y=eps_in_step).update_layout(
        xaxis_title="Step", yaxis_title="Epsilon")
    fig1.show()
