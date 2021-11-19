from dqn_agent import DQN_Agent
import gym
from tqdm import tqdm

from empty_env import Environment
import random
import numpy as np

params = {'episodes_solve':10,
'episodes' : 100,
'epsilon' : 1,
}

env_generator = gym.make('CartPole-v0')
input_dim_generator = env_generator.observation_space.shape[0] + 1
output_dim_generator = env_generator.action_space.n
generator = DQN_Agent(seed=1423, network_dims=[input_dim_generator,64,output_dim_generator], lr=1e-3, sync_freq=5,
                  exp_replay_size=256)


env_solver = gym.make('CartPole-v0')
input_dim_solver = env_solver.observation_space.shape[0]
output_dim_solver = env_solver.action_space.n
solver = DQN_Agent(seed=46, network_dims=[input_dim_solver,64,output_dim_solver], lr=1e-3, sync_freq=5,
                  exp_replay_size=25)


def initiliaze_experiance_replay(net, env, is_generator,):
    obs = env.reset()
    done = False
    index = 0
    final_reward = 0
    while not done:
        if is_generator:
            aux = random.choice([-1.,-0.5,0.,0.5,1.])
            obs = np.append(obs,aux)
        A = net.get_action(obs, env.action_space.n, epsilon=1)
        obs_next, reward, done, _ = env.step(A.item())
        #------solver-------
        if is_generator:
            for i in range(solver.experience_replay.maxlen):
                external_reward = initiliaze_experiance_replay(solver,env_solver, False)
                r = reward * aux + external_reward
                net.collect_experience([obs, A.item(), r, np.append(obs_next,aux)])
        else:
            final_reward += reward
            net.collect_experience([obs, A.item(), reward, obs_next])
        # --------------
        obs = obs_next
        index += 1
        if index > net.experience_replay.maxlen:
            break
    return final_reward

def learning_episode(net, env, epsilon, index, is_generator):
    obs, done, losses, ep_len, rew = env.reset(), False, 0, 0, 0
    external_reward = 0
    while not done:
        ep_len += 1
        if is_generator:
            aux = random.choice([-1.,-0.5,0.,0.5,1.])
            obs = np.append(obs,aux)
        A = net.get_action(obs, env.action_space.n, epsilon)
        obs_next, reward, done, _ = env.step(A.item())
        
        #------solver-------
        if is_generator:
            episodes_solve = 10
            epsilon_solve = 1
            index_solve = 128
            for i in range(solver.experience_replay.maxlen):
                external_reward = learning_episode(solver, env_solver, epsilon_solve, index_solve, False)
                r = reward * aux + external_reward
                net.collect_experience([obs, A.item(), r,  np.append(obs_next,aux)])

        else: 
            external_reward += reward
            net.collect_experience([obs, A.item(), reward, obs_next])
        # --------------
        

        obs = obs_next
        rew += reward
        index += 1

        if index > 128:
            index = 0
            for j in range(4):
                loss = net.train(batch_size=16)
                losses += loss
    if epsilon > 0.05:
        epsilon -= (1 / 5000)
    if is_generator:
        losses_list_generator.append(losses / ep_len), reward_list_generator.append(rew)
        episode_len_list_generator.append(ep_len), epsilon_list_generator.append(epsilon)
    else:
        losses_list_solver.append(losses / ep_len), reward_list_solver.append(rew)
        episode_len_list_solver.append(ep_len), epsilon_list_solver.append(epsilon)
    
    return external_reward



# Main training loop
losses_list_generator, reward_list_generator, episode_len_list_generator, epsilon_list_generator = [], [], [], []
losses_list_solver, reward_list_solver, episode_len_list_solver, epsilon_list_solver = [], [], [], []

# initiliaze experiance replay

for i in tqdm(range(generator.experience_replay.maxlen)):
    initiliaze_experiance_replay(generator,env_generator, True)


index = 128
for i in tqdm(range(params['episodes'])):
    learning_episode(generator,env_generator, params['epsilon'], index, True)
    

print("Saving trained model")
generator.save_trained_model("./models/generator_model.pth")
solver.save_trained_model("./models/solver_model.pth")