# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 16:28:20 2021

@author: Kaustubh
"""

from JSSP_Environment import env
from dqn_agent import Agent
import pandas as pd
import numpy as np
import torch
# import torchvision
from torch.utils.tensorboard import SummaryWriter
import optuna
# from torchvision import datasets, transforms

def epsilon_decay(eps,min_eps=0.05, decay=0.995):
	return max(eps*decay, min_eps)




def objective(trial):
    file = pd.ExcelFile(r'sasd_a2c.xlsx')
    state_index_oh = file.parse('state_index')
    MaxEpisodes = 2000
    Env = env()
    EPSILON = 1
    Total_Reward = []
    Avg_Rewards = []
    # output1_lst = []
    # output2_lst = []
    # input1_lst = []
    # input2_lst = []
    
    
    fc1_dims = trial.suggest_categorical('fc1_dims',[15,20,30])
    lr = trial.suggest_uniform("lr", 5e-6,1e-4)
    gamma = trial.suggest_categorical("gamma", [0.97, 0.98,0.99])
    lr_ns = trial.suggest_uniform("lr_ns", 1e-4, 1e-2)
    lr_r = trial.suggest_uniform("lr_r", 1e-4, 5e-3)
    lr_d = trial.suggest_uniform("lr_d",1e-4, 1e-3)

    agent1 = Agent(state_size=9, action_size=10, fc1_dims=fc1_dims, lr=lr, batch_size=64, buffer_size=100000, gamma=gamma, tau=0.002, lr_ns=lr_ns, lr_r=lr_r, lr_d=lr_d)   #fc1=32 lr=0.0009 gamma=0,98
    agent2 = Agent(state_size=9, action_size=10, fc1_dims=fc1_dims, lr=lr, batch_size=64, buffer_size=100000, gamma=gamma, tau=0.002, lr_ns=lr_ns, lr_r=lr_r, lr_d=lr_d)
    writer = SummaryWriter()
    writer.add_graph(agent1.q_network, torch.from_numpy(state_index_oh.iloc[:,2:].values).float())
    writer.add_graph(agent2.q_network, torch.from_numpy(state_index_oh.iloc[:,2:].values).float())
    writer.close()
    agent1.memory.buffer_reset()
    agent2.memory.buffer_reset()
    for ep in range(MaxEpisodes):
        state = Env.reset() #torch.zeros(1)
        # agent.memory.buffer_reset()
        done = False
        stepscounter = 0
        ep_reward = 0
        state_OH = state_index_oh.iloc[state.int().numpy(),2:].values.reshape(-1)
        
        while not done:
            stepscounter += 1
            
            action1 = agent1.act(state_OH, EPSILON)
            action = action1*10
            new_state, reward, done, obs = Env.next_state(action)
            ep_reward += reward
            new_state_OH = state_index_oh.iloc[new_state.int().numpy(),2:].values.reshape(-1)
            agent1.memory.store_transition(state_OH, action1, 2*((reward.item()+50)/100)-1, new_state, done)
            state_OH = new_state_OH
            
            if done == True:
                break
            action2 = agent2.act(state_OH, EPSILON)
            if action2 == action1:
                continue
            action = action2
            new_state, reward, done, obs = Env.next_state(action)
            ep_reward += reward
            new_state_OH = state_index_oh.iloc[new_state.int().numpy(),2:].values.reshape(-1)
            agent2.memory.store_transition(state_OH, action2, 2*((reward.item()+50)/100)-1, new_state, done)
            state_OH = new_state_OH
            
            agent1.learn()
            agent2.learn()
            update_model1 = agent1.train_model(1)
            update_model2 = agent2.train_model(2)
            for _ in range(5):
                agent1.sim_learn(1)
                agent2.sim_learn(2)
            
            
        output1 = obs[0].item()
        output2 = obs[1].item()
        input1 = obs[2].item()
        input2 = obs[3].item()
            
        EPSILON = epsilon_decay(eps=EPSILON)    
        Total_Reward.append(ep_reward)
        avg_reward = np.mean(Total_Reward[-100:])
        Avg_Rewards.append(avg_reward)
        
        
        
        
        if ep % 1 == 0:
            totalresult= 'episode: '+ str(ep+1)+ '  Total_Reward %.2f'% ep_reward+ '  Average_Reward %.2f'% avg_reward+ '  Steps '+str(stepscounter)+' Model Training Data: '+str(update_model1)+str(update_model2)#+' Output1: '+str(output1)+' Output2: '+str(output2)
            # dataCollect("Total Result",Total_Result,totalresult,i_episode)
            print(f'\r{totalresult}', end='\r')
        
        writer.add_scalar('reward/episode', ep_reward, ep)
        writer.add_scalar('Avgreward/episode', avg_reward, ep)
        writer.add_scalar('output1/episode', output1, ep)
        writer.add_scalar('output2/episode', output2, ep)
        writer.add_scalar('input1/episode', input1, ep)
        writer.add_scalar('input2/episode', input2, ep)
        
        trial.report(avg_reward, ep)
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return  avg_reward

sampler=optuna.samplers.TPESampler()
pruner = optuna.pruners.MedianPruner(n_startup_trials=5,n_warmup_steps=650, interval_steps=5)

study = optuna.create_study(sampler=sampler,pruner=pruner, direction='maximize', storage="sqlite:///optuna_runs.db")
study.optimize(objective, n_trials=50)
importances = optuna.importance.get_param_importances(study)

print(study.best_params)
print(importances)



        