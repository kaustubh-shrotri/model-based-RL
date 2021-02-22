# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 22:46:37 2021

@author: Kaustubh
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from nlrl import NLRL_AO,InverseSig

file = pd.ExcelFile(r'sasd_a2c.xlsx')
state_index_oh = file.parse('state_index')

class ReplayBuffer(object):
    def __init__(self, mem_size, batch_size, states, actionsz):
        
        self.max_mem = mem_size
        self.mem_cntr = 0
        self.batch_size = batch_size
        self.states = states
        self.actionsz = actionsz
        
    def buffer_reset(self):
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.max_mem, self.states))
        self.new_state_memory = np.zeros((self.max_mem, 1))
        self.action_memory = np.zeros((self.max_mem,self.actionsz))
        self.reward_memory = np.zeros((self.max_mem,1))
        self.terminal_memory = np.zeros((self.max_mem,1))
        
    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.max_mem
        
        self.state_memory[index] = state
        actions = np.zeros(self.actionsz)
        actions[action] = 1.0
        self.action_memory[index] = actions
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done
        
        self.mem_cntr += 1
        
    def sample_buffer(self):
        memory = min(self.max_mem, self.mem_cntr)
        
        batch = np.random.choice(memory, self.batch_size,replace = False)
        
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        new_states = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]
        
        return states, actions, rewards, new_states, dones

class QNetwork(nn.Module):
    def __init__(self, input_size, fc1_dims, output_size, lr):
        super(QNetwork, self).__init__()
        
        self.fc1 = nn.Linear(input_size, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, output_size)
        self.relu = nn.ReLU()
        self.mse_loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr = lr)
        self.device = torch.device('cpu')
        self.to(self.device)
        
        
    def forward(self, x):
        
        x = self.relu(self.fc1(x))
        out = self.fc2(x)
        
        return out
    
class NextStateModel(nn.Module):
    def __init__(self, input_dims, fc1_dims, action_dims,output_dims):
        super(NextStateModel, self).__init__()
        self.input_dims = input_dims
        self.action_dims = action_dims
        self.fc1 = nn.Linear(self.input_dims+self.action_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, output_dims)
        self.optimizer = optim.Adam(self.parameters(), lr=0.0005)
        self.loss_fn = nn.CrossEntropyLoss()
        self.relu = nn.ReLU()
        self.device = torch.device('cpu')
        self.to(self.device)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
class NextStateModelNLRL(nn.Module):
    def __init__(self, input_dims, fc1_dims, action_dims,output_dims, lr):
        super(NextStateModelNLRL, self).__init__()
        self.input_dims = input_dims
        self.action_dims = action_dims
        self.fc1 = nn.Linear(self.input_dims+self.action_dims, fc1_dims)
        self.fc2 = NLRL_AO(fc1_dims, output_dims)
        self.optimizer = optim.Adam(self.parameters(), lr=lr) #0.005
        self.loss_fn = nn.CrossEntropyLoss()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.device = torch.device('cpu')
        self.to(self.device)
        
    def forward(self, x):
        # x = self.relu(self.fc1(x))
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        
        return x

class RewardModel(nn.Module):
    def __init__(self, input_dims, fc1_dims, action_dims):
        super(RewardModel, self).__init__()
        self.input_dims = input_dims
        self.linear1 = nn.Linear(self.input_dims+action_dims, fc1_dims)
        self.linear2 = nn.Linear(fc1_dims, 1)
        self.relu = nn.ReLU()
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr = 0.001)
        self.device = torch.device('cpu')
        self.to(self.device)

    def forward(self, x):
        out = self.relu(self.linear1(x))
        out = self.linear2(out)
        return out

class RewardModelNLRL(nn.Module):
    def __init__(self, input_dims, fc1_dims, action_dims, lr):
        super(RewardModelNLRL, self).__init__()
        self.input_dims = input_dims
        self.linear1 = nn.Linear(self.input_dims+action_dims, fc1_dims)
        self.linear2 = nn.Linear(fc1_dims, 1)
        self.relu = nn.ReLU()
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr = lr)  #0.001
        self.device = torch.device('cpu')
        self.to(self.device)

    def forward(self, x):
        out = self.relu(self.linear1(x))
        out = self.linear2(out)
        return out
    
class DoneModel(nn.Module):
    def __init__(self, input_dims, fc1_dims, action_dims,output_dims):
        super(DoneModel, self).__init__()
        self.input_dims = input_dims
        self.action_dims = action_dims
        self.fc1 = nn.Linear(self.input_dims+self.action_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, output_dims)
        self.relu = nn.ReLU()
        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)
        self.loss_fn = nn.CrossEntropyLoss() #nn.BCEWithLogitsLoss()
        self.device = torch.device('cpu')
        self.to(self.device)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

class DoneModelNLRL(nn.Module):
    def __init__(self, input_dims, fc1_dims, action_dims,output_dims, lr):
        super(DoneModelNLRL, self).__init__()
        self.input_dims = input_dims
        self.action_dims = action_dims
        self.fc1 = NLRL_AO(self.input_dims+self.action_dims, output_dims)
        self.fc2 = nn.Linear(fc1_dims, output_dims)
        self.relu = nn.ReLU()
        self.optimizer = optim.Adam(self.parameters(), lr=lr) #0.0001
        self.loss_fn = nn.CrossEntropyLoss() #nn.BCEWithLogitsLoss()
        self.device = torch.device('cpu')
        self.to(self.device)
        
    def forward(self, x):
        # x = self.relu(self.fc1(x))
        x = self.fc1(x)
        
        return x
    
class Agent(object):
    def __init__(self, state_size, action_size, fc1_dims, lr, batch_size, buffer_size, gamma, tau, lr_ns, lr_r, lr_d):
        
        self.state_size = state_size
        self.action_size = action_size
        self.fc1_dims = fc1_dims
        self.lr = lr
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.tau = tau
        self.tstep = 0
        self.action_space = [i for i in range(self.action_size)]
        self.lr_ns =lr_ns
        self.lr_r =lr_r
        self.lr_d =lr_d
        
        self.q_network = QNetwork(self.state_size, self.fc1_dims, self.action_size, self.lr)
        self.target_q_network = QNetwork(self.state_size, self.fc1_dims, self.action_size, self.lr)
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size, self.state_size,self.action_size)
        self.next_state_model = NextStateModelNLRL(self.state_size, self.fc1_dims, self.action_size*2, output_dims=27, lr = self.lr_ns)
        self.reward_model = RewardModelNLRL(self.state_size, self.fc1_dims, self.action_size*2, lr = self.lr_r)
        self.done_model = DoneModelNLRL(self.state_size, self.fc1_dims, self.action_size*2, output_dims=2, lr = self.lr_d)
        
    
    def act(self, state, eps):
        state = torch.tensor(state, dtype= torch.float).to(self.q_network.device)
        
        if np.random.random() > eps:
            action_values = self.q_network.forward(state)
            action = torch.argmax(action_values).item()
        else:
            action = np.random.choice(self.action_size)
            
        return action
    
    def train_model(self, agent):
        if self.memory.mem_cntr < self.batch_size:
            return
        
        states, actions, rewards, new_states, dones = self.memory.sample_buffer()
        ns_correct = 0
        d_correct = 0
        total = dones.shape[0]
        
        # rewards = torch.tensor(rewards, dtype=torch.float).to(self.reward_model.device)
        # dones = torch.tensor(dones, dtype=torch.bool).to(self.done_model.device)
        # new_states = torch.tensor(new_states, dtype=torch.float).to(self.next_state_model.device)
        dummy_actions = np.zeros(actions.shape)
        if agent == 1:
            actions = np.concatenate((actions,dummy_actions),axis=1)
        else:
            actions = np.concatenate((dummy_actions,actions),axis=1)
        states_actions = np.concatenate((states,actions), axis=1)
        
        # new_states = torch.tensor(state_index_oh.iloc[new_states.int().numpy(),2:].values, dtype=torch.float).to(self.q_network.device)#.reshape(-1)
        
        states_actionsns = torch.tensor(states_actions, dtype=torch.float).to(self.next_state_model.device)
        new_states = torch.tensor(new_states, dtype=torch.long).to(self.next_state_model.device)
        new_state_=  self.next_state_model(states_actionsns) 
        ns_loss = self.next_state_model.loss_fn(new_state_, new_states.view(-1))
        self.next_state_model.optimizer.zero_grad()
        ns_loss.backward()
        self.next_state_model.optimizer.step()
        _,ns_pred = torch.max(new_state_, dim=1)
        ns_correct += (ns_pred == new_states.view(-1)).sum().item()
        ns_acc = 100*ns_correct/total
        
        states_actionsr = torch.tensor(states_actions, dtype=torch.float).to(self.reward_model.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.reward_model.device)
        reward_ = self.reward_model(states_actionsr)
        r_loss = self.reward_model.loss_fn(reward_, rewards)
        self.reward_model.optimizer.zero_grad()
        r_loss.backward()
        self.reward_model.optimizer.step()
        
        states_actionsd = torch.tensor(states_actions, dtype=torch.float).to(self.done_model.device)
        dones = torch.tensor(dones, dtype=torch.long).to(self.reward_model.device)
        dones_ = self.done_model(states_actionsd)
        d_loss = self.done_model.loss_fn(dones_, dones.view(-1))
        self.done_model.optimizer.zero_grad()
        d_loss.backward()
        self.done_model.optimizer.step()
        _,d_pred = torch.max(dones_, dim=1)
        d_correct += (d_pred == dones.view(-1)).sum().item()
        d_acc = 100*d_correct/total
        return [round(ns_acc,2), round(r_loss.item(),2), round(d_acc,2)]
        
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        
        states, actions, rewards, new_states, dones = self.memory.sample_buffer()
        
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.q_network.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.q_network.device)
        new_states = torch.tensor(state_index_oh.iloc[new_states.reshape(-1),2:].values, dtype=torch.float).to(self.q_network.device)#.reshape(-1)
        # action_batch = self.action_memory[batch]
        action_values = np.array(self.action_space, dtype=np.int32)
        action_indices = np.dot(actions, action_values).reshape(-1,1)
        actions = torch.tensor(action_indices, dtype=torch.long).to(self.q_network.device)
        states = torch.tensor(states, dtype=torch.float).to(self.q_network.device)
        
        q_values = self.q_network.forward(states).gather(1, actions)
        
        Qsa_prime_target_values = self.target_q_network(new_states).detach()
        Qsa_prime_targets = Qsa_prime_target_values.max(1)[0].unsqueeze(1)
        
        Qsa_prime_targets[dones]=0.0
        # Compute Q targets for current states 
        Qsa_targets = rewards + (self.gamma * Qsa_prime_targets)# *  (1-done))
        
        loss = self.q_network.mse_loss(q_values, Qsa_targets).to(self.q_network.device)
        
        # Minimize the loss
        self.q_network.optimizer.zero_grad()
        loss.backward()
        self.q_network.optimizer.step()
        
        self.tstep+=1
        if self.tstep%100 ==0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())
            self.tstep = 0
            
    def sim_learn(self, agent):
        if self.memory.mem_cntr < self.batch_size:
            return
        
        states, actions, rewards, new_states, dones = self.memory.sample_buffer()
        dummy_actions = np.zeros(actions.shape)
        if agent == 1:
            states_actions = np.concatenate((states,actions,dummy_actions),axis=1)
        else:
            states_actions = np.concatenate((states,dummy_actions,actions),axis=1)
        # states_actions = np.concatenate((states,actions), axis=1)
        
        states_actionsns = torch.tensor(states_actions, dtype=torch.float).to(self.next_state_model.device)
        states_actionsr = torch.tensor(states_actions, dtype=torch.float).to(self.reward_model.device)
        states_actionsd = torch.tensor(states_actions, dtype=torch.float).to(self.done_model.device)
        
        new_states=  self.next_state_model(states_actionsns)
        _, new_states = torch.max(new_states, dim=1)
        new_states = torch.tensor(state_index_oh.iloc[new_states.int().numpy(),2:].values, dtype=torch.float).to(self.q_network.device)#.reshape(-1)
        rewards = self.reward_model(states_actionsr)
        dones = self.done_model(states_actionsd)
        _, dones = torch.max(dones, dim=1)
        
        action_values = np.array(self.action_space, dtype=np.int32)
        action_indices = np.dot(actions, action_values).reshape(-1,1)
        actions = torch.tensor(action_indices, dtype=torch.long).to(self.q_network.device)
        states = torch.tensor(states, dtype=torch.float).to(self.q_network.device)
        
        q_values = self.q_network.forward(states).gather(1, actions)
        
        Qsa_prime_target_values = self.target_q_network(new_states).detach()
        Qsa_prime_targets = Qsa_prime_target_values.max(1)[0].unsqueeze(1)
        
        Qsa_prime_targets[dones]=0.0
        # Compute Q targets for current states 
        Qsa_targets = rewards + (self.gamma * Qsa_prime_targets)# *  (1-done))
        
        loss = self.q_network.mse_loss(q_values, Qsa_targets).to(self.q_network.device)
        
        # Minimize the loss
        self.q_network.optimizer.zero_grad()
        loss.backward()
        self.q_network.optimizer.step()
        

        
        
        
        
        
        