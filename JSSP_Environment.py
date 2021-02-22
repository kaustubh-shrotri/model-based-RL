#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 14:04:54 2019

@author: at-lab
"""
import torch
import numpy as np
import pandas as pd
import math

'''
Ensure that the excel file 'sasd_a2c.xlsx' is always kept in the same folder as the 'JSSP_Environment.py'
'''

file = r'sasd_a2c.xlsx'
load_sp = pd.ExcelFile(file)

# state_action_state' table
df1 = load_sp.parse('state_action_nstate_reward')
state_action_nstate = torch.FloatTensor(df1.values)

# action index
df2 = load_sp.parse('action_index')
action_index_data = torch.FloatTensor(df2.values)

class env():

    def __init__(self):
        super(env, self).__init__()
        self.s = 0
        self.count_op1 = 0
        self.count_op2 = 0
        self.count_ip1 = 0
        self.count_ip2 = 0
        self.done = False

    def check(self, n_state, rew):
        '''Method to reset the environment based on terminal conditions of States'''
        if n_state == 14 or n_state == 17:
            rew = torch.tensor([-50])
            self.done = True
            
        return n_state, rew

    def check1(self, count_op1, count_op2,rew):
        '''Method to reset the environment based on terminal conditions of Output counts'''
        if self.count_op1 >= 10 and self.count_op2 >= 10:
            rew = torch.tensor([+50])
            self.done = True
            
        elif self.count_op2 > 12:
            rew = torch.tensor([-50])
            self.done = True
            
        elif self.count_op1 > 12 :
            rew = torch.tensor([-50])
            self.done = True
        return rew

    def check2(self, count_ip1, count_ip2,rew):
        '''Method to reset the environment based on terminal conditions of Input counts'''
        if self.count_ip1 >= 10:
            rew = torch.tensor([-50])
            self.done = True

        elif self.count_ip2 >= 10:
            rew = torch.tensor([-50])
            self.done = True
        return rew

    def reset(self):
        '''Method to reset the Environment'''
        self.s = torch.zeros(1)
        self.count_op1 = 0
        self.count_op2 = 0
        self.count_ip1 = 0
        self.count_ip2 = 0
        self.done = False
        return self.s



    def next_state(self, a):
        '''Here in this method we give action "a" as input
        and receive state "s", reward "rew", info of reset as outputs'''

        a1 = math.trunc(a/10)
        a2 = a - (a1*10)
        y = state_action_nstate[np.where(state_action_nstate[:,0]==self.s),:]
        z = y[0][np.where(y[0][:,1]==a1),:]
        self.s = z[0][np.where(z[0][:,2]==a2),3].view(-1)
        op1_count = z[0][np.where(z[0][:,2]==a2),4].view(-1)
        op2_count = z[0][np.where(z[0][:,2]==a2),5].view(-1)
        rew1 = z[0][np.where(z[0][:,2]==a2),6].view(-1)
        rew2 = z[0][np.where(z[0][:,2]==a2),7].view(-1)
        ip1_count = z[0][np.where(z[0][:,2]==a2),8].view(-1)
        ip2_count = z[0][np.where(z[0][:,2]==a2),9].view(-1)
        self.count_op1+=op1_count
        self.count_op2+=op2_count
        self.count_ip1+=ip1_count
        self.count_ip2+=ip2_count
        rew = rew1 + rew2

        self.s, rew= self.check(self.s, rew)
        rew = self.check1(self.count_op1, self.count_op2, rew)
        rew = self.check2(self.count_ip1, self.count_ip2, rew)
        # if self.done == True:
        #     self.reset()

        return self.s, rew, self.done, [self.count_op1, self.count_op2, self.count_ip1, self.count_ip2]


# jssp = env()
# actions = action_index_data[:,0]

# for i in range(len(actions)):
#     action = np.random.choice(actions.numpy())
#     new_state , reward, done = jssp.next_state(action)
#     print(i,'. new state: ', new_state, 'reward: ', reward, 'done: ', done)
#     if done == True:
#         break

'''
sample code:
    x = [5,15,27,38,49] # A variable'x' with random values

# Through the function below we are feeding the random values of 'x' as
  actions 'a' to our environment and acquiring 'Next state', 'Reward'
  and 'Reset Info' as outputs

Inputs:

from environment import env   # Importing the Environment
x = [5,14,27,38,49]
for i in range(len(x)):
        a = x[i]
    print(env1.next_state(a))

Outputs:
    1.(tensor([6.]), tensor([-0.1000]), False) # Here intially at (state-0) we have
    given 5 as action-value to environment and we received next state as (6) and reward
    as (-0.1) and reset info as false.

    2.(tensor([15.]),tensor([-0.2000]), False) # At (state-6) we have given 15 as action-value
    to environment and we received next state as (15) and reward as (-0.2) and reset info false.

    3.(tensor([0.]), tensor([-0.2000]), False) # At (state-15) we have given 27 as action-value
    to environment and we received next state as (0) and reward as (-0.2) and reset info false.

    4.(tensor([0.]), tensor([-0.2000]), False) # At (state-0) we have given 38 as action-value
    to environment and we received next state as (0) and reward as (-0.2) and reset info false.

    5.(tensor([0.]), tensor([-0.2000]), False) # At (state-0) we have given 49 as action-value
    to environment and we received next state as (0) and reward as (-0.2) and reset info false.

## In this manner this environment will generate output according to your input#

'''