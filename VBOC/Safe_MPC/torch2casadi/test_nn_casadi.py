import os
import sys
sys.path.insert(1, os.getcwd() + '/..')

import numpy as np
import time
import torch
from doublependulum_nn_casadi import OCPdoublependulumHardTerm, SYMdoublependulum
from my_nn import NeuralNetDIR

import warnings
warnings.filterwarnings("ignore")

start_time = time.time()

# Pytorch params:
device = torch.device("cpu")

model = NeuralNetDIR(4, 300, 1).to(device)
model.load_state_dict(torch.load('../model_2dof_vboc'))
mean = torch.load('../mean_2dof_vboc')
std = torch.load('../std_2dof_vboc')
safety_margin = 5.0

cpu_num = 5
test_num = 100

time_step = 5*1e-3
tot_time = 0.16 - 4 * time_step
tot_steps = 100

regenerate = True

ocp = OCPdoublependulumHardTerm("SQP_RTI", time_step, tot_time, model, mean, std, safety_margin, regenerate)
print('OCP ok')

elapsed_time = time.time() - start_time
print('Elapsed time: ', elapsed_time)
