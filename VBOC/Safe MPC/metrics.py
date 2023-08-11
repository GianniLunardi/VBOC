import numpy as np
import pickle
import matplotlib.pyplot as plt

def state_traj_cost(x, x_ref, Q):
    n = x.shape[0]
    cost = 0
    for i in range(n):
        cost += (x[i] - x_ref).T @ Q @ (x[i] - x_ref)
    return cost

def mean_cost(x_traj, x_ref, Q):
    n = len(x_traj)
    cost = 0
    for i in range(n):
        temp = x_traj[i]
        temp = temp[~np.isnan(x_traj[i]).any(axis=1)]
        cost += state_traj_cost(temp, x_ref, Q)
    return cost/n

# Load data
data_dir = 'data_pickle/'
data_no = pickle.load(open(data_dir + "results_no_constraint.pickle", 'rb'))
data_hard = pickle.load(open(data_dir + "results_hardterm.pickle", 'rb'))
data_soft = pickle.load(open(data_dir + "results_softterm.pickle", 'rb'))
data_rec_hard = pickle.load(open(data_dir + "results_receiding_hardsoft.pickle", 'rb'))
data_rec_soft = pickle.load(open(data_dir + "results_receiding_softsoft.pickle", 'rb'))

res_steps_no = data_no['res_steps']
res_steps_hard = data_hard['res_steps']
res_steps_soft = data_soft['res_steps']
res_steps_rec_hard = data_rec_hard['res_steps']
res_steps_rec_soft = data_rec_soft['res_steps']

print('#### Mean residual steps ####')
print('Naive MPC: ' + str(np.mean(res_steps_no)))
print('MPC with hard terminal constraints: ' + str(np.mean(res_steps_hard)))
print('MPC with soft terminal constraints: ' + str(np.mean(res_steps_soft)))
print('MPC with receding (hard + hard) constraints: ' + str(np.mean(res_steps_rec_hard)))
print('MPC with receding (soft + soft) constraints: ' + str(np.mean(res_steps_rec_soft)))

# Reference joint configuration
x_ref = np.array([5/4*np.pi - 0.05, np.pi, np.pi, 0., 0., 0.])

# Compute the running costs for each trajectory
Q = np.eye(6) * 1e-4
Q[0,0] = 1e4

cost_no = mean_cost(data_no['x_traj'], x_ref, Q)
cost_hard = mean_cost(data_hard['x_traj'], x_ref, Q)
cost_soft = mean_cost(data_soft['x_traj'], x_ref, Q)
cost_rec_hard = mean_cost(data_rec_hard['x_traj'], x_ref, Q)
cost_rec_soft = mean_cost(data_rec_soft['x_traj'], x_ref, Q)

print('#### Mean running cost ####')
print('Naive MPC: ' + str(cost_no))
print('MPC with hard terminal constraints: ' + str(cost_hard))
print('MPC with soft terminal constraints: ' + str(cost_soft))
print('MPC with receding (hard + hard) constraints: ' + str(cost_rec_hard))
print('MPC with receding (soft + soft) constraints: ' + str(cost_rec_soft))