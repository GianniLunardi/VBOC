import numpy as np
import pickle
import matplotlib.pyplot as plt
from functools import reduce

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
dofs = 3
if dofs == 2:
    data_dir = 'data_2dof/'
    x_ref = np.array([5 / 4 * np.pi - 0.05, np.pi, 0., 0.])
    Q = np.eye(4) * 1e-4
else:
    data_dir = 'data_3dof_safety_2/'
    x_ref = np.array([5 / 4 * np.pi - 0.05, np.pi, np.pi, 0., 0., 0.])
    Q = np.eye(6) * 1e-4
Q[0, 0] = 500

data_no = pickle.load(open(data_dir + "results_no_constraint.pkl", 'rb'))
data_hard = pickle.load(open(data_dir + "results_hardterm.pkl", 'rb'))
data_soft = pickle.load(open(data_dir + "results_softterm.pkl", 'rb'))
data_straight = pickle.load(open(data_dir + "results_softstraight.pkl", 'rb'))
data_rec = pickle.load(open(data_dir + "results_receiding_softsoft.pkl", 'rb'))

res_steps_no = np.asarray(data_no['res_steps'])
res_steps_hard = np.asarray(data_hard['res_steps'])
res_steps_soft = np.asarray(data_soft['res_steps'])
res_steps_straight = np.asarray(data_straight['res_steps'])
res_steps_rec = np.asarray(data_rec['res_steps'])

fail_no = data_no['failed']
fail_hard = data_hard['failed']
fail_soft = data_soft['failed']
fail_straight = data_straight['failed']
fail_rec_soft = data_rec['failed']

print('#### Better, equal, worse ####')
print('hard term: ' + str(data_hard['better']) + ', ' + str(data_hard['equal']) + ', ' + str(data_hard['worse']))
print('soft term: ' + str(data_soft['better']) + ', ' + str(data_soft['equal']) + ', ' + str(data_soft['worse']))
print('straight: ' + str(data_straight['better']) + ', ' + str(data_straight['equal']) + ', ' + str(data_straight['worse']))
print('rec soft: ' + str(data_rec['better']) + ', ' + str(data_rec['equal']) + ', ' + str(data_rec['worse']))

print('#### Mean and total solver failures ####')
print('Naive MPC: ' + str(np.mean(fail_no)) + ', ' + str(np.sum(fail_no)))
print('hard term: ' + str(np.mean(fail_hard)) + ', ' + str(np.sum(fail_hard)))
print('soft term: ' + str(np.mean(fail_soft)) + ', ' + str(np.sum(fail_soft)))
print('straight: ' + str(np.mean(fail_straight)) + ', ' + str(np.sum(fail_straight)))
print('rec soft: ' + str(np.mean(fail_rec_soft)) + ', ' + str(np.sum(fail_rec_soft)))

print('#### Mean residual steps ####')
print('Naive MPC: ' + str(np.mean(res_steps_no)))
print('MPC with hard terminal constraints: ' + str(np.mean(res_steps_hard)))
print('MPC with soft terminal constraints: ' + str(np.mean(res_steps_soft)))
print('MPC with straight soft terminal constraints: ' + str(np.mean(res_steps_straight)))
print('MPC with receding constraints: ' + str(np.mean(res_steps_rec)))

print('#### Completed tasks ####')
idx_no = np.where(res_steps_no == len(res_steps_no) - 1)[0]
idx_hard = np.where(res_steps_hard == len(res_steps_no) - 1)[0]
idx_soft = np.where(res_steps_soft == len(res_steps_no) - 1)[0]
idx_straight = np.where(res_steps_straight == len(res_steps_no) - 1)[0]
idx_rec = np.where(res_steps_rec == len(res_steps_no) - 1)[0]
idx_common = reduce(np.intersect1d, (idx_no, idx_hard, idx_soft, idx_straight, idx_rec))
print('Naive MPC: ' + str(len(idx_no)))
print('MPC with hard terminal constraints: ' + str(len(idx_hard)))
print('MPC with soft terminal constraints: ' + str(len(idx_soft)))
print('MPC with straight soft terminal constraints: ' + str(len(idx_straight)))
print('MPC with receding constraints: ' + str(len(idx_rec)))
print('Number of tests completed by ALL methods: ' + str(len(idx_common)))

cost_no = mean_cost(np.asarray(data_no['x_traj'])[idx_common], x_ref, Q)
cost_hard = mean_cost(np.asarray(data_hard['x_traj'])[idx_common], x_ref, Q)
cost_soft = mean_cost(np.asarray(data_soft['x_traj'])[idx_common], x_ref, Q)
cost_straight = mean_cost(np.asarray(data_straight['x_traj'])[idx_common], x_ref, Q)
cost_rec_soft = mean_cost(np.asarray(data_rec['x_traj'])[idx_common], x_ref, Q)

print('Times (Mean and 99 quantile) ####')
print('Naive MPC: ' + str(np.mean(data_no['times'])) + ', ' + str(np.quantile(data_no['times'], 0.99)))
print('MPC with hard terminal constraints: ' + str(np.mean(data_hard['times'])) + ', ' + str(np.quantile(data_hard['times'], 0.99)))
print('MPC with soft terminal constraints: ' + str(np.mean(data_soft['times'])) + ', ' + str(np.quantile(data_soft['times'], 0.99)))
print('MPC with straight soft terminal constraints: ' + str(np.mean(data_straight['times'])) + ', ' + str(np.quantile(data_straight['times'], 0.99)))
print('MPC with receding constraints: ' + str(np.mean(data_rec['times'])) + ', ' + str(np.quantile(data_rec['times'], 0.99)))

print('#### Mean running cost ####')
print('Naive MPC: ' + str(cost_no))
print('MPC with hard terminal constraints: ' + str(cost_hard))
print('MPC with soft terminal constraints: ' + str(cost_soft))
print('MPC with straight soft terminal constraints: ' + str(cost_straight))
print('MPC with receding constraints: ' + str(cost_rec_soft))

