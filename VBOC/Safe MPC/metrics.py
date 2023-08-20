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
dofs = 2
if dofs == 2:
    data_dir = 'data_2dof/'
    x_ref = np.array([5 / 4 * np.pi - 0.05, np.pi, 0., 0.])
    Q = np.eye(4) * 1e-4
else:
    data_dir = 'data_3dof/'
    x_ref = np.array([5 / 4 * np.pi - 0.05, np.pi, np.pi, 0., 0., 0.])
    Q = np.eye(6) * 1e-4
Q[0, 0] = 1e4

data_no = pickle.load(open(data_dir + "results_no_constraint.pickle", 'rb'))
data_hard = pickle.load(open(data_dir + "results_hardterm.pickle", 'rb'))
data_soft = pickle.load(open(data_dir + "results_softterm.pickle", 'rb'))
data_rec_hard = pickle.load(open(data_dir + "results_receiding_hardsoft.pickle", 'rb'))
data_rec_soft = pickle.load(open(data_dir + "results_receiding_softsoft.pickle", 'rb'))

res_steps_no = np.asarray(data_no['res_steps'])
res_steps_hard = np.asarray(data_hard['res_steps'])
res_steps_soft = np.asarray(data_soft['res_steps'])
res_steps_rec_hard = np.asarray(data_rec_hard['res_steps'])
res_steps_rec_soft = np.asarray(data_rec_soft['res_steps'])

fail_no = data_no['failed']
fail_hard = data_hard['failed']
fail_soft = data_soft['failed']
fail_rec_hard = data_rec_hard['failed']
fail_rec_soft = data_rec_soft['failed']

print('#### Better, equal, worse ####')
print('hard term: ' + str(data_hard['better']) + ', ' + str(data_hard['equal']) + ', ' + str(data_hard['worse']))
# print('soft term: ' + str(data_soft['better']) + ', ' + str(data_soft['equal']) + ', ' + str(data_soft['worse']))
# print('rec hard: ' + str(data_rec_hard['better']) + ', ' + str(data_rec_hard['equal']) + ', ' + str(data_rec_hard['worse']))
print('rec soft: ' + str(data_rec_soft['better']) + ', ' + str(data_rec_soft['equal']) + ', ' + str(data_rec_soft['worse']))

print('#### Mean and total solver failures ####')
print('Naive MPC: ' + str(np.mean(fail_no)) + ', ' + str(np.sum(fail_no)))
print('hard term: ' + str(np.mean(fail_hard)) + ', ' + str(np.sum(fail_hard)))
# print('soft term: ' + str(np.mean(fail_soft)) + ', ' + str(np.sum(fail_soft)))
# print('rec hard: ' + str(np.mean(fail_rec_hard)) + ', ' + str(np.sum(fail_rec_hard)))
print('rec soft: ' + str(np.mean(fail_rec_soft)) + ', ' + str(np.sum(fail_rec_soft)))

print('#### Mean residual steps ####')
print('Naive MPC: ' + str(np.mean(res_steps_no)))
print('MPC with hard terminal constraints: ' + str(np.mean(res_steps_hard)))
# print('MPC with soft terminal constraints: ' + str(np.mean(res_steps_soft)))
# print('MPC with receding (hard + soft) constraints: ' + str(np.mean(res_steps_rec_hard)))
print('MPC with receding (soft + soft) constraints: ' + str(np.mean(res_steps_rec_soft)))

idx_no = np.where(res_steps_no != len(res_steps_no) - 1)[0]
idx_hard = np.where(res_steps_hard != len(res_steps_no) - 1)[0]
idx_rec_soft = np.where(res_steps_rec_soft != len(res_steps_no) - 1)[0]

cost_no = mean_cost(np.asarray(data_no['x_traj'])[idx_no], x_ref, Q)
cost_hard = mean_cost(np.asarray(data_hard['x_traj'])[idx_hard], x_ref, Q)
cost_soft = mean_cost(data_soft['x_traj'], x_ref, Q)
cost_rec_hard = mean_cost(data_rec_hard['x_traj'], x_ref, Q)
cost_rec_soft = mean_cost(np.asarray(data_rec_soft['x_traj'])[idx_rec_soft], x_ref, Q)

print('#### Mean running cost ####')
print('Naive MPC: ' + str(cost_no))
print('MPC with hard terminal constraints: ' + str(cost_hard))
# print('MPC with soft terminal constraints: ' + str(cost_soft))
# print('MPC with receding (hard + soft) constraints: ' + str(cost_rec_hard))
print('MPC with receding (soft + soft) constraints: ' + str(cost_rec_soft))
