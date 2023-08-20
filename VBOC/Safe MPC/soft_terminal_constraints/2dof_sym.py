import os
import sys
sys.path.insert(1, os.getcwd() + '/..')

import numpy as np
import time
import torch
from doublependulum_class_vboc import OCPdoublependulumSoftTerm, SYMdoublependulum
from my_nn import NeuralNetDIR
from multiprocessing import Pool
from scipy.stats import qmc
import pickle

import warnings
warnings.filterwarnings("ignore")

def simulate(p):

    x0 = np.zeros((ocp.ocp.dims.nx,))
    x0[:ocp.ocp.dims.nu] = data[p]

    simX = np.empty((tot_steps + 1, ocp.ocp.dims.nx)) * np.nan
    simU = np.empty((tot_steps, ocp.ocp.dims.nu)) * np.nan
    simX[0] = np.copy(x0)

    times = np.empty(tot_steps) * np.nan

    failed_iter = 0
    failed_tot = 0

    # Guess:
    x_sol_guess = x_sol_guess_vec[p]
    u_sol_guess = u_sol_guess_vec[p]

    for f in range(tot_steps):
       
        status = ocp.OCP_solve(simX[f], x_sol_guess, u_sol_guess, ocp.thetamax-0.05, 0)
        times[f] = ocp.ocp_solver.get_stats('time_tot')

        if status != 0:

            if failed_iter >= N-1 or failed_iter < 0:
                break

            failed_iter += 1
            failed_tot += 1

            simU[f] = u_sol_guess[0]

            for i in range(N-1):
                x_sol_guess[i] = np.copy(x_sol_guess[i+1])
                u_sol_guess[i] = np.copy(u_sol_guess[i+1])

            x_sol_guess[N-1] = np.copy(x_sol_guess[N])

        else:
            failed_iter = 0

            simU[f] = ocp.ocp_solver.get(0, "u")

            for i in range(N-1):
                x_sol_guess[i] = ocp.ocp_solver.get(i+1, "x")
                u_sol_guess[i] = ocp.ocp_solver.get(i+1, "u")

            x_sol_guess[N-1] = ocp.ocp_solver.get(N, "x")
            x_sol_guess[N] = np.copy(x_sol_guess[N-1])
            u_sol_guess[N-1] = np.copy(u_sol_guess[N-2])

        simU[f] += noise_vec[f]

        sim.acados_integrator.set("u", simU[f])
        sim.acados_integrator.set("x", simX[f])
        status = sim.acados_integrator.solve()
        simX[f+1] = sim.acados_integrator.get("x")

    return f, times, simX, simU, failed_tot

start_time = time.time()

# Pytorch params:
device = torch.device("cpu") 

model = NeuralNetDIR(4, 300, 1).to(device)
model.load_state_dict(torch.load('../model_2dof_vboc'))
mean = torch.load('../mean_2dof_vboc')
std = torch.load('../std_2dof_vboc')
safety_margin = 2.0

cpu_num = 5
test_num = 100

time_step = 5*1e-3
tot_time = 0.16 - 4*time_step
tot_steps = 100

regenerate = False

x_sol_guess_vec = np.load('../x_sol_guess.npy')
u_sol_guess_vec = np.load('../u_sol_guess.npy')
noise_vec = np.load('../noise.npy')
# joint_vec = np.load('../selected_joint.npy')
#
# quant = 10.
# r = 1
#
# while quant > time_step - 1e-3:

ocp = OCPdoublependulumSoftTerm("SQP_RTI", time_step, tot_time, list(model.parameters()), mean, std, safety_margin, regenerate)
sim = SYMdoublependulum(time_step, tot_time, True)

N = ocp.ocp.dims.N

# Generate low-discrepancy unlabeled samples:
sampler = qmc.Halton(d=ocp.ocp.dims.nu, scramble=False)
sample = sampler.random(n=test_num)
l_bounds = ocp.Xmin_limits[:ocp.ocp.dims.nu]
u_bounds = ocp.Xmax_limits[:ocp.ocp.dims.nu]
data = qmc.scale(sample, l_bounds, u_bounds)

ocp.ocp_solver.cost_set(N, "Zl", 1e4*np.ones((1,)))

# MPC controller without terminal constraints:
with Pool(cpu_num) as p:
    res = p.map(simulate, range(data.shape[0]))

res_steps_term, stats, x_traj, u_traj, failed = zip(*res)

times = np.array([i for f in stats for i in f ])
times = times[~np.isnan(times)]

quant = np.quantile(times, 0.99)

# print('iter: ', str(r))
print('tot time: ' + str(tot_time))
print('99 percent quantile solve time: ' + str(quant))
print('Mean solve time: ' + str(np.mean(times)))

# tot_time -= time_step
# r += 1

print(np.array(res_steps_term).astype(int))

del ocp

np.save('res_steps_softterm.npy', np.array(res_steps_term).astype(int))

res_steps = np.load('../no_constraints/res_steps_noconstr.npy')

better = 0
equal = 0
worse = 0

for i in range(res_steps.shape[0]):
    if res_steps_term[i]-res_steps[i]>0:
        better += 1
    elif res_steps_term[i]-res_steps[i]==0:
        equal += 1
    else:
        worse += 1

print('MPC standard vs MPC with soft term constraints')
print('Percentage of initial states in which the MPC+VBOC behaves better: ' + str(better))
print('Percentage of initial states in which the MPC+VBOC behaves equal: ' + str(equal))
print('Percentage of initial states in which the MPC+VBOC behaves worse: ' + str(worse))

end_time = time.time()
print('Elapsed time: ' + str(end_time-start_time))

# Compute the x_init
x_arr = np.asarray(x_traj)
res_arr = np.asarray(res_steps_term)
idx = np.where(res_arr != tot_steps - 1)[0]
x_init = x_arr[idx, res_arr[idx]]

# Save pickle file
data_dir = '../data_2dof/'
with open(data_dir + 'results_softterm.pickle', 'wb') as f:
    all_data = dict()
    all_data['times'] = times
    all_data['dt'] = time_step
    all_data['tot_time'] = tot_time
    all_data['res_steps'] = res_steps_term
    all_data['failed'] = failed
    all_data['x_init'] = x_init
    all_data['x_traj'] = x_traj
    all_data['u_traj'] = u_traj
    all_data['better'] = better
    all_data['worse'] = worse
    all_data['equal'] = equal
    pickle.dump(all_data, f)