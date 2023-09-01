import os
import sys
sys.path.insert(1, os.getcwd() + '/..')

import numpy as np
import time
import torch
from triplependulum_class_vboc import OCPtriplependulumReceidingSoft, SYMtriplependulum, nn_decisionfunction_conservative
from my_nn import NeuralNetDIR
from multiprocessing import Pool
import scipy.linalg as lin
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

    receiding = 0
    sanity_check = 0
    x_rec = np.copy(x0)

    for f in range(tot_steps):

        receiding = 0

        if failed_iter == 0 and f > 0:
            for i in range(1, N+1):
                if nn_decisionfunction_conservative(params, mean, std, safety_margin, ocp.ocp_solver.get(i, 'x')) >= 0.:
                    receiding = N - i + 1
                    x_rec = np.copy(ocp.ocp_solver.get(i, 'x'))

        receiding_iter = N-failed_iter-receiding

        for i in range(1, N):
            if i == receiding_iter:
                ocp.ocp_solver.cost_set(i, "Zl", 1e7*np.ones((1,)))
                if nn_decisionfunction_conservative(params, mean, std, safety_margin, ocp.ocp_solver.get(i, 'x')) < 0. and failed_iter == 0:
                    sanity_check += 1
            else:
                ocp.ocp_solver.cost_set(i, "Zl", np.zeros((1,)))
       
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

    return f, times, simX, simU, sanity_check, failed_tot, x_rec

start_time = time.time()

# Pytorch params:
device = torch.device("cpu") 

model = NeuralNetDIR(6, 500, 1).to(device)
model.load_state_dict(torch.load('../model_3dof_vboc'))
mean = torch.load('../mean_3dof_vboc')
std = torch.load('../std_3dof_vboc')
safety_margin = 5.0

cpu_num = 12
test_num = 100

time_step = 5*1e-3
tot_time = 0.18 - 2 * time_step
tot_steps = 100

regenerate = True

x_sol_guess_vec = np.load('../x_sol_guess.npy')
u_sol_guess_vec = np.load('../u_sol_guess.npy')
noise_vec = np.load('../noise.npy')
joint_vec = np.load('../selected_joint.npy')

params = list(model.parameters())

ocp = OCPtriplependulumReceidingSoft("SQP_RTI", time_step, tot_time, params, mean, std, safety_margin, regenerate)
sim = SYMtriplependulum(time_step, tot_time, regenerate)

# Generate low-discrepancy unlabeled samples:
sampler = qmc.Halton(d=ocp.ocp.dims.nu, scramble=False)
sample = sampler.random(n=test_num)
l_bounds = ocp.Xmin_limits[:ocp.ocp.dims.nu]
u_bounds = ocp.Xmax_limits[:ocp.ocp.dims.nu]
data = qmc.scale(sample, l_bounds, u_bounds)

N = ocp.ocp.dims.N

ocp.ocp_solver.cost_set(N, "Zl", 1e4*np.ones((1,)))

# MPC controller without terminal constraints:
with Pool(cpu_num) as p:
    res = p.map(simulate, range(data.shape[0]))

res_steps_traj, stats, x_traj, u_traj, sanity, failed, x_rec = zip(*res)

times = np.array([i for f in stats for i in f ])
times = times[~np.isnan(times)]

quant = np.quantile(times, 0.99)

print('tot time: ' + str(tot_time))
print('99 percent quantile solve time: ' + str(quant))
print('Mean solve time: ' + str(np.mean(times)))

print(np.array(res_steps_traj).astype(int))
print("Sanity check: ")
print(np.array(sanity).astype(int))

np.save('res_steps_receiding.npy', np.array(res_steps_traj).astype(int))

res_steps = np.load('../no_constraints/res_steps_noconstr.npy')

better = 0
equal = 0
worse = 0

for i in range(res_steps.shape[0]):
    if res_steps_traj[i]-res_steps[i]>0:
        better += 1
    elif res_steps_traj[i]-res_steps[i]==0:
        equal += 1
    else:
        worse += 1

print('MPC standard vs MPC with receding constraints (soft receding + soft terminal)')
print('Percentage of initial states in which the MPC+VBOC behaves better: ' + str(better))
print('Percentage of initial states in which the MPC+VBOC behaves equal: ' + str(equal))
print('Percentage of initial states in which the MPC+VBOC behaves worse: ' + str(worse))

# np.savez('../data/results_receiding_softsoft.npz', res_steps_term=res_steps_traj,
#          better=better, worse=worse, equal=equal, times=times,
#          dt=time_step, tot_time=tot_time)

end_time = time.time()
print('Elapsed time: ' + str(end_time-start_time))

# Remove all the x_rec in the case in which the full MPC succeeds
res_arr = np.array(res_steps_traj)
idx = np.where(res_arr != tot_steps - 1)[0]
x_init = np.asarray(x_rec)[idx]

# Save pickle file
data_dir = '../data_3dof/'
with open(data_dir + 'results_receiding_softsoft.pkl', 'wb') as f:
    all_data = dict()
    all_data['times'] = times
    all_data['dt'] = time_step
    all_data['tot_time'] = tot_time
    all_data['res_steps'] = res_steps_traj
    all_data['failed'] = failed
    all_data['x_init'] = x_init
    all_data['sanity'] = sanity
    all_data['x_traj'] = x_traj
    all_data['u_traj'] = u_traj
    all_data['better'] = better
    all_data['worse'] = worse
    all_data['equal'] = equal
    pickle.dump(all_data, f)