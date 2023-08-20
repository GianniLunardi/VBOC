import os
import sys
sys.path.insert(1, os.getcwd() + '/..')

import numpy as np
import time
from doublependulum_class_vboc import OCPdoublependulumSTD, SYMdoublependulum
from multiprocessing import Pool
from scipy.stats import qmc
import random
import matplotlib.pyplot as plt
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

def init_guess(p):

    x0 = np.zeros((ocp.ocp.dims.nx,))
    x0[:ocp.ocp.dims.nu] = data[p]

    # Guess:
    x_sol_guess = np.full((N+1, ocp.ocp.dims.nx), x0)
    u_sol_guess = np.zeros((N, ocp.ocp.dims.nu))
       
    status = ocp.OCP_solve(x0, x_sol_guess, u_sol_guess, ocp.thetamax-0.05, 0)

    if status == 0:

        for i in range(N):
            x_sol_guess[i] = ocp.ocp_solver.get(i, "x")
            u_sol_guess[i] = ocp.ocp_solver.get(i, "u")

        x_sol_guess[N] = ocp.ocp_solver.get(N, "x")

    return x_sol_guess, u_sol_guess

start_time = time.time()

cpu_num = 1
test_num = 100

time_step = 5*1e-3
tot_time = 0.16
tot_steps = 100

ocp = OCPdoublependulumSTD("SQP", time_step, 0.2, True)

# Generate low-discrepancy unlabeled samples:
sampler = qmc.Halton(d=ocp.ocp.dims.nu, scramble=False)
sample = sampler.random(n=test_num)
l_bounds = ocp.Xmin_limits[:ocp.ocp.dims.nu]
u_bounds = ocp.Xmax_limits[:ocp.ocp.dims.nu]
data = qmc.scale(sample, l_bounds, u_bounds)

N = ocp.ocp.dims.N

# joint_vec = np.array([random.choice(range(ocp.nu)) for _ in range(tot_steps)])
# np.save('../selected_joint', joint_vec)
# joint_vec = np.load('../selected_joint.npy')            #NOOOOOOOOOOOO

with Pool(30) as p:
    res = p.map(init_guess, range(data.shape[0]))

x_sol_guess_vec, u_sol_guess_vec = zip(*res)

np.save('../x_sol_guess', np.asarray(x_sol_guess_vec))
np.save('../u_sol_guess', np.asarray(u_sol_guess_vec))

noise_perc = 0
noise_vec = np.full((ocp.ocp.dims.nu,tot_steps), np.random.normal(0,noise_perc*ocp.Cmax/200,tot_steps))
noise_vec = noise_vec.reshape((tot_steps,ocp.ocp.dims.nu))
np.save('../noise', noise_vec)

del ocp

ocp = OCPdoublependulumSTD("SQP_RTI", time_step, tot_time, True)
sim = SYMdoublependulum(time_step, tot_time, True)

N = ocp.ocp.dims.N

# MPC controller without terminal constraints:
with Pool(cpu_num) as p:
    res = p.map(simulate, range(data.shape[0]))

res_steps, stats, x_traj, u_traj, failed = zip(*res)

times = np.array([i for f in stats for i in f ])
times = times[~np.isnan(times)]

print('99 percent quantile solve time: ' + str(np.quantile(times, 0.99)))
print('Mean solve time: ' + str(np.mean(times)))

print(np.array(res_steps).astype(int))
print(np.mean(res_steps))

np.save('res_steps_noconstr.npy', np.array(res_steps).astype(int))

end_time = time.time()
print('Elapsed time: ' + str(end_time-start_time))

# Compute the x_init
x_arr = np.asarray(x_traj)
res_arr = np.asarray(res_steps)
idx = np.where(res_arr != tot_steps - 1)[0]
x_init = x_arr[idx, res_arr[idx]]

# Save pickle file
data_dir = '../data_2dof/'
with open(data_dir + 'results_no_constraint.pickle', 'wb') as f:
    all_data = dict()
    all_data['times'] = times
    all_data['dt'] = time_step
    all_data['tot_time'] = tot_time
    all_data['res_steps'] = res_steps
    all_data['failed'] = failed
    all_data['x_init'] = x_init
    all_data['x_traj'] = x_traj
    all_data['u_traj'] = u_traj
    pickle.dump(all_data, f)
