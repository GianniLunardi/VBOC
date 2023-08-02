import os
import sys
sys.path.insert(1, os.getcwd() + '/..')

import numpy as np
import time
from triplependulum_class_vboc import OCPtriplependulumSTD, SYMtriplependulum
from multiprocessing import Pool
from scipy.stats import qmc
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

def simulate(p):

    x0 = np.zeros((ocp.ocp.dims.nx,))
    x0[:ocp.ocp.dims.nu] = data[p]

    simX = np.ndarray((tot_steps + 1, ocp.ocp.dims.nx))
    simU = np.ndarray((tot_steps, ocp.ocp.dims.nu))
    simX[0] = np.copy(x0)

    times = [None] * tot_steps
    times_cpu = [None] * tot_steps

    failed_iter = -1

    # Guess:
    x_sol_guess = x_sol_guess_vec[p]
    u_sol_guess = u_sol_guess_vec[p]

    for f in range(tot_steps):
       
        temp = time.time()
        status = ocp.OCP_solve(simX[f], x_sol_guess, u_sol_guess)
        times[f] = time.time() - temp

        if status != 0:

            if failed_iter >= N-1 or failed_iter < 0:
                break

            failed_iter += 1

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

        sim.acados_integrator.set("u", simU[f])
        sim.acados_integrator.set("x", simX[f])
        status = sim.acados_integrator.solve()
        simX[f+1] = sim.acados_integrator.get("x")
        simU[f] = u_sol_guess[0]
        # print('Iteration: %d' % f)
        # print('Time of linearization: %.6f' % ocp.ocp_solver.get_stats('time_lin'))
        # print('Time integration (contribution external calls): %.6f' % ocp.ocp_solver.get_stats('time_sim_ad'))
        # print('Time integration: %.6f' % ocp.ocp_solver.get_stats('time_sim'))
        # print('Time QP: %.6f' % ocp.ocp_solver.get_stats('time_qp'))
        # print('Total time: %.6f' % ocp.ocp_solver.get_stats('time_tot'))
        times_cpu[f] = ocp.ocp_solver.get_stats('time_tot')

    return f, times, times_cpu

def init_guess(p):

    x0 = np.zeros((ocp.ocp.dims.nx,))
    x0[:ocp.ocp.dims.nu] = data[p]

    # Guess:
    x_sol_guess = np.full((N+1, ocp.ocp.dims.nx), x0)
    u_sol_guess = np.zeros((N, ocp.ocp.dims.nu))
       
    status = ocp.OCP_solve(x0, x_sol_guess, u_sol_guess)

    if status == 0:

        for i in range(N):
            x_sol_guess[i] = ocp.ocp_solver.get(i, "x")
            u_sol_guess[i] = ocp.ocp_solver.get(i, "u")

        x_sol_guess[N] = ocp.ocp_solver.get(N, "x")

    return x_sol_guess, u_sol_guess

start_time = time.time()

cpu_num = 1
test_num = 100

time_step = 4*1e-3
tot_time = 0.148
tot_steps = 100

ocp = OCPtriplependulumSTD("SQP", time_step, 0.2, True)

# Generate low-discrepancy unlabeled samples:
sampler = qmc.Halton(d=ocp.ocp.dims.nu, scramble=False)
sample = sampler.random(n=test_num)
l_bounds = ocp.Xmin_limits[:ocp.ocp.dims.nu]
u_bounds = ocp.Xmax_limits[:ocp.ocp.dims.nu]
data = qmc.scale(sample, l_bounds, u_bounds)

N = ocp.ocp.dims.N

with Pool(30) as p:
    res = p.map(init_guess, range(data.shape[0]))

x_sol_guess_vec, u_sol_guess_vec = zip(*res)

np.save('../x_sol_guess', np.asarray(x_sol_guess_vec))
np.save('../u_sol_guess', np.asarray(u_sol_guess_vec))

del ocp

ocp = OCPtriplependulumSTD("SQP_RTI", time_step, tot_time, True)
sim = SYMtriplependulum(time_step, tot_time, True)

N = ocp.ocp.dims.N

# MPC controller without terminal constraints:
with Pool(cpu_num) as p:
    res = p.map(simulate, range(data.shape[0]))

res_steps, stats, stats_cpu = zip(*res)

times = np.array([i for f in stats for i in f if i is not None])
times_cpu = np.array([i for f in stats_cpu for i in f if i is not None])

print('90 percent quantile solve time: ' + str(np.quantile(times, 0.9)))
print('90 percent quantile CPU time: ' + str(np.quantile(times_cpu, 0.9)))
print('Mean solve time: ' + str(np.mean(times)))
print('Standard deviation of solve time: %.6f' % np.std(times))

print(np.array(res_steps).astype(int))

# np.save('res_steps_noconstr.npy', np.array(res_steps).astype(int))

# Plot timing
# plt.figure()
# plt.plot(np.linspace(0, len(times), len(times)), times)
# plt.xlabel('Iteration')
# plt.ylabel('Solve time [s]')
# plt.show()

# Save the results in an npz file
# np.savez('../data/results_no_constraint.npz', times=times,
#          dt=time_step, tot_time=tot_time, res_steps=res_steps)