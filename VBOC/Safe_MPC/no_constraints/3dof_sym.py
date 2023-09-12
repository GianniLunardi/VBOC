import os
import sys
sys.path.insert(1, os.getcwd() + '/..')

import numpy as np
import time
from triplependulum_class_vboc import OCPtriplependulumSTD, SYMtriplependulum, create_guess
from scipy.stats import qmc
from multiprocessing import Pool
import pickle

import warnings
warnings.filterwarnings("ignore")


def simulate(p):

    x0 = np.zeros((ocp.ocp.dims.nx,))
    x0[:ocp.ocp.dims.nu] = x0_vec[p]

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

        u0 = ocp.ocp_solver.get(0, "u")
        sim.acados_integrator.set("u", u0)
        sim.acados_integrator.set("x", simX[f])
        sim.acados_integrator.solve()
        x1 = sim.acados_integrator.get("x")

        # Check if the solution is infeasible: 1) solver status, 2) u0 outside U, 3) x1 outside X
        if status != 0 or np.all((u0 <= ocp.Cmin_limits) | (u0 >= ocp.Cmax_limits)) or \
                np.all((x1 <= ocp.Xmin_limits) | (x1 >= ocp.Xmax_limits)):

            if failed_iter >= N:
                break

            failed_iter += 1
            failed_tot += 1
            simU[f] = u_sol_guess[0]
            x_sol_guess = np.roll(x_sol_guess, -1, axis=0)
            u_sol_guess = np.roll(u_sol_guess, -1, axis=0)

            sim.acados_integrator.set("u", simU[f])
            sim.acados_integrator.set("x", simX[f])
            sim.acados_integrator.solve()
            simX[f+1] = sim.acados_integrator.get("x")

        else:
            failed_iter = 0
            simU[f] = u0
            simX[f+1] = x1

            for i in range(N-1):
                x_sol_guess[i] = ocp.ocp_solver.get(i+1, "x")
                u_sol_guess[i] = ocp.ocp_solver.get(i+1, "u")

            x_sol_guess[N-1] = ocp.ocp_solver.get(N, "x")

        # Copy the last elements of the guess
        x_sol_guess[-1] = np.copy(x_sol_guess[-2])
        u_sol_guess[-1] = np.copy(u_sol_guess[-2])

    return f, times, simX, simU, failed_tot

def init_guess(p):

    x0 = np.zeros((ocp.ocp.dims.nx,))
    x0[:ocp.ocp.dims.nu] = x0_vec[p]

    # Create initial guess
    x_sol_guess, u_sol_guess = create_guess(sim, ocp, x0, x_ref)
       
    status = ocp.OCP_solve(x0, x_sol_guess, u_sol_guess, ocp.thetamax-0.05, 0)
    success = 0
    solver_fails = 0

    if status == 0 or status == 2:

        for i in range(N):
            x_sol_guess[i] = ocp.ocp_solver.get(i, "x")
            u_sol_guess[i] = ocp.ocp_solver.get(i, "u")

        x_sol_guess[N] = ocp.ocp_solver.get(N, "x")

        if np.all((x_sol_guess >= ocp.Xmin_limits) & (x_sol_guess <= ocp.Xmax_limits)) and \
           np.all((u_sol_guess >= ocp.Cmin_limits) & (u_sol_guess <= ocp.Cmax_limits)):
            success = 1
        else:
            success = 0
            x_sol_guess = np.full((N + 1, ocp.ocp.dims.nx), x0)
            u_sol_guess = np.zeros((N, ocp.ocp.dims.nu))

    else:
        solver_fails = 1

    return x_sol_guess, u_sol_guess, success, solver_fails

start_time = time.time()
cpu_num = 1
time_step = 5*1e-3
tot_time = 0.18
tot_steps = 100

folder = '../viable_init/'
x0_vec = np.load(folder + 'initial_conditions.npy')
test_num = len(x0_vec)

sim = SYMtriplependulum(time_step, tot_time, True)
ocp = OCPtriplependulumSTD("SQP", time_step, tot_time, True)
x_ref = np.array([ocp.thetamax-0.05, np.pi, np.pi, 0, 0, 0])

N = ocp.ocp.dims.N

with Pool(30) as p:
    res = p.map(init_guess, range(test_num))

x_sol_guess_vec, u_sol_guess_vec, succ, fails = zip(*res)
print('Init guess success: ' + str(np.sum(succ)) + ' over ' + str(test_num))
print('Init guess fails: ' + str(np.sum(fails)) + ' over ' + str(test_num))

# np.save('../x_sol_guess', np.asarray(x_sol_guess_vec))
# np.save('../u_sol_guess', np.asarray(u_sol_guess_vec))

del ocp
ocp = OCPtriplependulumSTD("SQP_RTI", time_step, tot_time, True)
N = ocp.ocp.dims.N

# MPC controller without terminal constraints:
with Pool(cpu_num) as p:
    res = p.map(simulate, range(test_num))

res_steps, stats, x_traj, u_traj, failed = zip(*res)

times = np.array([i for f in stats for i in f ])
times = times[~np.isnan(times)]

print('99 percent quantile solve time: ' + str(np.quantile(times, 0.99)))
print(np.array(res_steps).astype(int))
print('Mean iterations: ' + str(np.mean(res_steps)))
np.save('res_steps_noconstr.npy', np.array(res_steps).astype(int))

end_time = time.time()
print('Elapsed time: ' + str(end_time-start_time))

# Compute the x_init
x_arr = np.asarray(x_traj)
res_arr = np.asarray(res_steps)
idx = np.where(res_arr != tot_steps - 1)[0]
x_init = x_arr[idx, res_arr[idx]]
print('Completed tasks: ' + str(100 - len(idx)) + ' over 100')

# Save pickle file
data_dir = '../data_3dof/'
with open(data_dir + 'results_no_constraint.pkl', 'wb') as f:
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
