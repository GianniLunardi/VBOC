import os
import sys
import matplotlib.pyplot as plt
sys.path.insert(1, os.getcwd() + '/..')
from scipy.stats import qmc
import numpy as np
from triplependulum_class_vboc import OCPtriplependulumSTD, SYMtriplependulum

cpu_num = 30
test_num = 100
time_step = 5*1e-3
tot_time = 0.18 #- time_step
tot_steps = 100
t = np.arange(0, tot_time, time_step)
nx = 6

regenerate = True
sim = SYMtriplependulum(time_step, tot_time, regenerate)
ocp = OCPtriplependulumSTD("SQP", time_step, tot_time, regenerate)
N = ocp.ocp.dims.N

sampler = qmc.Halton(d=ocp.ocp.dims.nu, scramble=False)
sample = sampler.random(n=test_num)
l_bounds = ocp.Xmin_limits[:ocp.ocp.dims.nu]
u_bounds = ocp.Xmax_limits[:ocp.ocp.dims.nu]
data = qmc.scale(sample, l_bounds, u_bounds)

def init_guess(k):

    x0 = np.zeros((ocp.ocp.dims.nx,))
    x0[:ocp.ocp.dims.nu] = data[k]

    # Guess:
    x_sol_guess = np.full((N + 1, ocp.ocp.dims.nx), x0) 
    u_sol_guess = np.zeros((N, ocp.ocp.dims.nu)) 
    # x_sol_guess, u_sol_guess = create_guess(sim, ocp, x0, x_ref)

    status = ocp.OCP_solve(x0, x_sol_guess, u_sol_guess, ocp.thetamax - 0.05, 0)
    success = 0

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

    else:
        print('Solver failed: status ' + str(status))

    return x_sol_guess, u_sol_guess, success

def simulate(x0, u):
    x_sim = np.empty((N + 1, nx)) * np.nan
    x_sim[0] = x0

    for j in range(N):
        sim.acados_integrator.set("u", u[j])
        sim.acados_integrator.set("x", x_sim[j])
        sim.acados_integrator.solve()
        x_sim[j+1] = sim.acados_integrator.get("x")
    return x_sim


def plot_trajectory(x, u, x_sim):
    fig, ax = plt.subplots(3, 1, sharex='col')
    for i in range(3):
        ax[i].plot(t, ocp.thetamax * np.ones_like(t), color='red', linestyle='--')
        ax[i].plot(t, ocp.thetamin * np.ones_like(t), color='green', linestyle='--')
    ax[0].plot(t, x[:-1, 0], label='q1')
    # ax[0].plot(t, u[:, 0], label='u1', color='g')
    ax[0].plot(t, x_sim[:-1, 0], label='q1_sim', linestyle='--')
    ax[0].legend()
    ax[1].plot(t, x[:-1, 1], label='q2')
    # ax[1].plot(t, u[:, 1], label='u2', color='g')
    ax[1].plot(t, x_sim[:-1, 1], label='q2_sim', linestyle='--')
    ax[1].legend()
    ax[2].plot(t, x[:-1, 2], label='q3')
    # ax[2].plot(t, u[:, 2], label='u3', color='g')
    ax[2].plot(t, x_sim[:-1, 2], label='q3_sim', linestyle='--')
    ax[2].legend()
    ax[2].set_xlabel('time [s]')


# x_ocp = np.load('x_sol_guess_viable_2.npy')
# u_ocp = np.load('u_sol_guess_viable_2.npy')

res = []
for k in range(test_num):
    res.append(init_guess(k))
x_ocp, u_ocp, success = zip(*res)

for j in range(test_num):
    plot_trajectory(x_ocp[j], u_ocp[j], simulate(x_ocp[j][0,:], u_ocp[j]))
    plt.savefig('plots/test_integration' + str(j) + '.png')
    plt.close()