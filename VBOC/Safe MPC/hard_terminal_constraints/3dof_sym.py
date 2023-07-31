import os
import sys
sys.path.insert(1, os.getcwd() + '/..')

import numpy as np
import time
import torch
from triplependulum_class_vboc import OCPtriplependulumHardTerm, SYMtriplependulum
from my_nn import NeuralNetDIR
from multiprocessing import Pool
from scipy.stats import qmc

import warnings
warnings.filterwarnings("ignore")

def simulate(p):

    x0 = np.zeros((ocp.ocp.dims.nx,))
    x0[:ocp.ocp.dims.nu] = data[p]

    simX = np.ndarray((tot_steps + 1, ocp.ocp.dims.nx))
    simU = np.ndarray((tot_steps, ocp.ocp.dims.nu))
    simX[0] = np.copy(x0)

    times = [None] * tot_steps

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

    return f, times

start_time = time.time()

# Pytorch params:
device = torch.device("cpu") 

model = NeuralNetDIR(6, 500, 1).to(device)
model.load_state_dict(torch.load('../model_3dof_vboc'))
mean = torch.load('../mean_3dof_vboc')
std = torch.load('../std_3dof_vboc')
safety_margin = 2.0

cpu_num = 1
test_num = 100

time_step = 4*1e-3
tot_time = 0.16 #0.1 and 0.115 0.002s, 0.15 0.027s, 0.2 0.003s, 0.25 0.0035s, 0.3 0.0039s
tot_steps = 100

regenerate = True

x_sol_guess_vec = np.load('../x_sol_guess.npy')
u_sol_guess_vec = np.load('../u_sol_guess.npy')

quant = 10.
r = 1

while quant > 3*1e-3:

    ocp = OCPtriplependulumHardTerm("SQP_RTI", time_step, tot_time, list(model.parameters()), mean, std, regenerate)
    sim = SYMtriplependulum(time_step, tot_time, True)

    N = ocp.ocp.dims.N

    # Generate low-discrepancy unlabeled samples:
    sampler = qmc.Halton(d=ocp.ocp.dims.nu, scramble=False)
    sample = sampler.random(n=test_num)
    l_bounds = ocp.Xmin_limits[:ocp.ocp.dims.nu]
    u_bounds = ocp.Xmax_limits[:ocp.ocp.dims.nu]
    data = qmc.scale(sample, l_bounds, u_bounds)

    # MPC controller without terminal constraints:
    with Pool(cpu_num) as p:
        res = p.map(simulate, range(data.shape[0]))

    res_steps_term, stats = zip(*res)

    times = np.array([i for f in stats for i in f if i is not None])

    quant = np.quantile(times, 0.9)

    print('iter: ', str(r))
    print('tot time: ' + str(tot_time))
    print('90 percent quantile solve time: ' + str(quant))
    print('Mean solve time: ' + str(np.mean(times)))

    tot_time -= 2*1e-2
    r += 1

print(np.array(res_steps_term).astype(int))

np.save('res_steps_hardterm.npy', np.array(res_steps_term).astype(int))

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

print('MPC standard vs MPC with hard term constraints')
print('Percentage of initial states in which the MPC+VBOC behaves better: ' + str(better))
print('Percentage of initial states in which the MPC+VBOC behaves equal: ' + str(equal))
print('Percentage of initial states in which the MPC+VBOC behaves worse: ' + str(worse))