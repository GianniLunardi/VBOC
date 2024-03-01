import numpy as np
import random 
from numpy.linalg import norm as norm
from VBOC.triplependulum_class_vboc import OCPtriplependulumINIT
import warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pool

def testing(v):
    # Reset the number of steps used in the OCP:
    N = ocp.N
    ocp.ocp_solver.set_new_time_steps(np.full((N,), 1.))
    ocp.ocp_solver.update_qp_solver_cond_N(N)

    # Time step duration:
    dt_sym = 1e-2

    # Initial velocity optimization direction (the cost is in the form p[0] * dtheta1 + p[1] * dtheta2 + p[2] * dt):
    ran1 = random.choice([-1, 1]) * random.random()
    ran2 = random.choice([-1, 1]) * random.random() 
    ran3 = random.choice([-1, 1]) * random.random() 
    norm_weights = norm(np.array([ran1, ran2, ran3]))         
    p = np.array([ran1/norm_weights, ran2/norm_weights, ran3/norm_weights, 0.])

    # Bounds on the initial state:
    q_init_1 = q_min + random.random() * (q_max-q_min)
    q_init_2 = q_min + random.random() * (q_max-q_min)
    q_init_3 = q_min + random.random() * (q_max-q_min)
    q_init_lb = np.array([q_init_1, q_init_2, q_init_3, v_min, v_min, v_min, dt_sym])
    q_init_ub = np.array([q_init_1, q_init_2, q_init_3, v_max, v_max, v_max, dt_sym])

    # Bounds on the final state:
    q_fin_lb = np.array([q_min, q_min, q_min, 0., 0., 0., dt_sym])
    q_fin_ub = np.array([q_max, q_max, q_max, 0., 0., 0., dt_sym])

    # Guess:
    x_sol_guess = np.full((N, 7), np.array([q_init_1, q_init_2, q_init_3, 0., 0., 0., dt_sym]))
    u_sol_guess = np.full((N, 3), np.array([0.,0.,0.]))

    # Iteratively solve the OCP with an increased number of time steps until the solution converges:
    cost = 1e6
    while True:
        # Reset current iterate:
        ocp.ocp_solver.reset()

        # Set parameters, guesses and constraints:
        for i in range(N):
            ocp.ocp_solver.set(i, 'x', x_sol_guess[i])
            ocp.ocp_solver.set(i, 'u', u_sol_guess[i])
            ocp.ocp_solver.set(i, 'p', p)
            ocp.ocp_solver.constraints_set(i, 'lbx', np.array([q_min, q_min, q_min, v_min, v_min, v_min, dt_sym])) 
            ocp.ocp_solver.constraints_set(i, 'ubx', np.array([q_max, q_max, q_max, v_max, v_max, v_max, dt_sym])) 
            ocp.ocp_solver.constraints_set(i, 'lbu', np.array([-tau_max, -tau_max, -tau_max]))
            ocp.ocp_solver.constraints_set(i, 'ubu', np.array([tau_max, tau_max, tau_max]))
            ocp.ocp_solver.constraints_set(i, 'C', np.zeros((3,7)))
            ocp.ocp_solver.constraints_set(i, 'D', np.zeros((3,3)))
            ocp.ocp_solver.constraints_set(i, 'lg', np.zeros((3)))
            ocp.ocp_solver.constraints_set(i, 'ug', np.zeros((3)))

        C = np.zeros((3,7))
        d = np.array([p[:3].tolist()])
        dt = np.transpose(d)
        C[:,3:6] = np.identity(3)-np.matmul(dt,d) # np.identity(3)-np.matmul(np.matmul(dt,np.linalg.inv(np.matmul(d,dt))),d)
        ocp.ocp_solver.constraints_set(0, "C", C, api='new') 

        ocp.ocp_solver.constraints_set(0, "lbx", q_init_lb)
        ocp.ocp_solver.constraints_set(0, "ubx", q_init_ub)

        ocp.ocp_solver.constraints_set(N, "lbx", np.array([q_min, q_min, q_min, 0., 0., 0., dt_sym]))
        ocp.ocp_solver.constraints_set(N, "ubx", np.array([q_max, q_max, q_max, 0., 0., 0., dt_sym]))
        ocp.ocp_solver.set(N, 'x', x_sol_guess[-1])
        ocp.ocp_solver.set(N, 'p', p)

        # Solve the OCP:
        status = ocp.ocp_solver.solve()

        # If the solver finds a solution, compare it with the previous. If the cost has decresed, keep increasing N, alternatively keep increasing N.
        # If the solver fails, reinitialize N and restart the iterations with slight different initial conditions.
        if status == 0: 
            # Compare the current cost with the previous:
            cost_new = ocp.ocp_solver.get_cost()
            if cost_new > float(f'{cost:.3f}') - 1e-3:
                break
            cost = cost_new

            # Update the guess with the current solution:
            x_sol_guess = np.empty((N+1,7))
            u_sol_guess = np.empty((N+1,3))
            for i in range(N):
                x_sol_guess[i] = ocp.ocp_solver.get(i, "x")
                u_sol_guess[i] = ocp.ocp_solver.get(i, "u")
            x_sol_guess[N] = ocp.ocp_solver.get(N, "x")
            u_sol_guess[N] = np.array([0.,0.,0.])

            # Increase the number of time steps:
            N = N + 1
            ocp.ocp_solver.set_new_time_steps(np.full((N,), 1.))
            ocp.ocp_solver.update_qp_solver_cond_N(N)
        else:
            # Reset the number of steps used in the OCP:
            N = ocp.N
            ocp.ocp_solver.set_new_time_steps(np.full((N,), 1.))
            ocp.ocp_solver.update_qp_solver_cond_N(N)

            # Initial velocity optimization direction:
            ran1 = ran1 + random.random() * random.choice([-1, 1]) * 0.01
            ran2 = ran2 + random.random() * random.choice([-1, 1]) * 0.01
            ran3 = ran3 + random.random() * random.choice([-1, 1]) * 0.01
            norm_weights = norm(np.array([ran1, ran2, ran3]))         
            p = np.array([ran1/norm_weights, ran2/norm_weights, ran3/norm_weights, 0.])
            
            # Bounds on the initial state:
            q_init_1 = q_init_1 + random.random() * random.choice([-1, 1]) * 0.01
            q_init_2 = q_init_2 + random.random() * random.choice([-1, 1]) * 0.01
            q_init_3 = q_init_3 + random.random() * random.choice([-1, 1]) * 0.01
            q_init_lb = np.array([q_init_1, q_init_2, q_init_3, v_min, v_min, v_min, dt_sym])
            q_init_ub = np.array([q_init_1, q_init_2, q_init_3, v_max, v_max, v_max, dt_sym])

            # Guess:
            x_sol_guess = np.full((N, 7), np.array([q_init_1, q_init_2, q_init_3, 0., 0., 0., dt_sym]))
            u_sol_guess = np.full((N, 3), np.array([0.,0.,0.]))

            cost = 1e6

    return ocp.ocp_solver.get(0, "x")[:6]

# Ocp initialization:
ocp = OCPtriplependulumINIT()

# Position, velocity and torque bounds:
v_max = ocp.dthetamax
v_min = - ocp.dthetamax
q_max = ocp.thetamax
q_min = ocp.thetamin
tau_max = ocp.Cmax

# Test data generation:
cpu_num = 30
num_prob = 1000

with Pool(cpu_num) as p:
    data = p.map(testing, range(num_prob))

X_test = np.array(data)
np.save('data3_test.npy', X_test)
