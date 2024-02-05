import os
import sys
from tqdm import tqdm
sys.path.insert(1, os.getcwd() + '/..')
import numpy as np
import random
import matplotlib.pyplot as plt
from numpy.linalg import norm as norm
import time
from triplependulum_class_vboc import OCPtriplependulumINIT, SYMtriplependulumINIT
from doublependulum_class_vboc import OCPdoublependulumINIT, SYMdoublependulumINIT
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from my_nn import NeuralNetDIR
from multiprocessing import Pool
from torch.utils.data import DataLoader
from plots_2dof import plots_2dof
from plots_3dof import plots_3dof
import torch.nn.utils.prune as prune


def setHorizon(N):
    ocp.N = N
    ocp.ocp_solver.set_new_time_steps(np.full((N,), 1.))
    ocp.ocp_solver.update_qp_solver_cond_N(N)

def checkPositionBounds(q):
    return np.logical_or(np.any(q < q_min + eps), np.any(q > q_max - eps))

def checkVelocityBounds(v):
    return np.logical_or(np.any(v < v_min + eps), np.any(v > v_max - eps))

def checkStateBounds(x):
    return np.logical_or(np.any(x < x_min + eps), np.any(x > x_max - eps))

def data_generation(v):
    count = 0
    valid_data = np.ndarray((0, ocp.ocp.dims.nx - 1))
    # verify the dimension of each trajectory
    valid_traj = []

    # Reset the time parameters:
    N = N_start
    setHorizon(N)

    # Selection of the reference joint:
    joint_sel = random.choice(range(system_sel))

    # Selection of the start position of the reference joint:
    vel_sel = random.choice([-1, 1])  # -1 to maximise initial vel, + 1 to minimize it

    # Initial velocity optimization direction:
    p = np.random.uniform(-1, 1, system_sel + 1)
    p[-1] = 0
    p[joint_sel] = np.random.random() * vel_sel
    p /= norm(p)

    q_init_sel = q_min + eps if vel_sel == -1 else q_max - eps
    q_fin_sel = q_max - eps if vel_sel == -1 else q_min + eps

    # Bounds on the initial state:
    q_init_lb = np.random.uniform(q_min + eps, q_max - eps, ocp.ocp.dims.nx)
    # POS
    q_init_lb[joint_sel] = q_init_sel
    q_init_ub = np.copy(q_init_lb)
    # VEL 
    q_init_lb[system_sel:-1], q_init_ub[system_sel:-1] = v_min, v_max
    # dt
    q_init_lb[-1], q_init_ub[-1] = dt_sym, dt_sym

    # State and input bounds:
    q_lb = np.copy(q_init_lb)
    q_ub = np.copy(q_init_ub)

    q_lb[:system_sel], q_ub[:system_sel] = q_min, q_max

    u_lb = np.full((system_sel), -tau_max)
    u_ub = np.full((system_sel), tau_max)

    # Bounds on the final state:
    q_fin_lb = np.copy(q_lb)
    q_fin_ub = np.copy(q_ub)

    q_fin_lb[system_sel:-1], q_fin_ub[system_sel:-1] = 0., 0.

    x_sol_guess = np.zeros((N, ocp.ocp.dims.nx))
    u_sol_guess = np.zeros((N, ocp.ocp.dims.nu))
    x_sol_guess[:, :system_sel] = np.full((N, system_sel), q_init_lb[:system_sel])
    x_sol_guess[:, joint_sel] = np.linspace(q_init_sel, q_fin_sel, N)
    x_sol_guess[:, -1] = dt_sym

    cost_old = np.inf
    all_ok = False

    # Iteratively solve the OCP with an increased number of time steps until the solution does not change.
    # If the solver fails, try with a slightly different initial condition:
    while True:

        # Solve the OCP:
        status = ocp.OCP_solve(x_sol_guess, u_sol_guess, p, q_lb, q_ub, u_lb, u_ub, q_init_lb, q_init_ub, q_fin_lb,
                               q_fin_ub)

        if status == 0:  # the solver has found a solution

            # Compare the current cost with the previous one:
            cost_new = ocp.ocp_solver.get_cost()

            if cost_new > cost_old - tol:  # the time is sufficient to have achieved an optimal solution
                all_ok = True
                break

            cost_old = cost_new

            # Update the guess with the current solution:
            x_sol_guess = np.empty((N + 1, ocp.ocp.dims.nx))
            u_sol_guess = np.empty((N + 1, ocp.ocp.dims.nu))
            for i in range(N):
                x_sol_guess[i] = ocp.ocp_solver.get(i, "x")
                u_sol_guess[i] = ocp.ocp_solver.get(i, "u")
            x_sol_guess[N] = ocp.ocp_solver.get(N, "x")
            u_sol_guess[N] = np.zeros((ocp.ocp.dims.nu))

            # Increase the number of time steps:
            N += 1
            setHorizon(N)
        else:
            break

    if all_ok:

        # Save the optimal trajectory:
        x_sol = np.empty((N + 1, ocp.ocp.dims.nx))
        u_sol = np.empty((N, ocp.ocp.dims.nu))

        for i in range(N):
            x_sol[i] = ocp.ocp_solver.get(i, "x")
            u_sol[i] = ocp.ocp_solver.get(i, "u")

        x_sol[N] = ocp.ocp_solver.get(N, "x")

        # Generate the unviable sample in the cost direction:
        x_sym = np.full((N + 1, ocp.ocp.dims.nx - 1), None)

        x_out = np.copy(x_sol[0][:ocp.ocp.dims.nx - 1])
        x_out[system_sel:] -= eps * p[:system_sel]

        # save the initial state:
        valid_data = np.append(valid_data, [x_sol[0][:ocp.ocp.dims.nx - 1]], axis=0)
        valid_traj.append(x_sol)

        # Check if initial velocities lie on a limit:
        if checkVelocityBounds(x_out[system_sel:]):
            is_x_at_limit = True  # the state is on dX
        else:
            is_x_at_limit = False  # the state is on dV
            x_sym[0] = x_out

        # Iterate through the trajectory to verify the location of the states with respect to V:
        for f in range(1, N):

            if is_x_at_limit:

                # If the previous state was on a limit, the current state location cannot be identified using
                # the corresponding unviable state but it has to rely on the proximity to the state limits 
                # (more restrictive):
                if checkStateBounds(x_sol[f,:-1]):
                    is_x_at_limit = True  # the state is on dX
                else:
                    is_x_at_limit = False  # the state is either on the interior of V or on dV

                    # if the traj de-touches from a position limit it usually enters V:
                    # if any(i > q_max - eps or i < q_min + eps for i in x_sol[f - 1][:system_sel]):
                    if checkPositionBounds(x_sol[f - 1][:system_sel]):
                        break

                    # Solve an OCP to verify whether the following part of the trajectory is on V or dV. To do so
                    # the initial joint positions are set to the current ones and the final state is fixed to the
                    # final state of the trajectory. The initial velocities are left free and maximized in the 
                    # direction of the current joint velocities.

                    N_test = N - f
                    setHorizon(N_test)

                    # Cost:
                    p = np.zeros((system_sel + 1)) 
                    norm_weights = norm(x_sol[f][system_sel:-1])
                    p[:system_sel] = - x_sol[f][system_sel:-1] / norm_weights

                    # Bounds on the initial state:
                    for l in range(system_sel):
                        q_init_lb[l] = x_sol[f][l]
                        q_init_ub[l] = x_sol[f][l]

                    # Guess:
                    x_sol_guess = np.empty((N_test + 1, ocp.ocp.dims.nx))
                    u_sol_guess = np.empty((N_test + 1, ocp.ocp.dims.nu))
                    for i in range(N_test):
                        x_sol_guess[i] = x_sol[i + f]
                        u_sol_guess[i] = u_sol[i + f]
                    x_sol_guess[N_test] = x_sol[N]
                    u_sol_guess[N_test] = np.zeros(ocp.ocp.dims.nu)

                    norm_old = norm_weights  # velocity norm of the original solution
                    norm_bef = 0
                    all_ok = False

                    for _ in range(5):              # TODO: should be a while True, but it takes too long 

                        # Solve the OCP:
                        status = ocp.OCP_solve(x_sol_guess, u_sol_guess, p, q_lb, q_ub, u_lb, u_ub, q_init_lb,
                                               q_init_ub, q_fin_lb, q_fin_ub)

                        if status == 0:  # the solver has found a solution

                            # Compare the current cost with the previous one:
                            x0_new = ocp.ocp_solver.get(0, "x")
                            norm_new = norm(x0_new[system_sel:-1])

                            if norm_new < norm_bef + tol:
                                all_ok = True
                                break

                            norm_bef = norm_new

                            # Update the guess with the current solution:
                            x_sol_guess = np.empty((N_test + 1, ocp.ocp.dims.nx))
                            u_sol_guess = np.empty((N_test + 1, ocp.ocp.dims.nu))
                            for i in range(N_test):
                                x_sol_guess[i] = ocp.ocp_solver.get(i, "x")
                                u_sol_guess[i] = ocp.ocp_solver.get(i, "u")
                            x_sol_guess[N_test] = ocp.ocp_solver.get(N_test, "x")
                            u_sol_guess[N_test] = np.zeros(ocp.ocp.dims.nu)

                            # Increase the number of time steps:
                            N_test += 1
                            setHorizon(N_test)
                        else:
                            break

                    if all_ok:

                        # Compare the old and new velocity norms:  
                        if norm_new > norm_old + tol:  # the state is inside V

                            # Update the optimal solution:
                            for i in range(N - f):
                                x_sol[i + f] = ocp.ocp_solver.get(i, "x")
                                u_sol[i + f] = ocp.ocp_solver.get(i, "u")

                            x_out = np.copy(x_sol[f][:ocp.ocp.dims.nx - 1])
                            x_out[system_sel:] += eps * x_out[system_sel:] / norm_new

                            # TODO: verify the sign of eps p
                            # for l in range(system_sel):
                            #     x_out[l + system_sel] = x_out[l + system_sel] + eps * x_out[l + system_sel] / norm_new

                            # Check if velocities lie on a limit:
                            if checkVelocityBounds(x_out[system_sel:]):
                                is_x_at_limit = True  # the state is on dX
                            else:
                                is_x_at_limit = False  # the state is on dV
                                x_sym[f] = x_out

                        else:
                            is_x_at_limit = False  # the state is on dV

                            # Generate the new corresponding unviable state in the cost direction:
                            x_out = np.copy(x_sol[f][:ocp.ocp.dims.nx - 1])
                            x_out[system_sel:] -= eps * p[:system_sel]

                            x_sym[f] = x_out

                    else:  # we cannot say whether the state is on dV or inside V

                        for r in range(f, N):
                            if checkVelocityBounds(x_sol[r, system_sel:]):
                                # Save the viable states at velocity limits:
                                valid_data = np.append(valid_data, [x_sol[r][:ocp.ocp.dims.nx - 1]], axis=0)

                        break

            else:
                # If the previous state was not on a limit, the current state location can be identified using
                # the corresponding unviable state which can be computed by simulating the system starting from 
                # the previous unviable state.

                # Simulate next unviable state:
                u_sym = np.copy(u_sol[f - 1])
                sim.acados_integrator.set("u", u_sym)
                sim.acados_integrator.set("x", x_sym[f - 1])
                status = sim.acados_integrator.solve()
                x_out = sim.acados_integrator.get("x")
                x_sym[f] = x_out

                # When the state of the unviable simulated trajectory violates a limit, the corresponding viable state
                # of the optimal trajectory is on dX:
                if checkStateBounds(x_out):
                    is_x_at_limit = True  # the state is on dX
                else:
                    is_x_at_limit = False  # the state is on dV

            if all(i < q_max - eps and i > q_min + eps for i in x_sol[f][:system_sel]) and all(
                    abs(i) > tol for i in x_sol[f][system_sel:ocp.ocp.dims.nx - 1]):
                # Save the viable and unviable states:
                valid_data = np.append(valid_data, [x_sol[f][:ocp.ocp.dims.nx - 1]], axis=0)
        # print("Times in which the state bounds hold: ", count)
        return valid_data.tolist(), valid_traj

    else:
        return None, None


start_time = time.time()

# Select system:
system_sel = 2  # 2 for double pendulum, 3 for triple pendulum

DATA_GEN = False
MODEL_TRAIN = True

# Prune the model:
prune_model = False
prune_amount = 0.5  # percentage of connections to delete

# Ocp initialization:
if system_sel == 3:
    ocp = OCPtriplependulumINIT()
    sim = SYMtriplependulumINIT()
elif system_sel == 2:
    ocp = OCPdoublependulumINIT()
    sim = SYMdoublependulumINIT()
else:
    raise Exception("Sorry, the selected system is not recognised")

X_test = np.load('../data' + str(system_sel) + '_test.npy')

# Position, velocity and torque bounds:
v_max = ocp.dthetamax
v_min = - ocp.dthetamax
q_max = ocp.thetamax
q_min = ocp.thetamin
tau_max = ocp.Cmax
x_min = np.hstack([[q_min] * system_sel, [v_min] * system_sel])
x_max = np.hstack([[q_max] * system_sel, [v_max] * system_sel])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # pytorch device

dt_sym = 1e-2  # time step duration
N_start = 100  # initial number of time steps
tol = ocp.ocp.solver_options.nlp_solver_tol_stat  # OCP cost tolerance
eps = tol * 10  # unviable data generation parameter

if DATA_GEN:
    print('Start data generation')

    # Data generation:
    cpu_num = 24
    num_prob = 500
    with Pool(cpu_num) as p:
        res = list(tqdm(p.imap(data_generation, range(num_prob)), total=num_prob))

    x0, traj = zip(*res)
    X_temp = [i for i in x0 if i is not None]
    traj_temp = [i for i in traj if i is not None]
    print('Data generation completed')

    # Print data generations statistics:
    solved = len(X_temp)
    print('Solved/tot', len(X_temp) / num_prob)
    X_save = np.array([i for f in X_temp for i in f])
    X_traj = np.array([i for f in traj_temp for i in f])
    print('Saved/tot', len(X_save) / (solved * 100))

    # Save training data:
    np.save('data_' + str(system_sel) + 'dof_vboc', np.asarray(X_save))
    # np.save('traj_' + str(system_sel) + 'dof_vboc', np.asarray(X_traj))

    # Plot the data
    plt.figure()
    plt.xlim(q_min, q_max)
    plt.ylim(v_min, v_max)
    plt.scatter(X_save[:, 0], X_save[:, 2], s=1, label='q1')
    plt.scatter(X_save[:, 1], X_save[:, 3], s=1, label='q2')
    plt.legend()
    plt.show()
else:
    print('Load data')
    X_save = np.load('data_' + str(system_sel) + 'dof_vboc' + '.npy')

# Pytorch params:
input_layers = ocp.ocp.dims.nx - 1
hidden_layers = (input_layers - 1) * 100
output_layers = 1
learning_rate = 1e-3

# Model and optimizer:
model_dir = NeuralNetDIR(input_layers, hidden_layers, output_layers).to(device)
criterion_dir = nn.MSELoss()
optimizer_dir = torch.optim.Adam(model_dir.parameters(), lr=learning_rate)

# model_dir.load_state_dict(torch.load('model_' + system_sel + 'dof_vboc'))

# Joint positions mean and variance:
mean_dir, std_dir = torch.mean(torch.tensor(X_save[:, :system_sel].tolist())).to(device).item(), torch.std(
    torch.tensor(X_save[:, :system_sel].tolist())).to(device).item()
torch.save(mean_dir, 'mean_' + str(system_sel) + 'dof_vboc')
torch.save(std_dir, 'std_' + str(system_sel) + 'dof_vboc')

# Rewrite data in the form [normalized positions, velocity direction, velocity norm]:
X_train_dir = np.empty((X_save.shape[0], ocp.ocp.dims.nx))

for i in range(X_train_dir.shape[0]):

    vel_norm = norm(X_save[i][system_sel:ocp.ocp.dims.nx - 1])
    X_train_dir[i][ocp.ocp.dims.nx - 1] = vel_norm

    for l in range(system_sel):
        X_train_dir[i][l] = (X_save[i][l] - mean_dir) / std_dir
        X_train_dir[i][l + system_sel] = X_save[i][l + system_sel] / vel_norm

beta = 0.95
n_minibatch = 4096
B = int(X_save.shape[0] * 100 / n_minibatch)  # number of iterations for 100 epoch
it_max = B * 10

training_evol = []

if MODEL_TRAIN:
    print('Start model training')

    it = 1
    val = max(X_train_dir[:, ocp.ocp.dims.nx - 1])

    # Train the model
    while val > 1e-3 and it < it_max:
        ind = random.sample(range(len(X_train_dir)), n_minibatch)

        X_iter_tensor = torch.Tensor([X_train_dir[i][:ocp.ocp.dims.nx - 1] for i in ind]).to(device)
        y_iter_tensor = torch.Tensor([[X_train_dir[i][ocp.ocp.dims.nx - 1]] for i in ind]).to(device)

        # Forward pass
        outputs = model_dir(X_iter_tensor)
        loss = criterion_dir(outputs, y_iter_tensor)

        # Backward and optimize
        loss.backward()
        optimizer_dir.step()
        optimizer_dir.zero_grad()

        val = beta * val + (1 - beta) * loss.item()
        it += 1

        if it % B == 0:
            print(val)
            training_evol.append(val)

    print('Model training completed')
    print('At iter: ', it)

    # Show the resulting RMSE on the training data:
    outputs = np.empty((len(X_train_dir), 1))
    n_minibatch_model = pow(2, 15)
    with torch.no_grad():
        X_iter_tensor = torch.Tensor(X_train_dir[:, :ocp.ocp.dims.nx - 1]).to(device)
        y_iter_tensor = torch.Tensor(X_train_dir[:, ocp.ocp.dims.nx - 1:]).to(device)
        my_dataloader = DataLoader(X_iter_tensor, batch_size=n_minibatch_model, shuffle=False)
        for (idx, batch) in enumerate(my_dataloader):
            if n_minibatch_model * (idx + 1) > len(X_train_dir):
                outputs[n_minibatch_model * idx:len(X_train_dir)] = model_dir(batch).cpu().numpy()
            else:
                outputs[n_minibatch_model * idx:n_minibatch_model * (idx + 1)] = model_dir(batch).cpu().numpy()
        outputs_tensor = torch.Tensor(outputs).to(device)
        print('RMSE train data: ', torch.sqrt(criterion_dir(outputs_tensor, y_iter_tensor)))

    # Compute resulting RMSE wrt testing data:

    X_test_dir = np.empty((X_test.shape[0], ocp.ocp.dims.nx))
    for i in range(X_test_dir.shape[0]):
        vel_norm = norm(X_test[i][system_sel:ocp.ocp.dims.nx - 1])
        X_test_dir[i][ocp.ocp.dims.nx - 1] = vel_norm
        for l in range(system_sel):
            X_test_dir[i][l] = (X_test[i][l] - mean_dir) / std_dir
            X_test_dir[i][l + system_sel] = X_test[i][l + system_sel] / vel_norm

    with torch.no_grad():
        X_iter_tensor = torch.Tensor(X_test_dir[:, :ocp.ocp.dims.nx - 1]).to(device)
        y_iter_tensor = torch.Tensor(X_test_dir[:, ocp.ocp.dims.nx - 1:]).to(device)
        outputs = model_dir(X_iter_tensor)
        print('RMSE test data: ', torch.sqrt(criterion_dir(outputs, y_iter_tensor)))

    # Save the model:
    torch.save(model_dir.state_dict(), 'model_' + str(system_sel) + 'dof_vboc')

    if prune_model:

        parameters_to_prune = ((model_dir.linear_relu_stack[0], 'weight'), (model_dir.linear_relu_stack[2], 'weight'),
                               (model_dir.linear_relu_stack[4], 'weight'), (model_dir.linear_relu_stack[0], 'bias'),
                               (model_dir.linear_relu_stack[2], 'bias'), (model_dir.linear_relu_stack[4], 'bias'),)

        prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=prune_amount, )

        # prune.l1_unstructured(model_dir.linear_relu_stack[0], name='weight', amount=prune_amount)
        # prune.l1_unstructured(model_dir.linear_relu_stack[2], name='weight', amount=prune_amount)
        # prune.l1_unstructured(model_dir.linear_relu_stack[4], name='weight', amount=prune_amount)
        # prune.l1_unstructured(model_dir.linear_relu_stack[0], name='bias', amount=prune_amount)
        # prune.l1_unstructured(model_dir.linear_relu_stack[2], name='bias', amount=prune_amount)
        # prune.l1_unstructured(model_dir.linear_relu_stack[4], name='bias', amount=prune_amount)

        # model_dir.load_state_dict({k: v for k, v in initial_model_params.items()
        # if 'weight' in k or 'bias' in k}, strict=False)

        print('Restart model training after pruning')

        it = 1
        val = max(X_train_dir[:, ocp.ocp.dims.nx - 1])

        # Train the model
        while val > 1e-3 and it < it_max:
            ind = random.sample(range(len(X_train_dir)), n_minibatch)

            X_iter_tensor = torch.Tensor([X_train_dir[i][:ocp.ocp.dims.nx - 1] for i in ind]).to(device)
            y_iter_tensor = torch.Tensor([[X_train_dir[i][ocp.ocp.dims.nx - 1]] for i in ind]).to(device)

            # Forward pass
            outputs = model_dir(X_iter_tensor)
            loss = criterion_dir(outputs, y_iter_tensor)

            # Backward and optimize
            loss.backward()
            optimizer_dir.step()
            optimizer_dir.zero_grad()

            val = beta * val + (1 - beta) * loss.item()
            it += 1

            if it % B == 0:
                print(val)
                training_evol.append(val)

        print('Model training completed')

        prune.remove(model_dir.linear_relu_stack[0], 'weight')
        prune.remove(model_dir.linear_relu_stack[2], 'weight')
        prune.remove(model_dir.linear_relu_stack[4], 'weight')
        prune.remove(model_dir.linear_relu_stack[0], 'bias')
        prune.remove(model_dir.linear_relu_stack[2], 'bias')
        prune.remove(model_dir.linear_relu_stack[4], 'bias')

        # print('----------------------')
        # print(list(model_dir.named_parameters()))
        # print(list(model_dir.named_buffers()))

        with torch.no_grad():
            X_iter_tensor = torch.Tensor(X_test_dir[:, :ocp.ocp.dims.nx - 1]).to(device)
            y_iter_tensor = torch.Tensor(X_test_dir[:, ocp.ocp.dims.nx - 1:]).to(device)
            outputs = model_dir(X_iter_tensor)
            print('RMSE test data after pruning: ', torch.sqrt(criterion_dir(outputs, y_iter_tensor)))
            print('Pruning percentage: ', prune_amount * 100)

    with torch.no_grad():
        # Compute safety margin:
        outputs = model_dir(X_iter_tensor).cpu().numpy()
        safety_margin = np.amax(np.array(
            [(outputs[i] - X_test_dir[i][-1]) / X_test_dir[i][-1] for i in range(X_test_dir.shape[0]) if
             outputs[i] - X_test_dir[i][-1] > 0]))
        print('Maximum error wrt test data', safety_margin)

    # Save the pruned model:
    torch.save(model_dir.state_dict(), 'model_' + str(system_sel) + 'dof_vboc')

    print("Execution time: %s seconds" % (time.time() - start_time))

    # Show the training evolution:
    plt.figure()
    plt.plot(training_evol)

    # Show training data and resulting set approximation:
    if system_sel == 3:
        plots_3dof(X_save, q_min, q_max, v_min, v_max, model_dir, mean_dir, std_dir, device)
    elif system_sel == 2:
        plots_2dof(X_save, q_min, q_max, v_min, v_max, model_dir, mean_dir, std_dir, device)

    plt.show()
