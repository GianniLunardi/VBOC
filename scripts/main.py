import os
import time 
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from multiprocessing import Pool
from vboc.parser import Parameters, parse_args
from vboc.abstract import AdamModel
from vboc.controller import ViabilityController
from vboc.learning import NeuralNetwork, RegressionNN, plot_viability_kernel


def computeDataOnBorder(q, N_guess, test_flag=0):
    valid_data = np.empty((0, model.nx))
    controller.resetHorizon(N_guess)

    # Randomize the initial state
    d = np.array([random.uniform(-1, 1) for _ in range(model.nv)])

    # Set the initial guess
    x_guess = np.zeros((N_guess, model.nx))
    u_guess = np.zeros((N_guess, model.nu))
    x_guess[:, :nq] = np.full((N_guess, nq), q)

    if not test_flag:
        # Dense sampling near the joint position bounds
        vel_dir = random.choice([-1, 1])
        i = random.choice(range(nq))
        q_init = model.x_min[i] + model.eps if vel_dir == -1 else model.x_max[i] - model.eps
        q_fin = model.x_max[i] - model.eps if vel_dir == -1 else model.x_min[i] + model.eps
        d[i] = random.random() * vel_dir
        q[i] = q_init
        x_guess[:, i] = np.linspace(q_init, q_fin, N_guess)

    d /= np.linalg.norm(d)
    controller.setGuess(x_guess, u_guess)

    # Solve the OCP
    x_star, u_star, N = controller.solveVBOC(q, d, N_guess, n=N_increment, repeat=50)
    if x_star is None:
        return None, None
    if test_flag:
        return x_star[0], x_star
    # Add a final node
    x_star = np.vstack([x_star, x_star[-1]])
    # Save the initial state as valid data
    valid_data = np.vstack([valid_data, x_star[0]])

    # Generate unviable sample in the cost direction
    x_tilde = np.full((N + 1, model.nx), None)
    x_tilde[0] = np.copy(x_star[0])
    x_tilde[0, nq:] -= model.eps * d

    x_limit = True if model.checkVelocityBounds(x_tilde[0, nq:]) else False

    # Iterate along the trajectory to verify the viability of the solution
    for j in range(1, N):
        if x_limit:
            if model.checkStateBounds(x_star[j]):
                x_limit = True
            else:

                if model.checkPositionBounds(x_star[j - 1, :nq]):
                    break

                gamma = np.linalg.norm(x_star[j, nq:])
                d = - x_star[j, nq:]
                d /= np.linalg.norm(d)
                controller.resetHorizon(N - j)
                controller.setGuess(x_star[j:N], u_star[j:N])
                x_new, u_new, _ = controller.solveVBOC(x_star[j, :nq], d, N - j, n=N_increment, repeat=5)
                if x_new is not None:
                    x0 = controller.ocp_solver.get(0, 'x')
                    gamma_new = np.linalg.norm(x0[nq:])
                    if gamma_new > gamma + controller.tol:
                        # Update the optimal trajectory
                        x_star[j:N], u_star[j:N] = x_new[:N - j], u_new[:N - j]

                        # Create unviable state
                        x_tilde[j] = np.copy(x_star[j])
                        x_tilde[j, nq:] += model.eps * x_tilde[j, nq:] / gamma_new

                        # Check if the unviable state is on bound
                        x_limit = True if model.checkVelocityBounds(x_tilde[j, nq:]) else False

                    else:
                        x_limit = False
                        x_tilde[j] = np.copy(x_star[j])
                        x_tilde[j, nq:] -= model.eps * d
                else:
                    for k in range(j, N):
                        if model.checkVelocityBounds(x_star[k, nq:]):
                            valid_data = np.vstack([valid_data, x_star[k]])
                    break
        else:
            x_tilde[j, :nq] = x_tilde[j - 1, :nq] + params.dt * x_tilde[j - 1, nq:]  \
                              + 0.5 * params.dt**2 * u_star[j - 1]
            x_tilde[j, nq:] = x_tilde[j - 1, nq:] + params.dt * u_star[j - 1]
            x_limit = True if model.checkStateBounds(x_tilde[j]) else False
        if model.insideStateConstraints(x_star[j]):
            valid_data = np.vstack([valid_data, x_star[j]])
    return valid_data


if __name__ == '__main__':
    
    start_time = time.time()

    args = parse_args()
    # Define the available systems
    available_systems = ['pendulum', 'double_pendulum', 'ur5', 'z1']
    try:
        if args['system'] not in available_systems:
            raise NameError
    except NameError:
        print('\nSystem not available! Available: ', available_systems, '\n')
        exit()
    params = Parameters(args['system']) 
    model = AdamModel(params, n_dofs=args['dofs'])
    controller = ViabilityController(model)
    nq = model.nq

    # Check if data and nn folders exist, if not create it
    if not os.path.exists(params.DATA_DIR):
        os.makedirs(params.DATA_DIR)
    if not os.path.exists(params.NN_DIR):
        os.makedirs(params.NN_DIR)

    N = 100
    N_increment = 1
    horizon = args['horizon']
    if horizon:
        N = horizon
        N_increment = 0   
    try:
        if horizon < 0:
            raise ValueError
    except ValueError:
        print('\nThe horizon must be greater than 0!\n')
        exit()

    # DATA GENERATION
    if args['vboc']:
        # Generate all the initial condition
        q_init = np.random.uniform(model.x_min[:nq], model.x_max[:nq], size=(params.prob_num, nq))
        print('Start data generation')
        with Pool(params.cpu_num) as p:
            # inputs --> (horizon, flag to compute only the initial state)
            # TRIAL --> uniform sampling of the initial state --> test_flag = 1
            res = p.starmap(computeDataOnBorder, [(q0, N, 1) for q0 in q_init])

        data, traj = zip(*res)
        x_data = np.array([i for i in data if i is not None])
        # x_save = np.array([i for f in x_data for i in f])
        x_traj = [i for i in traj if i is not None]

        solved = len(x_data)
        print('Solved/numb of problems: %.3f' % (solved / params.prob_num))
        # print('Saved/tot: %.3f' % (len(x_save) / (solved * N)))
        # print('Total number of points: %d' % len(x_save))
        np.save(params.DATA_DIR + str(nq) + 'dof_vboc', np.asarray(x_data))
        np.save(params.DATA_DIR + str(nq) + 'dof_traj', np.asarray(x_traj))

        # Plot points (no sense if more than 2 dofs)
        if nq < 3: 
            plt.figure()
            for i in range(nq):
                plt.scatter(x_data[:, i], x_data[:, i + nq], s=1, label=f'q_{i + 1}')
            plt.xlabel('q')
            plt.ylabel('dq')
            plt.legend()
            plt.savefig(params.DATA_DIR + f'{nq}dof_vboc.png')

    # TRAINING
    if args['training']:
        # Load the data
        x_train = np.load(params.DATA_DIR + str(nq) + 'dof_vboc.npy')
        beta = args['beta']
        batch_size = args['batch_size']
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        nn_model = NeuralNetwork(model.nx, (model.nx - 1) * 100, 1).to(device)
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(nn_model.parameters(), lr=args['learning_rate'])
        regressor = RegressionNN(nn_model, loss_fn, optimizer, beta, batch_size)

        # Compute outputs and inputs
        y_train = np.linalg.norm(x_train[:, nq:], axis=1).reshape(len(x_train), 1)
        mean = np.mean(x_train[:, :nq])
        std = np.std(x_train[:, :nq])
        x_train[:, :nq] = (x_train[:, :nq] - mean) / std
        x_train[:, nq:] /= y_train

        print('Start training\n')
        evolution = regressor.training(x_train, y_train)
        print('Training completed\n')

        print('Evaluate the model')
        _, rmse_train = regressor.testing(x_train, y_train)
        print('RMSE on Training data: %.5f' % rmse_train)

        # Compute the test data
        print('Generation of testing data (only x0*)')
        q_test = np.random.uniform(model.x_min[:nq], model.x_max[:nq], size=(params.test_num, nq))
        with Pool(params.cpu_num) as p:
            data = p.starmap(computeDataOnBorder, [(q0, N, 1) for q0 in q_test])

        data, _ = zip(*data)
        x_test = np.array([i for i in data if i is not None])
        y_test = np.linalg.norm(x_test[:, nq:], axis=1).reshape(len(x_test), 1)
        x_test[:, :nq] = (x_test[:, :nq] - mean) / std
        x_test[:, nq:] /= y_test
        out_test, rmse_test = regressor.testing(x_test, y_test)
        print('RMSE on Test data: %.5f' % rmse_test)

        # Safety margin
        safety_margin = np.amax((out_test - y_test) / y_test)
        print(f'Maximum error wrt test data: {safety_margin:.5f}')

        # Save the model
        torch.save({'mean': mean, 'std': std, 'model': nn_model.state_dict()},
                   params.NN_DIR + 'model_' + str(nq) + 'dof.pt')

        print('\nPlot the loss evolution')
        # Plot the loss evolution
        plt.figure()
        plt.plot(evolution)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Training evolution, horion {N}')
        plt.savefig(params.DATA_DIR + f'evolution_{N}.png')

    # PLOT THE VIABILITY KERNEL
    if args['plot']:
        nn_data = torch.load(params.NN_DIR + 'model_' + str(nq) + 'dof.pt')
        nn_model = NeuralNetwork(model.nx, (model.nx - 1) * 100, 1)
        nn_model.load_state_dict(nn_data['model'])
        plot_viability_kernel(params, model, nn_model, nn_data['mean'], nn_data['std'], N)
    # plt.show()

    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    print(f'Elapsed time: {hours}:{minutes:2d}:{seconds:2d}')