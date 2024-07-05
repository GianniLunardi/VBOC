import os
import time 
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool
from urdf_parser_py.urdf import URDF
import adam 
from adam.numpy import KinDynComputations as KinDynNumpy
from adam.casadi import KinDynComputations as KinDynCasadi
import torch
from vboc.parser import Parameters, parse_args
from vboc.abstract import AdamModel
from vboc.controller import ViabilityController
from vboc.learning import NeuralNetwork, RegressionNN, plot_viability_kernel


def computeDataOnBorder(q, N_guess):
    controller.resetHorizon(N_guess)

    # Randomize the initial state
    d = np.array([random.uniform(-1, 1) for _ in range(model.nv)])

    # Set the initial guess
    x_guess = np.zeros((N_guess, model.nx))
    u_guess = np.zeros((N_guess, model.nu))
    x_guess[:, :nq] = np.full((N_guess, nq), q)

    d /= np.linalg.norm(d)
    controller.setGuess(x_guess, u_guess)

    # Solve the OCP
    x_star, u_star, _, status = controller.solveVBOC(q, d, N_guess, n=N_increment, repeat=50)
    if x_star is None:
        return None, None, None, status
    else:
        return x_star[0], x_star, u_star, status
    

def fixedVelocityDir(N_guess, n_pts=200):
    """ Compute data on section of the viability kernel"""
    controller.resetHorizon(N_guess)
    x_sec = np.empty((0, model.nx))
    x_traj = []
    for i in range(nq):
        j = 0
        jj = 0
        while j < n_pts:
            q_try = (model.x_max[:nq] + model.x_min[:nq]) / 2
            q_try[i] = np.random.uniform(model.x_min[i], model.x_max[i])
            
            if params.obs_flag:
                T_ee = kin_dyn_np.forward_kinematics(params.frame_name, np.eye(4), q_try)
                t_glob = T_ee[:3, 3] + T_ee[:3, :3] @ model.t_loc
                d_ee = t_glob - model.t_ball
                if t_glob[2] <= model.z_bounds[0] or np.dot(d_ee, d_ee) <= model.ball_bounds[0]:
                    continue
                jj += 1

            x_guess = np.zeros((N_guess, model.nx))
            u_guess = np.zeros((N_guess, model.nu))
            x_guess[:, :nq] = np.full((N_guess, nq), q_try)

            d = np.zeros(model.nv)
            d[i] = random.choice([-1, 1])

            controller.setGuess(x_guess, u_guess)
            x_star, _, _, _ = controller.solveVBOC(q_try, d, N_guess, n=N_increment, repeat=50)
            if x_star is not None:
                x_sec = np.vstack([x_sec, x_star[0]])
                j += 1
                x_traj.append(x_star)
    return x_sec, x_traj
                


if __name__ == '__main__':
    
    start_time = time.time()

    args = parse_args()
    # Define the available systems
    available_systems = ['pendulum', 'double_pendulum', 'ur5', 'z1', 'dsr']
    try:
        if args['system'] not in available_systems:
            raise NameError
    except NameError:
        print('\nSystem not available! Available: ', available_systems, '\n')
        exit()
    params = Parameters(args['system']) 
    params.build = args['build']

    # Load the model of the robot
    robot = URDF.from_xml_file(params.robot_urdf)
    
    try:
        n_dofs = args['dofs'] if args['dofs'] else len(robot.joints)
        if n_dofs > len(robot.joints) or n_dofs < 1:
            raise ValueError
    except ValueError:
        print(f'\nInvalid number of degrees of freedom! Must be >= 1 and <= {len(robot.joints)}\n')
        exit()

    robot_joints = robot.joints[1:n_dofs+1] if params.urdf_name == 'z1' else robot.joints[:n_dofs]
    joint_names = [joint.name for joint in robot_joints]
    kin_dyn_np = KinDynNumpy(params.robot_urdf, joint_names, robot.get_root())
    kin_dyn_cs = KinDynCasadi(params.robot_urdf, joint_names, robot.get_root())
    kin_dyn_np.set_frame_velocity_representation(adam.Representations.BODY_FIXED_REPRESENTATION)
    kin_dyn_cs.set_frame_velocity_representation(adam.Representations.BODY_FIXED_REPRESENTATION)

    if params.payload:
        # Add a payload to the robot --> cylinder m = 3 kg, r = 0.02 m, h = 0.1 m
        # TODO: this is only for a cylinder on z1 gripper, define for general payload/robot 

        m = 1.
        r = 0.02
        h = 0.1
        Ixx = 1 / 12 * m * (3 * r**2 + h**2)
        Izz = 1 / 2 * m * r**2
        
        kin_dyn_cs.rbdalgos.model.links[params.frame_name].inertial.mass += m
        kin_dyn_cs.rbdalgos.model.links[params.frame_name].inertial.inertia.ixx += Ixx
        kin_dyn_cs.rbdalgos.model.links[params.frame_name].inertial.inertia.iyy += Ixx
        kin_dyn_cs.rbdalgos.model.links[params.frame_name].inertial.inertia.izz += Izz

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AdamModel(params, kin_dyn_cs, robot_joints, n_dofs)
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
    try:
        if horizon < 1:
            raise ValueError
    except ValueError:
        print('\nThe horizon must be greater than 0!\n')
        exit()
    if horizon < N:
        N = horizon
        N_increment = 0   
    obs_string = '_obs' if params.obs_flag else ''
    nn_filename = f'{params.NN_DIR}model_{nq}dof{obs_string}.pt'

    # DATA GENERATION
    if args['vboc']:
        # Generate all the initial condition (training + test)
        if params.obs_flag:
            progress_bar = tqdm(total=params.prob_num + params.test_num, desc='ICs')
            i = 0
            H_b = np.eye(4)
            q_init = np.empty((0, nq)) 
            while i < params.prob_num + params.test_num:
                q_try = np.random.uniform(model.x_min[:nq], model.x_max[:nq])
                T_ee = kin_dyn_np.forward_kinematics(params.frame_name, H_b, q_try)
                t_glob = T_ee[:3, 3] + T_ee[:3, :3] @ model.t_loc 
                d_ee = t_glob - model.t_ball
                # if t_glob[2] > -0.25:
                if t_glob[2] > model.z_bounds[0] and np.dot(d_ee, d_ee) > model.ball_bounds[0]:
                    q_init = np.vstack([q_init, q_try])
                    i += 1
                    progress_bar.update(1)
            progress_bar.close()
        else:
            # No obstacles --> random initial conditions inside the bounds
            q_init = np.random.uniform(model.x_min[:nq], model.x_max[:nq], 
                                       size=(params.prob_num + params.test_num, nq))

        print('Start data generation')
        with Pool(params.cpu_num) as p:
            # inputs --> (initial random configuration, horizon)
            res = p.starmap(computeDataOnBorder, [(q0, N) for q0 in q_init])
        
        x_data_temp, x_t, u_t, status = zip(*res)
        x_data = np.vstack([i for i in x_data_temp if i is not None])
        x_traj = np.asarray([i for i in x_t if i is not None])
        u_traj = np.asarray([i for i in u_t if i is not None])
    
        solved = len(x_data)
        print('Solved/numb of problems: %.3f' % (solved / (params.prob_num + params.test_num)))
        print('Total number of points: %d' % len(x_data))
        np.save(params.DATA_DIR + str(nq) + 'dof_vboc', x_data)
        np.save(params.DATA_DIR + 'traj_vboc', x_traj)

        # Plot trajectory solution
        PLOTSSS = 0

        if PLOTSSS:
            colors = np.linspace(0, 1, horizon)
            t = np.linspace(0, horizon * params.dt, horizon)
            def computed_torque(x, u):
                tau = np.zeros((len(x), nq))
                for i in range(len(x)):
                    tau[i] = kin_dyn_np.mass_matrix(H_b, x[i, :nq])[6:, 6:] @ u[i] + \
                            kin_dyn_np.bias_force(H_b, x[i, :nq], np.zeros(6), x[i, nq:])[6:]
                return tau
            # clear the plots directory
            for file in os.listdir(params.DATA_DIR + '/plots'):
                os.remove(params.DATA_DIR + '/plots/' + file)
            for k in range(len(x_traj)):
                fig, ax = plt.subplots(2, 2)
                ax = ax.reshape(-1)
                for i in range(nq):
                    ax[i].grid(True)
                    ax[i].scatter(x_traj[k][:, i], x_traj[k][:, nq + i], c=colors, cmap='coolwarm')
                    ax[i].set_xlim([model.x_min[i], model.x_max[i]])
                    ax[i].set_ylim([model.x_min[nq + i], model.x_max[nq + i]])
                    ax[i].set_title(f'Joint {i + 1}')
                    ax[i].set_xlabel(f'q_{i + 1}')
                    ax[i].set_ylabel(f'dq_{i + 1}')
                plt.suptitle(f'Trajectory {k + 1}')
                plt.tight_layout()
                plt.savefig(params.DATA_DIR + f'/plots/traj_{k + 1}.png')
                plt.close(fig)

                fig, ax = plt.subplots(2, 2)
                ax = ax.reshape(-1)
                for i in range(nq):
                    ax[i].grid(True)
                    ax[i].plot(t, x_traj[k][:, nq + i], label=f'v_{i + 1}')
                    ax[i].plot(t, u_traj[k][:, i], label=f'a_{i + 1}')
                    ax[i].axhline(model.x_min[nq + i], color='b', linestyle='--')
                    ax[i].axhline(model.x_max[nq + i], color='b', linestyle='--')
                    ax[i].set_xlabel('Time [s]')
                    ax[i].set_ylabel('Velocity [rad/s]')
                    # ax[i].set_ylim([model.x_min[nq + i], model.x_max[nq + i]])
                    ax[i].legend()
                plt.suptitle(f'Trajectory {k + 1}')
                plt.tight_layout()
                plt.savefig(params.DATA_DIR + f'/plots/vel_{k + 1}.png')
                plt.close(fig)

                fig, ax = plt.subplots(2, 2)
                ax = ax.reshape(-1)
                for i in range(nq):
                    ax[i].grid(True)
                    ax[i].plot(t, computed_torque(x_traj[k], u_traj[k])[:, i], label=f'tau_{i + 1}')
                    ax[i].set_xlabel('Time [s]')
                    ax[i].set_ylabel('Torque [Nm]')
                    ax[i].set_ylim([model.tau_min[i], model.tau_max[i]])
                    ax[i].legend()
                plt.suptitle(f'Trajectory {k + 1}')
                plt.tight_layout()
                plt.savefig(params.DATA_DIR + f'/plots/torque_{k + 1}.png')
                plt.close(fig)

        # histogram of status
        plt.figure()
        plt.hist(status, bins=[0, 1, 2, 3, 4, 5], edgecolor='black', align='left', rwidth=0.8)
        plt.title('Histogram of status flags')
        plt.xlabel('Flag')
        plt.ylabel('Frequency')
        plt.xticks(range(5))

    # TRAINING
    if args['training']:
        # Load the data
        x_data = np.load(params.DATA_DIR + str(nq) + 'dof_vboc.npy')
        np.random.shuffle(x_data)
        nn_model = NeuralNetwork(model.nx, 300, 1, torch.nn.ReLU()).to(device)
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(nn_model.parameters(), lr=params.learning_rate)
        regressor = RegressionNN(params, nn_model, loss_fn, optimizer)

        # Compute outputs and inputs
        mean = np.mean(x_data[:, :nq])
        std = np.std(x_data[:, :nq])
        x_data[:, :nq] = (x_data[:, :nq] - mean) / std
        y_data = np.linalg.norm(x_data[:, nq:], axis=1).reshape(len(x_data), 1)
        for k in range(len(x_data)):
            if y_data[k] != 0.: 
                x_data[k, nq:] /= y_data[k] 
        
        x_train, y_train = x_data[:params.prob_num], y_data[:params.prob_num]
        print('Start training\n')
        evolution = regressor.training(x_train, y_train, args['epochs'])
        print('Training completed\n')

        print('Evaluate the model')
        _, rmse_train = regressor.testing(x_train, y_train)
        print('RMSE on Training data: %.5f' % rmse_train)

        x_test, y_test = x_data[-params.test_num:], y_data[-params.test_num:]
        out_test, rmse_test = regressor.testing(x_test, y_test)
        print('RMSE on Test data: %.5f' % rmse_test)

        # Safety margin
        safety_margin = np.amax((out_test - y_test) / y_test)
        print(f'Maximum error wrt test data: {safety_margin:.5f}')

        # Save the model
        torch.save({'mean': mean, 'std': std, 'model': nn_model.state_dict()}, nn_filename)

        print('\nPlot the loss evolution')
        # Plot the loss evolution
        plt.figure()
        plt.grid(True, which='both')
        plt.semilogy(evolution)
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss (LP filtered)')
        plt.title(f'Training evolution, horizon {N}')
        plt.savefig(params.DATA_DIR + f'evolution_{N}.png')

    # PLOT THE VIABILITY KERNEL
    if args['plot']:
        nn_data = torch.load(nn_filename)
        nn_model = NeuralNetwork(model.nx, 300, 1, torch.nn.ReLU())
        nn_model.load_state_dict(nn_data['model'])

        x_fixed, x_traj = fixedVelocityDir(N)
        np.save(params.DATA_DIR + 'fixed_dir_traj.npy', x_traj)

        x_f_test = np.empty_like(x_fixed)
        x_f_test[:, :nq] = (x_fixed[:, :nq] - nn_data['mean']) / nn_data['std']
        y_f_test = np.linalg.norm(x_fixed[:, nq:], axis=1).reshape(len(x_fixed), 1)
        x_f_test[:, nq:] = x_fixed[:, nq:] / y_f_test
        out_f_test = np.empty_like(y_f_test)
        with torch.no_grad():
            for i in range(len(x_f_test)):
                out_f_test[i] = nn_model(torch.Tensor(x_f_test[i])).numpy()
        print(f'Safety margin: {np.amax((out_f_test - y_f_test) / y_f_test):.5f}')

        plot_viability_kernel(params, model, kin_dyn_np, nn_model, nn_data['mean'], nn_data['std'], x_fixed, N)
    plt.show()

    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    print(f'Elapsed time: {hours}:{minutes:2d}:{seconds:2d}')