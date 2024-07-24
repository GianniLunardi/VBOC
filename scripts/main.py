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
    

def fixedVelocityDir(N_guess, n_pts=50):
    """ Compute data on section of the viability kernel"""
    sec_pts = []
    controller.resetHorizon(N_guess)
    for i in range(nq):
        q_grid = np.linspace(model.x_min[i], model.x_max[i], n_pts)
        q_grid = np.tile(q_grid, 2)
        x_sec = np.empty((0, model.nx))
        for j in range(n_pts * 2):
            q_try = (model.x_max[:nq] + model.x_min[:nq]) / 2
            q_try[i] = q_grid[j]
            
            if params.obs_flag:
                T_ee = kin_dyn_np.forward_kinematics(params.frame_name, np.eye(4), q_try)
                t_glob = T_ee[:3, 3] + T_ee[:3, :3] @ model.t_loc
                d_ee = t_glob - model.t_ball
                if t_glob[2] <= model.z_bounds[0] or np.dot(d_ee, d_ee) <= model.ball_bounds[0]:
                    continue

            x_guess = np.zeros((N_guess, model.nx))
            u_guess = np.zeros((N_guess, model.nu))
            x_guess[:, :nq] = np.full((N_guess, nq), q_try)

            d = np.zeros(model.nv)
            d[i] = 1 if j < n_pts else -1

            controller.setGuess(x_guess, u_guess)
            x_star, _, _, status = controller.solveVBOC(q_try, d, N_guess, n=N_increment, repeat=50)
            if x_star is not None:
                x_sec = np.vstack([x_sec, x_star[0]])
            # else:
            #     print(f'No solution found at dof {i}, step {j}, flag: {status}')
        sec_pts.append(x_sec)
    return sec_pts
                


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
            progress_bar = tqdm(total=params.prob_num, desc='ICs')
            i = 0
            H_b = np.eye(4)
            q_init = np.empty((0, nq)) 
            while i < params.prob_num:
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
            q_init = np.random.uniform(model.x_min[:nq], model.x_max[:nq], size=(params.prob_num, nq))

        print('Start data generation')
        with Pool(params.cpu_num) as p:
            # inputs --> (initial random configuration, horizon)
            res = p.starmap(computeDataOnBorder, [(q0, N) for q0 in q_init])
        
        x_data_temp, x_t, u_t, status = zip(*res)
        x_data = np.vstack([i for i in x_data_temp if i is not None])
        x_traj = np.asarray([i for i in x_t if i is not None])
        u_traj = np.asarray([i for i in u_t if i is not None])
    
        solved = len(x_data)
        print('Perc solved/numb of problems: %.2f' % (solved / params.prob_num * 100))
        print('Total number of points: %d' % len(x_data))
        np.save(params.DATA_DIR + str(nq) + 'dof_vboc', x_data)
        np.save(params.DATA_DIR + str(nq) + 'dof_trajx', x_traj)
        np.save(params.DATA_DIR + str(nq) + 'dof_traju', u_traj)

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
        ub = max(model.x_max[nq:]) * np.sqrt(nq)
        nn_model = NeuralNetwork(model.nx, 400, 1, torch.nn.ReLU()).to(device)
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(nn_model.parameters(), lr=params.learning_rate)
        regressor = RegressionNN(params, nn_model, loss_fn, optimizer)

        # Compute outputs and inputs
        n = len(x_data)
        mean = np.mean(x_data[:, :nq])
        std = np.std(x_data[:, :nq])
        x_data[:, :nq] = (x_data[:, :nq] - mean) / std
        y_data = np.linalg.norm(x_data[:, nq:], axis=1).reshape(n, 1)
        for k in range(n):
            if y_data[k] != 0.: 
                x_data[k, nq:] /= y_data[k] 

        train_size = int(params.train_ratio * n)
        val_size = int(params.val_ratio * n)
        test_size = n - train_size - val_size
        
        x_data = torch.Tensor(x_data).to(device)
        y_data = torch.Tensor(y_data).to(device)
        x_train_val, y_train_val = x_data[:-test_size], y_data[:-test_size]
        print('Start training\n')
        train_evol, val_evol = regressor.training(x_train_val, y_train_val, 
                                                  train_size, args['epochs'], refine=False)
        print('Training completed\n')

        print('Evaluate the model')
        rmse_train, rel_err = regressor.testing(x_train_val, y_train_val)
        print(f'RMSE on Training data: {rmse_train:.5f}')
        print(f'Maximum error wrt training data: {torch.max(rel_err).item():.5f}')

        x_test, y_test = x_data[-test_size:], y_data[-test_size:]
        rmse_test, rel_err = regressor.testing(x_test, y_test)
        print(f'RMSE on Test data: {rmse_test:.5f}')
        print(f'Maximum error wrt test data: {torch.max(rel_err).item():.5f}')

        # Save the model
        torch.save({'mean': mean, 'std': std, 'model': nn_model.state_dict()}, nn_filename)

        # Plot the loss evolution
        plt.figure()
        plt.grid(True, which='both')
        plt.semilogy(train_evol, label='Training', c='b', lw=2)
        plt.semilogy(val_evol, label='Validation', c='g', lw=2)
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss (LP filtered)')
        plt.title(f'Training evolution, horizon {N}')
        plt.savefig(params.DATA_DIR + f'evolution_{N}.png')

        # # Plot the relative (mean) error evolution
        # plt.figure()
        # plt.grid(True, which='both')
        # plt.semilogy(err_evol, label='Rel Error', c='b', lw=2)
        # plt.legend()
        # plt.xlabel('Epochs')
        # plt.ylabel('Relative Error')
        # plt.title(f'Relative error evolution, horizon {N}')
        # plt.savefig(params.DATA_DIR + f'error_{N}.png')

        # Box plot of the relative error
        plt.figure()
        plt.boxplot(rel_err.cpu().numpy(), notch=True)
        plt.title('Box plot of the relative error')
        plt.xlabel('Test data')
        plt.ylabel('Relative error')
        plt.savefig(params.DATA_DIR + 'boxplot.png')

        # Difference between predicted and true values
        with torch.no_grad():
            nn_model.eval()
            y_pred = nn_model(x_test).cpu().numpy()
        plt.figure()
        plt.plot(y_pred - y_test.cpu().numpy())
        plt.xlabel('Test data')
        plt.ylabel('Output')
        plt.savefig(params.DATA_DIR + 'difference.png')

    # PLOT THE VIABILITY KERNEL
    if args['plot']:
        nn_data = torch.load(nn_filename)
        nn_model = NeuralNetwork(model.nx, 400, 1)
        nn_model.load_state_dict(nn_data['model'])

        x_fixed = fixedVelocityDir(N)
        plot_viability_kernel(params, model, kin_dyn_np, nn_model, nn_data['mean'], nn_data['std'], x_fixed, N)
    plt.show()

    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    print(f'Elapsed time: {hours}:{minutes:2d}:{seconds:2d}')
