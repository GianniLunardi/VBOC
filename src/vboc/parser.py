import os
import yaml
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--system', type=str, default='double_pendulum',
                        help='Systems to test. Available: pendulum, double_pendulum, ur5, z1')
    parser.add_argument('--dofs', type=int, default=False, nargs='?',
                        help='Number of desired degrees of freedom of the system')
    parser.add_argument('-b', '--build', action='store_true',
                        help='Build the code of the embedded controller')
    parser.add_argument('-v', '--vboc', action='store_true',
                        help='Compute data on border of the viability kernel')
    parser.add_argument('--horizon', type=int, default=100,
                        help='Horizon of the optimal control problem')
    parser.add_argument('-t', '--training', action='store_true',
                        help='Train the neural network model that approximates the viability kernel')
    parser.add_argument('-p', '--plot', action='store_true',
                        help='Plot the approximated viability kernel')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs for training the neural network')
    return vars(parser.parse_args())


class Parameters:
    def __init__(self, urdf_name):
        self.urdf_name = urdf_name
        # Define all the useful paths
        self.PKG_DIR = os.path.dirname(os.path.abspath(__file__))
        self.ROOT_DIR = os.path.join(self.PKG_DIR, '../..')
        self.CONF_DIR = os.path.join(self.ROOT_DIR, 'config/')
        self.DATA_DIR = os.path.join(self.ROOT_DIR, 'data/')
        self.GEN_DIR = os.path.join(self.ROOT_DIR, 'generated/')
        self.NN_DIR = os.path.join(self.ROOT_DIR, 'nn_models/' + urdf_name + '/')
        self.ROBOTS_DIR = os.path.join(self.ROOT_DIR, 'robots/')
        # temp solution
        if urdf_name == 'ur5':
            self.robot_urdf = f'{self.ROBOTS_DIR}/ur_description/urdf/{urdf_name}_robot.urdf'
        else:
            self.robot_urdf = f'{self.ROBOTS_DIR}/{urdf_name}_description/urdf/{urdf_name}.urdf'

        parameters = yaml.load(open(self.ROOT_DIR + '/config.yaml'), Loader=yaml.FullLoader)

        self.prob_num = int(parameters['prob_num'])
        self.n_steps = int(parameters['n_steps'])
        self.cpu_num = int(parameters['cpu_num'])
        self.build = False
        
        self.T = float(parameters['T'])
        self.dt = float(parameters['dt'])
        self.alpha = int(parameters['alpha'])

        self.solver_type = 'SQP'
        self.solver_mode = parameters['solver_mode']
        self.nlp_max_iter = int(parameters['nlp_max_iter'])
        self.qp_max_iter = int(parameters['qp_max_iter'])
        self.nlp_tol_stat = float(parameters['nlp_tol_stat'])
        self.alpha_reduction = float(parameters['alpha_reduction'])
        self.alpha_min = float(parameters['alpha_min'])
        self.levenberg_marquardt = float(parameters['levenberg_marquardt'])

        self.state_tol = float(parameters['state_tol'])
        # self.conv_tol = float(parameters['conv_tol'])
        self.cost_tol = float(parameters['cost_tol'])
        self.globalization = 'MERIT_BACKTRACKING'

        self.learning_rate = float(parameters['learning_rate'])
        self.batch_size = int(parameters['batch_size'])
        self.beta = float(parameters['beta'])
        self.train_ratio = float(parameters['train_ratio'])
        self.val_ratio = float(parameters['val_ratio'])

        # For cartesian constraint
        self.obs_flag = bool(parameters['obs_flag'])
        if urdf_name == 'double_pendulum':
            frame_name = 'link2' 
        elif urdf_name == 'z1':
            frame_name = 'gripperMover'
        else:
            frame_name = 'none'
        self.frame_name = frame_name       #  TODO: dependence on the robot

        # Payload 
        self.payload = bool(parameters['payload'])
