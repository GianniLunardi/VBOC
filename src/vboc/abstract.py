import re
import numpy as np
from casadi import MX, vertcat
from urdf_parser_py.urdf import URDF
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver


class AdamModel:
    def __init__(self, params, kin_dyn, robot_joints, nq):
        self.params = params

        self.mass = kin_dyn.mass_matrix_fun()                           # Mass matrix
        self.bias = kin_dyn.bias_force_fun()                            # Nonlinear effects 
        # TODO: this function must be defined for each part where collision is possible 
        if params.obs_flag:
            self.fk = kin_dyn.forward_kinematics_fun(params.frame_name)     # Forward kinematics

        self.amodel = AcadosModel()
        self.amodel.name = params.urdf_name
        self.x = MX.sym("x", nq * 2)
        self.x_dot = MX.sym("x_dot", nq * 2)
        self.u = MX.sym("u", nq)
        self.p = MX.sym("p", nq)
        # Double-integrator 
        self.f_disc = vertcat(
            self.x[:nq] + params.dt * self.x[nq:] + 0.5 * params.dt**2 * self.u,
            self.x[nq:] + params.dt * self.u
        ) 
            
        self.amodel.x = self.x
        self.amodel.u = self.u
        self.amodel.disc_dyn_expr = self.f_disc
        self.amodel.p = self.p

        self.nx = self.amodel.x.size()[0]
        self.nu = self.amodel.u.size()[0]
        self.ny = self.nx + self.nu
        self.nq = nq
        self.nv = nq

        # Joint limits
        joint_lower = np.array([joint.limit.lower for joint in robot_joints])
        joint_upper = np.array([joint.limit.upper for joint in robot_joints])
        joint_velocity = np.array([joint.limit.velocity for joint in robot_joints])
        joint_effort = np.array([joint.limit.effort for joint in robot_joints]) 

        self.tau_min = - joint_effort
        self.tau_max = joint_effort
        self.x_min = np.hstack([joint_lower, - joint_velocity])
        self.x_max = np.hstack([joint_upper, joint_velocity])
        self.eps = params.state_tol

        # Cartesian constraint
        if params.urdf_name == 'double_pendulum':
            self.t_loc = np.array([0., 0., 0.2])
            self.z_bounds = np.array([-0.25, 1e6])
        elif params.urdf_name == 'z1':
            self.t_loc = np.array([0.035, 0., 0.])
            self.z_bounds = np.array([0., 1e6])
            self.x_bounds = np.array([-0.5, 1e6])
        else:
            self.t_loc = np.array([0., 0., 0.2]) 

    def checkPositionBounds(self, q):
        return np.logical_or(np.any(q < self.x_min[:self.nq] + self.eps), np.any(q > self.x_max[:self.nq] - self.eps))

    def checkVelocityBounds(self, v):
        return np.logical_or(np.any(v < self.x_min[self.nq:] + self.eps), np.any(v > self.x_max[self.nq:] - self.eps))

    def checkStateBounds(self, x):
        return np.logical_or(np.any(x < self.x_min + self.eps), np.any(x > self.x_max - self.eps))


class AbstractController:
    def __init__(self, model):
        self.ocp_name = "".join(re.findall('[A-Z][^A-Z]*', self.__class__.__name__)[:-1]).lower()
        self.params = model.params
        self.model = model

        self.N = int(self.params.T / self.params.dt)
        self.ocp = AcadosOcp()

        # Dimensions
        self.ocp.solver_options.tf = self.params.T 
        self.ocp.dims.N = self.N

        # Model
        self.ocp.model = self.model.amodel

        # Cost
        self.addCost()

        # Constraints
        self.ocp.constraints.lbx_0 = self.model.x_min
        self.ocp.constraints.ubx_0 = self.model.x_max
        self.ocp.constraints.idxbx_0 = np.arange(self.model.nx)

        self.ocp.constraints.lbx = self.model.x_min
        self.ocp.constraints.ubx = self.model.x_max
        self.ocp.constraints.idxbx = np.arange(self.model.nx)

        self.ocp.constraints.lbx_e = self.model.x_min
        self.ocp.constraints.ubx_e = self.model.x_max
        self.ocp.constraints.idxbx_e = np.arange(self.model.nx)

        # Nonlinear constraint 
        nl_constraints = []
        nl_lb = []
        nl_ub = []
        H_b = np.eye(4)     # Base roto-translation matrix   
        
        # --> dynamics
        computed_torque = self.model.mass(H_b, self.model.x[:self.model.nq])[6:, 6:] @ self.model.u + \
                          self.model.bias(H_b, self.model.x[:self.model.nq], np.zeros(6), self.model.x[self.model.nq:])[6:]
        nl_constraints.append(computed_torque)
        nl_lb.append(self.model.tau_min)
        nl_ub.append(self.model.tau_max)

        # --> collision
        if self.params.obs_flag:
            T_ee = self.model.fk(H_b, self.model.x[:self.model.nq]) 
            T_ee[:3, 3] += T_ee[:3, :3] @ self.model.t_loc
            nl_constraints.append(T_ee[2, 3])

            nl_lb.append(self.model.z_bounds[0])
            nl_ub.append(self.model.z_bounds[1])

        
        self.model.amodel.con_h_expr_0 = vertcat(*nl_constraints)   
        self.model.amodel.con_h_expr = vertcat(*nl_constraints)
        self.model.amodel.con_h_expr_e = vertcat(*nl_constraints[1:])

        
        self.ocp.constraints.lh_0 = np.hstack(nl_lb)
        self.ocp.constraints.uh_0 = np.hstack(nl_ub)
        self.ocp.constraints.lh = np.hstack(nl_lb)
        self.ocp.constraints.uh = np.hstack(nl_ub)
        self.ocp.constraints.lh_e = np.array(nl_lb[1:])
        self.ocp.constraints.uh_e = np.array(nl_ub[1:])
            
        # Additional constraints
        self.addConstraint()

        # Solver options
        self.ocp.solver_options.integrator_type = "DISCRETE"
        self.ocp.solver_options.hessian_approx = "EXACT"
        self.ocp.solver_options.exact_hess_constr = 0
        self.ocp.solver_options.exact_hess_dyn = 0
        self.ocp.solver_options.nlp_solver_type = self.params.solver_type
        self.ocp.solver_options.hpipm_mode = self.params.solver_mode
        self.ocp.solver_options.nlp_solver_max_iter = self.params.nlp_max_iter
        self.ocp.solver_options.qp_solver_iter_max = self.params.qp_max_iter
        self.ocp.solver_options.globalization = self.params.globalization
        self.ocp.solver_options.alpha_reduction = self.params.alpha_reduction
        self.ocp.solver_options.alpha_min = self.params.alpha_min
        self.ocp.solver_options.levenberg_marquardt = self.params.levenberg_marquardt

        # Generate OCP solver
        gen_name = self.params.GEN_DIR + 'ocp_' + self.ocp_name + '_' + self.model.amodel.name
        self.ocp.code_export_directory = gen_name
        self.ocp_solver = AcadosOcpSolver(self.ocp, json_file=gen_name + '.json', build=self.params.build)

        # Storage
        self.x_guess = np.zeros((self.N, self.model.nx))
        self.u_guess = np.zeros((self.N, self.model.nu))
        self.tol = self.params.cost_tol

    def addCost(self):
        pass

    def addConstraint(self):
        pass

    def setGuess(self, x_guess, u_guess):
        self.x_guess = x_guess
        self.u_guess = u_guess

    def getGuess(self):
        return np.copy(self.x_guess), np.copy(self.u_guess)

    def resetHorizon(self, N):
        self.N = N
        self.ocp_solver.set_new_time_steps(np.full(N, self.params.dt))
        self.ocp_solver.update_qp_solver_cond_N(N)
