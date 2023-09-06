from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSim, AcadosSimSolver
import numpy as np
from acados_template import AcadosModel
from casadi import SX, vertcat, cos, sin, fmax, norm_2, if_else, DM
import scipy.linalg as lin


class MODELdoublependulum:
    def __init__(self, time_step, tot_time):
        model_name = "double_pendulum_ode"

        # constants
        self.m1 = 0.4  # mass of the first link [kself.g]
        self.m2 = 0.4  # mass of the second link [kself.g]
        self.g = 9.81  # self.gravity constant [m/s^2]
        self.l1 = 0.8  # lenself.gth of the first link [m]
        self.l2 = 0.8  # lenself.gth of the second link [m]

        self.time_step = time_step
        self.tot_time = tot_time

        # states
        theta1 = SX.sym("theta1")
        theta2 = SX.sym("theta2")
        dtheta1 = SX.sym("dtheta1")
        dtheta2 = SX.sym("dtheta2")
        self.x = vertcat(theta1, theta2, dtheta1, dtheta2)

        # xdot
        theta1_dot = SX.sym("theta1_dot")
        theta2_dot = SX.sym("theta2_dot")
        dtheta1_dot = SX.sym("dtheta1_dot")
        dtheta2_dot = SX.sym("dtheta2_dot")
        xdot = vertcat(theta1_dot, theta2_dot, dtheta1_dot, dtheta2_dot)

        # controls
        C1 = SX.sym("C1")
        C2 = SX.sym("C2")
        u = vertcat(C1, C2)

        # parameters
        self.p = SX.sym("safe")

        # dynamics
        f_expl = vertcat(
            dtheta1,
            dtheta2,
            (
                    self.l1 ** 2
                    * self.l2
                    * self.m2
                    * dtheta1 ** 2
                    * sin(-2 * theta2 + 2 * theta1)
                    + 2 * C2 * cos(-theta2 + theta1) * self.l1
                    + 2
                    * (
                            self.g * sin(-2 * theta2 + theta1) * self.l1 * self.m2 / 2
                            + sin(-theta2 + theta1) * dtheta2 ** 2 * self.l1 * self.l2 * self.m2
                            + self.g * self.l1 * (self.m1 + self.m2 / 2) * sin(theta1)
                            - C1
                    )
                    * self.l2
            )
            / self.l1 ** 2
            / self.l2
            / (self.m2 * cos(-2 * theta2 + 2 * theta1) - 2 * self.m1 - self.m2),
            (
                    -self.g
                    * self.l1
                    * self.l2
                    * self.m2
                    * (self.m1 + self.m2)
                    * sin(-theta2 + 2 * theta1)
                    - self.l1
                    * self.l2 ** 2
                    * self.m2 ** 2
                    * dtheta2 ** 2
                    * sin(-2 * theta2 + 2 * theta1)
                    - 2
                    * dtheta1 ** 2
                    * self.l1 ** 2
                    * self.l2
                    * self.m2
                    * (self.m1 + self.m2)
                    * sin(-theta2 + theta1)
                    + 2 * C1 * cos(-theta2 + theta1) * self.l2 * self.m2
                    + self.l1
                    * (self.m1 + self.m2)
                    * (sin(theta2) * self.g * self.l2 * self.m2 - 2 * C2)
            )
            / self.l2 ** 2
            / self.l1
            / self.m2
            / (self.m2 * cos(-2 * theta2 + 2 * theta1) - 2 * self.m1 - self.m2)
        )

        self.model = AcadosModel()

        f_impl = xdot - f_expl

        self.model.f_expl_expr = f_expl
        self.model.f_impl_expr = f_impl
        self.model.x = self.x
        self.model.xdot = xdot
        self.model.u = u
        self.model.p = self.p
        self.model.name = model_name


class SYMdoublependulum(MODELdoublependulum):
    def __init__(self, time_step, tot_time, regenerate):
        # inherit initialization
        super().__init__(time_step, tot_time)

        sim = AcadosSim()
        sim.model = self.model
        sim.solver_options.T = self.time_step
        sim.solver_options.num_stages = 4
        sim.parameter_values = np.array([0.])
        self.acados_integrator = AcadosSimSolver(sim, build=regenerate)


class OCPdoublependulum(MODELdoublependulum):
    def __init__(self, nlp_solver_type, time_step, tot_time):
        # inherit initialization
        super().__init__(time_step, tot_time)

        self.ocp = AcadosOcp()

        # times
        self.ocp.solver_options.tf = self.tot_time
        self.ocp.dims.N = int(self.tot_time / self.time_step)

        self.nx = self.model.x.size()[0]
        self.nu = self.model.u.size()[0]
        self.ny = self.nx + self.nu
        self.ny_e = self.nx

        # cost
        self.Q = np.diag([1e-4, 1e-4, 1e-4, 1e-4])
        self.R = np.diag([1e-4, 1e-4])

        self.ocp.cost.W_e = self.Q
        self.ocp.cost.W = lin.block_diag(self.Q, self.R)

        self.ocp.cost.cost_type = "LINEAR_LS"
        self.ocp.cost.cost_type_e = "LINEAR_LS"

        self.ocp.cost.Vx = np.zeros((self.ny, self.nx))
        self.ocp.cost.Vx[: self.nx, :self.nx] = np.eye(self.nx)
        self.ocp.cost.Vu = np.zeros((self.ny, self.nu))
        self.ocp.cost.Vu[self.nx:, :self.nu] = np.eye(self.nu)
        self.ocp.cost.Vx_e = np.eye(self.nx)

        # set constraints
        self.Cmax = 10.
        self.thetamax = np.pi / 4 + np.pi
        self.thetamin = - np.pi / 4 + np.pi
        self.dthetamax = 10.

        self.ocp.parameter_values = np.array([0.])

        self.yref = np.array([np.pi, np.pi, 0., 0., 0., 0.])

        # reference
        self.ocp.cost.yref = self.yref
        self.ocp.cost.yref_e = self.yref[:self.nx]

        self.Cmax_limits = np.array([self.Cmax, self.Cmax])
        self.Cmin_limits = np.array([-self.Cmax, -self.Cmax])
        self.Xmax_limits = np.array(
            [self.thetamax, self.thetamax, self.dthetamax, self.dthetamax])
        self.Xmin_limits = np.array(
            [self.thetamin, self.thetamin, -self.dthetamax, -self.dthetamax])

        self.ocp.constraints.lbu = self.Cmin_limits
        self.ocp.constraints.ubu = self.Cmax_limits
        self.ocp.constraints.idxbu = np.array([0, 1])
        self.ocp.constraints.lbx = self.Xmin_limits
        self.ocp.constraints.ubx = self.Xmax_limits
        self.ocp.constraints.idxbx = np.array([0, 1, 2, 3])

        self.ocp.constraints.lbx_e = self.Xmin_limits
        self.ocp.constraints.ubx_e = self.Xmax_limits
        self.ocp.constraints.idxbx_e = np.array([0, 1, 2, 3])

        self.ocp.constraints.lbx_0 = self.Xmin_limits
        self.ocp.constraints.ubx_0 = self.Xmax_limits
        self.ocp.constraints.idxbx_0 = np.array([0, 1, 2, 3])

        # options
        self.ocp.solver_options.nlp_solver_type = nlp_solver_type
        self.ocp.solver_options.qp_solver_iter_max = 100
        self.ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
        self.ocp.solver_options.nlp_solver_max_iter = 1000
        self.ocp.solver_options.globalization = "MERIT_BACKTRACKING"
        self.ocp.solver_options.alpha_reduction = 0.3
        self.ocp.solver_options.alpha_min = 1e-2
        self.ocp.solver_options.levenberg_marquardt = 1e-2
        # self.ocp.solver_options.nlp_solver_ext_qp_res = 1

    def OCP_solve(self, x0, x_sol_guess, u_sol_guess, ref, joint):
        # Reset current iterate:
        self.ocp_solver.reset()

        self.ocp_solver.constraints_set(0, "lbx", x0)
        self.ocp_solver.constraints_set(0, "ubx", x0)

        yref = self.yref
        yref[joint] = ref
        Q = self.Q
        Q[joint,joint] = 1e4
        W = lin.block_diag(Q, self.R)

        # Set parameters, guesses and constraints:
        for i in range(self.ocp.dims.N):
            self.ocp_solver.set(i, 'x', x_sol_guess[i])
            self.ocp_solver.set(i, 'u', u_sol_guess[i])
            self.ocp_solver.cost_set(i, 'yref', yref, api='new')
            self.ocp_solver.cost_set(i, 'W', W, api='new')

        self.ocp_solver.set(self.ocp.dims.N, 'x', x_sol_guess[self.ocp.dims.N])
        self.ocp_solver.cost_set(self.ocp.dims.N, 'yref', yref[:self.ocp.dims.nx], api='new')
        self.ocp_solver.cost_set(self.ocp.dims.N, 'W', Q, api='new')

        # Solve the OCP:
        status = self.ocp_solver.solve()

        return status


class OCPdoublependulumSTD(OCPdoublependulum):
    def __init__(self, nlp_solver_type, time_step, tot_time, regenerate):
        # inherit initialization
        super().__init__(nlp_solver_type, time_step, tot_time)

        # ocp model
        self.ocp.model = self.model

        # solver
        self.ocp_solver = AcadosOcpSolver(self.ocp, build=regenerate)


def nn_decisionfunction_conservative(params, mean, std, safety_margin, x):
    vel_norm = fmax(norm_2(x[2:]), 1e-3)

    mean = vertcat(mean, mean, 0., 0.)
    std = vertcat(std, std, vel_norm, vel_norm)

    out = (x - mean) / std
    it = 0

    for param in params:

        param = SX(param.tolist())

        if it % 2 == 0:
            out = param @ out
        else:
            out = param + out

            if it == 1 or it == 3:
                out = fmax(0., out)

        it += 1

    return out * (100 - safety_margin) / 100 - vel_norm


class OCPdoublependulumHardTerm(OCPdoublependulum):
    def __init__(self, nlp_solver_type, time_step, tot_time, nn_params, mean, std, regenerate):
        # inherit initialization
        super().__init__(nlp_solver_type, time_step, tot_time)

        # nonlinear constraints
        self.model.con_h_expr_e = nn_decisionfunction_conservative(nn_params, mean, std, self.p, self.x)

        self.ocp.constraints.lh_e = np.array([0.])
        self.ocp.constraints.uh_e = np.array([1e6])

        # ocp model
        self.ocp.model = self.model

        # solver
        self.ocp_solver = AcadosOcpSolver(self.ocp, build=regenerate)


class OCPdoublependulumSoftTerm(OCPdoublependulum):
    def __init__(self, nlp_solver_type, time_step, tot_time, nn_params, mean, std, safety_margin, regenerate):
        # inherit initialization
        super().__init__(nlp_solver_type, time_step, tot_time)

        # nonlinear constraints
        self.model.con_h_expr_e = nn_decisionfunction_conservative(nn_params, mean, std, safety_margin, self.x)

        self.ocp.constraints.lh_e = np.array([0.])
        self.ocp.constraints.uh_e = np.array([1e6])

        self.ocp.constraints.idxsh_e = np.array([0])

        self.ocp.cost.zl_e = np.zeros((1,))
        self.ocp.cost.zu_e = np.zeros((1,))
        self.ocp.cost.Zu_e = np.zeros((1,))
        self.ocp.cost.Zl_e = np.zeros((1,))

        # ocp model
        self.ocp.model = self.model

        # solver
        self.ocp_solver = AcadosOcpSolver(self.ocp, build=regenerate)


class OCPdoublependulumReceidingSoft(OCPdoublependulum):
    # Soft receding, soft terminal
    def __init__(self, nlp_solver_type, time_step, tot_time, nn_params, mean, std, safety_margin, regenerate):
        # inherit initialization
        super().__init__(nlp_solver_type, time_step, tot_time)

        # nonlinear constraints
        # soft terminal
        self.model.con_h_expr_e = nn_decisionfunction_conservative(nn_params, mean, std, safety_margin, self.x)

        self.ocp.constraints.lh_e = np.array([0.])
        self.ocp.constraints.uh_e = np.array([1e6])

        self.ocp.constraints.idxsh_e = np.array([0])

        self.ocp.cost.zl_e = np.zeros((1,))  # weight soft terminal
        self.ocp.cost.zu_e = np.zeros((1,))
        self.ocp.cost.Zu_e = np.zeros((1,))
        self.ocp.cost.Zl_e = np.zeros((1,))

        # soft receding
        self.model.con_h_expr = nn_decisionfunction_conservative(nn_params, mean, std, safety_margin, self.x)

        self.ocp.constraints.lh = np.array([0.])
        self.ocp.constraints.uh = np.array([1e6])

        self.ocp.constraints.idxsh = np.array([0])

        self.ocp.cost.zl = np.zeros((1,))  # weight soft running
        self.ocp.cost.zu = np.zeros((1,))
        self.ocp.cost.Zu = np.zeros((1,))
        self.ocp.cost.Zl = np.zeros((1,))

        # ocp model
        self.ocp.model = self.model

        # solver
        self.ocp_solver = AcadosOcpSolver(self.ocp, build=regenerate)


class OCPdoublependulumReceidingHard(OCPdoublependulum):
    # Soft receding, soft terminal
    def __init__(self, nlp_solver_type, time_step, tot_time, nn_params, mean, std, safety_margin, regenerate):
        # inherit initialization
        super().__init__(nlp_solver_type, time_step, tot_time)

        # nonlinear constraints
        # soft terminal
        self.model.con_h_expr_e = nn_decisionfunction_conservative(nn_params, mean, std, safety_margin, self.x)

        self.ocp.constraints.lh_e = np.array([0.])
        self.ocp.constraints.uh_e = np.array([1e6])

        self.ocp.constraints.idxsh_e = np.array([0])

        self.ocp.cost.zl_e = np.zeros((1,))  # weight soft terminal
        self.ocp.cost.zu_e = np.zeros((1,))
        self.ocp.cost.Zu_e = np.zeros((1,))
        self.ocp.cost.Zl_e = np.zeros((1,))

        # hard receding
        self.model.con_h_expr = nn_decisionfunction_conservative(nn_params, mean, std, safety_margin, self.x)

        self.ocp.constraints.lh = np.array([0.])
        self.ocp.constraints.uh = np.array([1e6])

        # ocp model
        self.ocp.model = self.model

        # solver
        self.ocp_solver = AcadosOcpSolver(self.ocp, build=regenerate)


class OCPBackupController(MODELdoublependulum):
    def __init__(self, time_step, tot_time, regenerate):
        # inherit initialization
        super().__init__(time_step, tot_time)

        self.ocp = AcadosOcp()

        # times
        self.ocp.solver_options.tf = self.tot_time
        self.ocp.dims.N = int(self.tot_time / self.time_step)

        self.nx = self.model.x.size()[0]
        self.nu = self.model.u.size()[0]
        self.ny = self.nx + self.nu
        self.ny_e = self.nx

        # cost
        self.Q = 1e2 * np.diag([1e-4, 1e-4, 1., 1.])
        self.R = np.diag([1e-4, 1e-4])

        self.ocp.cost.W_e = self.Q
        self.ocp.cost.W = lin.block_diag(self.Q, self.R)

        self.ocp.cost.cost_type = "LINEAR_LS"
        self.ocp.cost.cost_type_e = "LINEAR_LS"

        self.ocp.cost.Vx = np.zeros((self.ny, self.nx))
        self.ocp.cost.Vx[: self.nx, : self.nx] = np.eye(self.nx)
        self.ocp.cost.Vu = np.zeros((self.ny, self.nu))
        self.ocp.cost.Vu[self.nx:, :self.nu] = np.eye(self.nu)
        self.ocp.cost.Vx_e = np.eye(self.nx)

        # set constraints
        self.Cmax = 10.
        self.thetamax = np.pi / 4 + np.pi
        self.thetamin = - np.pi / 4 + np.pi
        self.dthetamax = 10.

        # reference
        self.yref = np.zeros((self.ny,))
        self.ocp.cost.yref = self.yref
        self.ocp.cost.yref_e = self.yref[:self.ny_e]

        self.Cmax_limits = np.array([self.Cmax, self.Cmax])
        self.Cmin_limits = np.array([-self.Cmax, -self.Cmax])
        self.Xmax_limits = np.array(
            [self.thetamax, self.thetamax, self.dthetamax, self.dthetamax])
        self.Xmin_limits = np.array(
            [self.thetamin, self.thetamin, -self.dthetamax, -self.dthetamax])

        self.ocp.constraints.lbu = self.Cmin_limits
        self.ocp.constraints.ubu = self.Cmax_limits
        self.ocp.constraints.idxbu = np.array([0, 1])
        self.ocp.constraints.lbx = self.Xmin_limits
        self.ocp.constraints.ubx = self.Xmax_limits
        self.ocp.constraints.idxbx = np.array([0, 1, 2, 3])

        self.ocp.constraints.lbx_0 = self.Xmin_limits
        self.ocp.constraints.ubx_0 = self.Xmax_limits
        self.ocp.constraints.idxbx_0 = np.array([0, 1, 2, 3])

        # Final velocity constraint to be zero
        self.Xmax_zerovel_e = np.array([self.thetamax, self.thetamax, 0.2, 0.2])
        self.Xmin_zerovel_e = np.array([self.thetamin, self.thetamin, -0.2, -0.2])

        self.ocp.constraints.lbx_e = self.Xmin_zerovel_e
        self.ocp.constraints.ubx_e = self.Xmax_zerovel_e
        self.ocp.constraints.idxbx_e = np.array([0, 1, 2, 3])

        # options
        self.ocp.solver_options.nlp_solver_type = "SQP"
        self.ocp.solver_options.hessian_approx = 'EXACT'
        self.ocp.solver_options.qp_solver_iter_max = 100
        # self.ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
        self.ocp.solver_options.nlp_solver_max_iter = 1000
        self.ocp.solver_options.globalization = "MERIT_BACKTRACKING"
        self.ocp.solver_options.alpha_reduction = 0.3
        self.ocp.solver_options.alpha_min = 1e-2
        self.ocp.solver_options.levenberg_marquardt = 1e-5

        # ocp model
        self.ocp.model = self.model

        # solver
        self.ocp_solver = AcadosOcpSolver(self.ocp, build=regenerate)


    def OCP_solve(self, x0, x_sol_guess, u_sol_guess):
        # Reset current iterate:
        self.ocp_solver.reset()

        self.ocp_solver.constraints_set(0, "lbx", x0)
        self.ocp_solver.constraints_set(0, "ubx", x0)

        # Set parameters, guesses and constraints:
        for i in range(self.ocp.dims.N):
            self.ocp_solver.set(i, 'x', x_sol_guess[i])
            self.ocp_solver.set(i, 'u', u_sol_guess[i])

        self.ocp_solver.set(self.ocp.dims.N, 'x', x_sol_guess[self.ocp.dims.N])

        # Solve the OCP:
        status = self.ocp_solver.solve()

        return status