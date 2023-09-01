from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSim, AcadosSimSolver
import numpy as np
from acados_template import AcadosModel
from casadi import MX, vertcat, cos, sin, fmax, norm_2
import scipy.linalg as lin
import l4casadi as l4c


class MODELdoublependulum:
    def __init__(self, time_step, tot_time):
        model_name = "double_pendulum"

        # constants
        self.m1 = 0.4  # mass of the first link [kself.g]
        self.m2 = 0.4  # mass of the second link [kself.g]
        self.g = 9.81  # self.gravity constant [m/s^2]
        self.l1 = 0.8  # lenself.gth of the first link [m]
        self.l2 = 0.8  # lenself.gth of the second link [m]

        self.time_step = time_step
        self.tot_time = tot_time

        # states
        self.x = MX.sym('x', 4)

        # xdot
        xdot = MX.sym('x_dot', 4)

        # controls
        u = MX.sym('u', 2)

        # parameters
        p = []

        # dynamics
        f_expl = vertcat(
            self.x[2],
            self.x[3],
            (
                    self.l1 ** 2
                    * self.l2
                    * self.m2
                    * self.x[2] ** 2
                    * sin(-2 * self.x[1] + 2 * self.x[0])
                    + 2 * u[1] * cos(-self.x[1] + self.x[0]) * self.l1
                    + 2
                    * (
                            self.g * sin(-2 * self.x[1] + self.x[0]) * self.l1 * self.m2 / 2
                            + sin(-self.x[1] + self.x[0]) * self.x[3] ** 2 * self.l1 * self.l2 * self.m2
                            + self.g * self.l1 * (self.m1 + self.m2 / 2) * sin(self.x[0])
                            - u[0]
                    )
                    * self.l2
            )
            / self.l1 ** 2
            / self.l2
            / (self.m2 * cos(-2 * self.x[1] + 2 * self.x[0]) - 2 * self.m1 - self.m2),
            (
                    -self.g
                    * self.l1
                    * self.l2
                    * self.m2
                    * (self.m1 + self.m2)
                    * sin(-self.x[1] + 2 * self.x[0])
                    - self.l1
                    * self.l2 ** 2
                    * self.m2 ** 2
                    * self.x[3] ** 2
                    * sin(-2 * self.x[1] + 2 * self.x[0])
                    - 2
                    * self.x[2] ** 2
                    * self.l1 ** 2
                    * self.l2
                    * self.m2
                    * (self.m1 + self.m2)
                    * sin(-self.x[1] + self.x[0])
                    + 2 * u[0] * cos(-self.x[1] + self.x[0]) * self.l2 * self.m2
                    + self.l1
                    * (self.m1 + self.m2)
                    * (sin(self.x[1]) * self.g * self.l2 * self.m2 - 2 * u[1])
            )
            / self.l2 ** 2
            / self.l1
            / self.m2
            / (self.m2 * cos(-2 * self.x[1] + 2 * self.x[0]) - 2 * self.m1 - self.m2)
        )

        self.model = AcadosModel()

        f_impl = xdot - f_expl

        self.model.f_expl_expr = f_expl
        self.model.f_impl_expr = f_impl
        self.model.x = self.x
        self.model.xdot = xdot
        self.model.u = u
        self.model.p = p
        self.model.name = model_name


class SYMdoublependulum(MODELdoublependulum):
    def __init__(self, time_step, tot_time, regenerate):
        # inherit initialization
        super().__init__(time_step, tot_time)

        sim = AcadosSim()
        sim.model = self.model
        sim.solver_options.T = self.time_step
        sim.solver_options.num_stages = 4
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


class OCPdoublependulumHardTerm(OCPdoublependulum):
    def __init__(self, nlp_solver_type, time_step, tot_time, nn_model, mean, std, safety_margin, regenerate):
        # inherit initialization
        super().__init__(nlp_solver_type, time_step, tot_time)

        mean_vec = vertcat(mean, mean, 0., 0.)
        vel_norm = fmax(norm_2(self.x[2:]), 1e-3)
        std_vec = vertcat(std, std, vel_norm, vel_norm)

        l4c_model = l4c.L4CasADi(nn_model, model_expects_batch_dim=True, device='cpu')

        # nonlinear constraints
        self.model.con_h_expr_e = l4c_model((self.x - mean_vec) / std_vec) * (100 - safety_margin) / 100 - vel_norm

        self.ocp.constraints.lh_e = np.array([0.])
        self.ocp.constraints.uh_e = np.array([1e6])

        self.ocp.solver_options.model_external_shared_lib_dir = l4c_model.shared_lib_dir
        self.ocp.solver_options.model_external_shared_lib_name = l4c_model.name

        # ocp model
        self.ocp.model = self.model

        # solver
        self.ocp_solver = AcadosOcpSolver(self.ocp, build=regenerate)





