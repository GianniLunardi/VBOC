from acados_template import AcadosModel
from casadi import SX, vertcat, sin, Function, exp, norm_2


def export_pendulum_ode_model():

    model_name = 'pendulum_ode'

    # constants
    m = 0.4  # mass of the ball [kg]
    g = 9.81  # gravity constant [m/s^2]
    l = 0.8  # length of the rod [m]
    b = 0.1  # damping

    # set up states & controls
    theta = SX.sym('theta')
    dtheta = SX.sym('dtheta')

    x = vertcat(theta, dtheta)

    # controls
    F = SX.sym('F')
    u = vertcat(F)

    # xdot
    theta_dot = SX.sym('theta_dot')
    dtheta_dot = SX.sym('dtheta_dot')

    xdot = vertcat(theta_dot, dtheta_dot)

    # parameters
    p = []

    # dynamics
    f_expl = vertcat(dtheta, (g*sin(theta)+F/m-b*dtheta/m)/(l*l))

    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.p = p
    model.name = model_name

    # model.con_h_expr_e = vertcat(clf_decisionfunction(clf, X_iter, x))

    return model


def clf_decisionfunction(clf, X_iter, x):
    dual_coef = clf.dual_coef_
    sup_vec = clf.support_vectors_
    const = clf.intercept_
    output = 0
    for i in range(sup_vec.shape[0]):
        output += dual_coef[0, i] * \
            exp(- (norm_2(x - sup_vec[i])**2)/(2*X_iter.var()))
    output += const

    return output


def export_pendulum_ode_model_with_discrete_rk4(dT):

    model = export_pendulum_ode_model()

    x = model.x
    u = model.u
    nx = x.size()[0]

    ode = Function('ode', [x, u], [model.f_expl_expr])
    # set up RK4
    k1 = ode(x,       u)
    k2 = ode(x+dT/2*k1, u)
    k3 = ode(x+dT/2*k2, u)
    k4 = ode(x+dT*k3,  u)
    xf = x + dT/6 * (k1 + 2*k2 + 2*k3 + k4)

    model.disc_dyn_expr = xf
    print("built RK4 for pendulum model with dT = ", dT)
    print(xf)
    return model


def export_augmented_pendulum_model():
    # pendulum model augmented with algebraic variable just for testing
    model = export_pendulum_ode_model()
    model_name = 'augmented_pendulum'

    z = SX.sym('z')

    f_impl = vertcat(model.xdot - model.f_expl_expr,
                     z - model.u*model.u)

    model.f_impl_expr = f_impl
    model.z = z
    model.name = model_name

    return model
