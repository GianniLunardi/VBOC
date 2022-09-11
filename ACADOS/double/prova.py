import cProfile
import numpy as np
import matplotlib.pyplot as plt
import time
from double_pendulum_ocp_class import OCPdoublependulumINIT, OCPdoublependulumNN
import warnings
from scipy.stats import entropy, qmc

warnings.filterwarnings("ignore")


with cProfile.Profile() as pr:

    start_time = time.time()

    # Ocp initialization:
    ocp = OCPdoublependulumINIT()

    nx = ocp.nx
    nu = ocp.nu
    N = ocp.N
    ocp_solver = ocp.ocp_solver
    Tf = ocp.Tf

    # Position and velocity bounds:
    v_max = ocp.dthetamax
    v_min = -ocp.dthetamax
    q_max = ocp.thetamax
    q_min = ocp.thetamin
    Cmax = ocp.Cmax
    
    # Generate low-discrepancy unlabeled samples:
    sampler = qmc.Halton(d=2, scramble=False)
    sample = sampler.random(n=pow(100, 2))
    l_bounds = [q_min, q_min]
    u_bounds = [q_max, q_max]
    Xu_iter = qmc.scale(sample, l_bounds, u_bounds).tolist()
    
    U = 0
    
    # Training of an initial classifier:
    for n in range(len(Xu_iter)):
        q0 = Xu_iter[n]
        v0 = [0.0, 0.0]
        # Data testing:
        res = ocp.compute_problem(q0, v0)
        Ux = ocp_solver.get(0, "u")
        print(Ux[0])
        Ux = abs(Ux[0])
        if Ux>U:
            U = Ux
		    
    print(U)
