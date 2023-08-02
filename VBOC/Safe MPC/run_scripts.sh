#!/bin/bash 

cd no_constraints/
python3 3dof_sym.py
cd ../hard_terminal_constraints/
python3 3dof_sym.py
cd ../soft_traj_constraints/
python3 3dof_sym.py
cd ../receiding_hard_constraints/
python3 3dof_sym.py
