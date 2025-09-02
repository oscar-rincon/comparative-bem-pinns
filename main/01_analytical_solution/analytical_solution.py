#%% Script start and time measurement
import time
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Record start time
start_time = time.time()

# Get script name
script_name = os.path.basename(__file__)

#%% Directory setup
current_dir = os.path.dirname(os.path.abspath(__file__))
utilities_dir = os.path.join(current_dir, '../../utilities')

# Change working directory
os.chdir(current_dir)
sys.path.insert(0, utilities_dir)

# Import functions
from analytical_solution_functions import sound_hard_circle_calc, mask_displacement
from plotting_functions import plot_exact_displacement

#%% Parameters and grid
r_i = np.pi/4  # Inner radius
l_se = np.pi    # Outer semi-length
k = 3          # Wave number
n_grid = 501   # Number of grid points in x and y 

Y, X = np.mgrid[-l_se:l_se:n_grid*1j, -l_se:l_se:n_grid*1j]
R_exact = np.sqrt(X**2 + Y**2)

#%% Compute displacements
u_inc_exact, u_scn_exact, u_exact = sound_hard_circle_calc(k, r_i, X, Y, n_terms=None)

u_inc_exact = mask_displacement(R_exact, r_i, l_se, u_inc_exact)
u_scn_exact = mask_displacement(R_exact, r_i, l_se, u_scn_exact)
u_exact = mask_displacement(R_exact, r_i, l_se, u_exact)

#%% Plot
plot_exact_displacement(X, Y, np.real(u_inc_exact), np.real(u_scn_exact), np.real(u_exact),
                        np.imag(u_inc_exact), np.imag(u_scn_exact), np.imag(u_exact))

#%% Record runtime and save to .txt
end_time = time.time()
elapsed_time = end_time - start_time

log_text = f"Script: {script_name}\nExecution time (s): {elapsed_time:.2f}\n"

log_filename = os.path.splitext(script_name)[0] + "_log.txt"
with open(log_filename, "w") as f:
    f.write(log_text)

print(f"Log saved to {log_filename}")
