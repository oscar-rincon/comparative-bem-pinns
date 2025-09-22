

# ============================================================
"""
Script: comparison_pinns.py

Description:
    This script evaluates the generalization of a trained 
    Physics-Informed Neural Network (PINN) for the scattering 
    problem of a sound-hard circular obstacle. It loads a 
    pre-trained model, predicts displacements on a square 
    domain, and compares results with the analytical solution. 

Inputs:
    - Pre-trained PINN model (path: ./models/)
    - Problem parameters: wave number k, inner radius, domain size.
    - Analytical displacement fields for validation.

Outputs:
    - Visualization of PINN prediction errors (amplitude & phase)
    - Relative L2 error printed to console
    - Log file (TXT) with script name and execution time, saved in ./logs/
"""
# ============================================================

#%%
# Standard library imports
from datetime import datetime
import sys
import os
import time
import numpy as np
from scipy.interpolate import griddata
from torch import nn
# Set the current directory and utilities path
current_dir = os.path.dirname(os.path.abspath(__file__))
utilities_dir = os.path.join(current_dir, '../../utilities')

# Change the working directory to the notebook's directory
os.chdir(current_dir)

# Modify the module search path to include utilities directory
sys.path.insert(0, utilities_dir)

import importlib
import analytical_solution_functions
import bem_solution_functions
import plotting_functions

# Reload them each time this file runs
importlib.reload(analytical_solution_functions)
importlib.reload(bem_solution_functions)
importlib.reload(plotting_functions)

# Import Functions
from analytical_solution_functions import sound_hard_circle_calc 
from analytical_solution_functions import mask_displacement
from plotting_functions import plot_pinns_error
from pinns_solution_functions import initialize_and_load_model
from pinns_solution_functions import predict_displacement_pinns 
from pinns_solution_functions import process_displacement_pinns
from pinns_solution_functions import set_seed

set_seed(42)

#%% Start time measurement
# Record start time
start_time = time.time()

# Get script name without extension
script_name = os.path.splitext(os.path.basename(__file__))[0]

# Define output folder (e.g., "logs" inside the current script directory)
output_folder = os.path.join(os.path.dirname(__file__), "logs")

# Create folder if it does not exist
os.makedirs(output_folder, exist_ok=True)

# Define output file path
output_file = os.path.join(output_folder, f"{script_name}_log.txt")

#%%
r_i = np.pi/4 # Inner radius
l_se = np.pi # Outer semi-length
k = 3  # Wave number
n_grid = 501 # Number of grid points in x and y 

# Create a grid of points in the domain
Y, X = np.mgrid[-l_se:l_se:n_grid*1j, -l_se:l_se:n_grid*1j]


# Calculate the radial distance from the origin for each point in the grid
R_exact = np.sqrt(X**2 + Y**2)

# Calculate the displacement for a sound-hard circular obstacle
# n_terms: number of terms in the series expansion
u_inc_exact, u_scn_exact, u_exact = sound_hard_circle_calc(k, r_i, X, Y, n_terms=None)

# Mask the displacement
u_inc_exact = mask_displacement(R_exact, r_i, l_se, u_inc_exact)
u_scn_exact = mask_displacement(R_exact, r_i, l_se, u_scn_exact)
u_exact = mask_displacement(R_exact, r_i, l_se, u_exact)


# Parameters
n_Omega_P = 10_000        # Number of points inside the annular region
n_Gamma_I = 100          # Number of points on the inner boundary (r = r_i)
n_Gamma_E = 250          # Number of points on the outer boundary (r = r_e)
r_i = np.pi / 4          # Inner radius
l_e = np.pi              # Length of the semi-edge of the square
k = 3.0                  # Wave number
iter = 0                 # Iteration counter
side_length = 2 * l_e    # Side length of the square
 
# Initialize and load the model
model_path = 'models/3_layers_50_neurons.pt'
model = initialize_and_load_model(model_path, 3, 50, nn.Tanh())

# Predict the displacement
u_sc_amp_pinns, u_sc_phase_pinns, u_amp_pinns, u_phase_pinns = predict_displacement_pinns(model, l_e, r_i, k, n_grid)

# Example usage
u_sc_amp_pinns,u_sc_phase_pinns,u_amp_pinns, u_phase_pinns, diff_uscn_amp_pinns, diff_u_scn_phase_pinns = process_displacement_pinns(
    model, l_e, r_i, k, n_grid, X, Y, R_exact, u_scn_exact
)

"""
Compute relative L2 error (real part of scattered field)
"""

# Create masked copies to zero-out interior region
R_grid = np.sqrt(X**2 + Y**2)
u_scn_exact_masked = np.copy(u_scn_exact)
u_scn_amp_masked   = np.copy(u_sc_amp_pinns)
u_scn_exact_masked[R_grid < r_i] = 0
u_scn_amp_masked[R_grid < r_i] = 0

relative_error = np.linalg.norm(u_scn_exact_masked.real - u_scn_amp_masked.real, 2) / \
                 np.linalg.norm(u_scn_exact_masked.real, 2)
print(f"Relative L2 error: {relative_error:.4e}")
#%%
plot_pinns_error(
    X, Y,
    u_sc_amp_pinns,
    np.real(u_inc_exact) + u_sc_amp_pinns,
    np.abs(np.real(u_scn_exact) - u_sc_amp_pinns),
    u_sc_phase_pinns,
    u_sc_phase_pinns + np.real(u_inc_exact),
    np.abs(np.imag(u_scn_exact) - u_sc_phase_pinns)
)

#%% Record runtime and save to .txt
end_time = time.time()
elapsed_time = end_time - start_time

# Build log text
log_text = f"Script: {script_name}\nExecution time (s): {elapsed_time:.2f}\n"

# Get current date and time
date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Define log filenames inside the logs folder
log_filename_with_date = os.path.join(output_folder, f"{script_name}_log_{date_str}.txt")
log_filename_no_date   = os.path.join(output_folder, f"{script_name}_log.txt")

# Write log file with date
with open(log_filename_with_date, "w") as f:
    f.write(log_text)

# Write log file without date
with open(log_filename_no_date, "w") as f:
    f.write(log_text)

print(f"Log saved to: {log_filename_with_date}")
print(f"Log also saved to: {log_filename_no_date}")
