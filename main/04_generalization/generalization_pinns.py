 
#%% ======================== IMPORTS ========================
# Standard library imports
import sys
import os
import numpy as np
import torch
from scipy.interpolate import griddata

#%% ======================== PATH SETUP ========================
# Set the current directory and utilities path
current_dir = os.path.dirname(os.path.abspath(__file__))
utilities_dir = os.path.join(current_dir, '../../utilities')

# Change the working directory to the notebook's directory
os.chdir(current_dir)

# Modify the module search path to include utilities directory
sys.path.insert(0, utilities_dir)

#%% ======================== FUNCTION IMPORTS ========================
from analytical_solution_functions import sound_hard_circle_calc 
from analytical_solution_functions import mask_displacement
from plotting_functions import plot_pinns_displacements_with_errorline
from pinns_solution_functions import initialize_and_load_model
from pinns_solution_functions import predict_displacement_pinns 
from pinns_solution_functions import process_displacement_pinns

#%% ======================== PARAMETERS ========================
# Parameters
r_i = np.pi / 4  # Inner radius
l_se = 2 * np.pi  # Outer semi-length
k = 3  # Wave number
n_grid = 2 * 501  # Number of grid points in x and y
r_exclude = np.pi / 4 # Radius of excluded circular region

#%% ======================== ANALYTICAL SOLUTION ========================
# Create a grid of points in the domain
Y, X = np.mgrid[-l_se:l_se:n_grid*1j, -l_se:l_se:n_grid*1j]

# Calculate the radial distance from the origin for each point in the grid
R_exact = np.sqrt(X**2 + Y**2)

# Calculate the displacement for a sound-hard circular obstacle
u_inc_exact, u_scn_exact, u_exact = sound_hard_circle_calc(k, r_i, X, Y, n_terms=None)

# Mask the displacement
u_inc_exact = mask_displacement(R_exact, r_i, l_se, u_inc_exact)
u_scn_exact = mask_displacement(R_exact, r_i, l_se, u_scn_exact)
u_exact = mask_displacement(R_exact, r_i, l_se, u_exact)

#%% ======================== MODEL SETUP ========================
# Additional parameters
n_Omega_P = 10_000  # Number of points inside the annular region
n_Gamma_I = 100  # Number of points on the inner boundary (r = r_i)
n_Gamma_E = 250  # Number of points on the outer boundary (r = r_e)
l_e = 2 * np.pi  # Length of the semi-edge of the square
k = 3.0  # Wave number
iter = 0  # Iteration counter
side_length = 2 * l_e  # Side length of the square

# Initialize and load the model
model_path = 'models/Scattering_2_75.pt'
model = initialize_and_load_model(model_path, 2, 75)

#%% ======================== PINNs PREDICTION & PROCESSING ========================
# Predict the displacement
u_sc_amp_pinns, u_sc_phase_pinns, u_amp_pinns, u_phase_pinns = predict_displacement_pinns(model, l_e, r_i, k, n_grid)

# Process the displacement
u_sc_amp_pinns, u_sc_phase_pinns, u_amp_pinns, u_phase_pinns, diff_uscn_amp_pinns, diff_u_scn_phase_pinns = process_displacement_pinns(
    model, l_e, r_i, k, n_grid, X, Y, R_exact, u_scn_exact
)

#%% ======================== ERROR COMPUTATION ========================
# Compute relative L2 error (real part of scattered field)
R_grid = np.sqrt(X**2 + Y**2)
u_scn_exact_masked = np.copy(u_scn_exact)
u_scn_amp_masked = np.copy(u_sc_amp_pinns)
u_scn_exact_masked[R_grid < r_i] = 0
u_scn_amp_masked[R_grid < r_i] = 0
relative_error = np.linalg.norm(u_scn_exact_masked.real - u_scn_amp_masked.real, 2) / np.linalg.norm(u_scn_exact_masked.real, 2)
print(f"Relative L2 error: {relative_error:.4e}")

#%% ======================== ERROR LINE PROFILE ========================
# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Radial line from the center towards +x, with y=0
x_line = np.linspace(np.pi, 10 * np.pi, 500)
y_line = np.zeros_like(x_line)

# Convert X and Y data to PyTorch tensors and reshape
X_ten = torch.tensor(x_line).float().reshape(-1, 1).to(device)
Y_ten = torch.tensor(y_line).float().reshape(-1, 1).to(device)

# Initialize and load the model
model_path = 'models/Scattering_2_75.pt'
model = initialize_and_load_model(model_path, 2, 75)

# Concatenate X and Y tensors into a single tensor
domain_ten = torch.cat([X_ten, Y_ten], dim=1)
u_sc_pred = model(domain_ten)
u_sc_amp_pred = u_sc_pred[:, 0].detach().cpu().numpy().reshape(x_line.shape)
u_sc_phase_pred = u_sc_pred[:, 1].detach().cpu().numpy().reshape(x_line.shape)

# Solución exacta
X_line, Y_line = x_line, y_line
u_inc_line, u_scn_exact_line, u_tot_exact_line = sound_hard_circle_calc(
    k, r_exclude,
    X_line, Y_line
)

error_line = np.abs(np.real(u_scn_exact_line) - np.real(u_sc_amp_pred))
rel_error_line = error_line / np.max(np.real(u_scn_exact_line))

#%% ======================== PLOTTING ========================
plot_pinns_displacements_with_errorline(
    X, Y,
    u_sc_amp_pinns,
    np.real(u_inc_exact) + u_sc_amp_pinns,
    np.abs(np.real(u_scn_exact) - u_sc_amp_pinns),
    u_sc_phase_pinns,
    u_sc_phase_pinns + np.real(u_inc_exact),
    np.abs(np.imag(u_scn_exact) - u_sc_phase_pinns),
    x_line,
    rel_error_line
)

# %%
