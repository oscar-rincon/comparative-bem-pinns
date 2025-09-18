 
#%% ======================== IMPORTS ========================
# Standard library imports
import sys
import os
import time
import numpy as np
import torch
from torch import nn
from scipy.interpolate import griddata
import pandas as pd

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

#%% ======================== PARAMETERS ========================
# Parameters
r_i = np.pi / 4  # Inner radius
l_se = 10 * np.pi  # Outer semi-length
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
l_e = 10 * np.pi  # Length of the semi-edge of the square
k = 3.0  # Wave number
iter = 0  # Iteration counter
side_length = 2 * l_e  # Side length of the square

# Initialize and load the model
model_path = 'models/Scattering_3_75.pt'
model = initialize_and_load_model(model_path, 3, 75, nn.Tanh())

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
print(f"Relative L2 error: {relative_error:.2e}")

# Save result to txt
with open("logs/error_results.txt", "w") as f:
    f.write("Relative L2 error computation\n")
    f.write("=============================\n")
    f.write(f"Relative L2 error: {relative_error:.6e}\n")

#%% ======================== ERROR LINE PROFILE ========================
# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Radial line from the center towards +x, with y=0
x_line = np.linspace(np.pi, 2 * np.pi, 200)
y_line = np.zeros_like(x_line)

# Convert X and Y data to PyTorch tensors and reshape
X_ten = torch.tensor(x_line).float().reshape(-1, 1).to(device)
Y_ten = torch.tensor(y_line).float().reshape(-1, 1).to(device)

# Initialize and load the model
model_path = 'models/Scattering_3_75.pt'
model = initialize_and_load_model(model_path, 3, 75, nn.Tanh())

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

error_line = np.abs(np.real(u_scn_exact_line) - u_sc_amp_pred)
rel_error_line = error_line / np.max(u_inc_line + u_sc_amp_pred)

#%% ======================== PLOTTING ========================
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, FuncFormatter
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

def plot_pinns_displacements_with_errorline(X, Y, u_inc_amp, u_scn_amp, u_amp,
                                            u_inc_phase, u_scn_phase, u_phase,
                                            x_line, rel_error_line):
    """
    Combina gráficos 2D y gráfico de error relativo 1D usando GridSpec.
    """

    # Square patch properties
    square_size = 2 * np.pi
    square_xy = (-square_size / 2, -square_size / 2)
    square_props = dict(edgecolor="gray", facecolor="none", lw=0.8)
    shrink = 0.8
    decimales = 1e+4

    # Choose slice along y = 0
    y_index = np.argmin(np.abs(Y[:, 0]))  # row closest to y=0

    X_slice = X[y_index, :]
    amp_slice = np.abs(u_amp[y_index, :]) / np.abs(u_scn_amp).max()

    # Restrict to [pi, 10pi]
    mask = (X_slice >= np.pi) & (X_slice <= 10*np.pi)
    X_slice = X_slice[mask]
    amp_slice = amp_slice[mask]

    # Create figure and GridSpec layout
    fig = plt.figure(figsize=(3.6, 6.0))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 0.7], hspace=0.5, wspace=0.05)

    # Subplots for amplitude
    ax0 = fig.add_subplot(gs[0, 0])
    c1 = ax0.pcolormesh(X, Y, u_inc_amp, cmap="RdYlBu", rasterized=True, vmin=-1.5, vmax=1.5)
    cb1 = fig.colorbar(c1, ax=ax0, shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb1.set_label(r"$u_{\rm{sct}}$", fontsize=8)
    cb1.set_ticks([-1.5, 1.5])
    cb1.set_ticklabels([f'{-1.5}', f'{1.5}'], fontsize=8)
    ax0.set_xlim(-2*np.pi, 2*np.pi)
    ax0.set_ylim(-2*np.pi, 2*np.pi)
    ax0.add_patch(Rectangle(square_xy, square_size, square_size, **square_props))
    ax0.axis("off")
    ax0.set_aspect("equal")


    ax1 = fig.add_subplot(gs[0, 1])
    c2 = ax1.pcolormesh(X, Y, np.abs(u_amp)/np.abs(u_scn_amp).max(), cmap="magma", rasterized=True)
    cb2 = fig.colorbar(c2, ax=ax1, shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb2.set_label(r"|Error| / max($u$)", fontsize=8)
    cb2.set_ticks([0, np.max(np.abs(u_amp)/np.abs(u_scn_amp).max())])
    cb2.set_ticklabels([f'{0:.1f}', f'{np.max(np.abs(u_amp)/np.abs(u_scn_amp).max()):.4f}'], fontsize=8)
    ax1.add_patch(Rectangle(square_xy, square_size, square_size, **square_props))
    ax1.axis("off")
    ax1.set_aspect("equal")
    ax1.set_xlim(-2*np.pi, 2*np.pi)
    ax1.set_ylim(-2*np.pi, 2*np.pi)
    y_center = 0  # or e.g., y_center = Y.mean() or another value of interest
    line2 = Line2D([np.pi, 2*np.pi], [y_center, y_center], color="#00ff0d", linewidth=1.0, linestyle='-')
    ax1.add_line(line2)

    # Subplots for phase
    ax2 = fig.add_subplot(gs[1, 0])
    c3 = ax2.pcolormesh(X, Y, u_inc_phase, cmap="twilight_shifted", rasterized=True, vmin=-(np.pi), vmax=(np.pi))
    cb3 = fig.colorbar(c3, ax=ax2, shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb3.set_label(r"$u_{\rm{sct}}$", fontsize=8)
    cb3.set_ticks([-(np.pi), (np.pi)])
    cb3.set_ticklabels([r'-$\pi$', r'$\pi$'], fontsize=8)  
    ax2.add_patch(Rectangle(square_xy, square_size, square_size, **square_props))
    ax2.axis("off")
    ax2.set_aspect("equal")
    ax2.set_xlim(-2*np.pi, 2*np.pi)
    ax2.set_ylim(-2*np.pi, 2*np.pi)       

    ax3 = fig.add_subplot(gs[1, 1])
    c4 = ax3.pcolormesh(X, Y, np.abs(u_phase)/np.abs(u_scn_phase).max(), cmap="magma", rasterized=True)
    cb4 = fig.colorbar(c4, ax=ax3, shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb4.set_label(r"|Error| / max($u$)", fontsize=8)
    cb4.set_ticks([0, np.max(np.abs(u_phase)/np.abs(u_scn_phase).max())])
    cb4.set_ticklabels([f'{0:.1f}', f'{np.max(np.abs(u_phase)/np.abs(u_scn_phase).max()):.4f}'], fontsize=8)
    ax3.add_patch(Rectangle(square_xy, square_size, square_size, **square_props))
    ax3.axis("off")
    ax3.set_aspect("equal")
    ax3.set_xlim(-2*np.pi, 2*np.pi)
    ax3.set_ylim(-2*np.pi, 2*np.pi)

    # Subplot 5: Relative error line plot
    ax_err = fig.add_subplot(gs[2, :])
    #ax_err.axvline(x=np.pi, color="#acacac", linestyle='-', linewidth=1)
    #ax_err.axvline(x=2*np.pi, color="#acacac", linestyle='-', linewidth=1)  
    ax_err.axvspan(np.pi, 2*np.pi, color='gray', alpha=0.2)   
    ax_err.plot(X_slice,amp_slice, label='Relative error', color="#00ff0d")
    #ax_err.plot(x_line, rel_error_line, label='Relative error', color="#00ff0d")
    # Agregar líneas verticales en pi y 2pi
    ax_err.set_xlabel(r'$x$', fontsize=8)
    ax_err.set_ylabel(r"$|$Error$|$ / max($u$)", fontsize=8)
    ax_err.set_ylim(0, 0.6)
    #ax_err.set_ylim(0, np.max(rel_error_line) * 1.1)
    #ax_err.set_xlim(0, 2*np.pi)
    ax_err.xaxis.set_major_locator(MultipleLocator(base=np.pi))

    def format_func(value, tick_number):
        N = int(np.round(value / np.pi))
        if N == 0:
            return "0"
        elif N == 1:
            return r"$\pi$"
        elif N == -1:
            return r"$-\pi$"
        else:
            return fr"${N}\pi$"

    ax_err.xaxis.set_major_formatter(FuncFormatter(format_func))
    ax_err.set_title('PINNs - Amplitude', fontsize=8)

    # Add rotated labels
    fig.text(0.08, 0.79, r'PINNs - Amplitude', fontsize=8, va='center', ha='center', rotation='vertical')
    fig.text(0.08, 0.48, r'PINNs - Phase', fontsize=8, va='center', ha='center', rotation='vertical')

    # Save and show
    plt.savefig("figures/generalization_pinns.svg", dpi=300, bbox_inches='tight')
    plt.show()

# 
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

#%% Record runtime and save to .txt
end_time = time.time()
elapsed_time = end_time - start_time
 
# Build log text
log_text = f"Script: {script_name}\nExecution time (s): {elapsed_time:.2f}\n"

# Define log filename inside the logs folder
log_filename = os.path.join(output_folder, f"{script_name}_log.txt")

# Write log file
with open(log_filename, "w") as f:
    f.write(log_text)

print(f"Log saved to: {log_filename}")

# %%
