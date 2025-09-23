# ============================================================
"""
Script: generalization_bem.py

Description:
    This script computes the scattered acoustic field produced by a 
    sound-hard circular obstacle using the Boundary Element Method (BEM). 
    It generates boundary meshes, applies Neumann boundary conditions, 
    solves the exterior boundary integral equation, interpolates results 
    on a uniform grid, and compares with the analytical solution. 

    The script also computes relative L2 errors and produces a set of plots 
    showing amplitudes, phases, and error distributions.

Inputs:
    - Problem parameters: wave number k, number of elements, grid resolution,
      obstacle radius, and domain size.
    - Utility functions: analytical solution, BEM solvers, and plotting.

Outputs:
    - Plots of amplitude, phase, and relative error along radial lines.
    - Figure saved in ./figures/generalization_bem.svg
    - Log file (TXT) with script name, start/end times, and execution 
      duration, saved in ./logs/ with timestamped filename.
"""
# ============================================================

#%% ======================= IMPORTS =======================
# Standard library imports
import sys
import os
import time
from datetime import datetime
import numpy as np
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

#%% ======================= PATH SETUP =======================
current_dir = os.path.dirname(os.path.abspath(__file__))
utilities_dir = os.path.join(current_dir, '../../utilities')

# Change the working directory to the script directory
os.chdir(current_dir)

# Add utilities directory to Python path
sys.path.insert(0, utilities_dir)

#%% ======================= FUNCTION IMPORTS =======================
from analytical_solution_functions import sound_hard_circle_calc, mask_displacement
from bem_solution_functions import (
    Circle_n, solveExteriorBoundary, solveExterior, generateInteriorPoints_excluding_circle
)
from plotting_functions import plot_bem_displacements_errors

#%% ======================= START TIME =======================
start_time = time.time()
script_name = os.path.splitext(os.path.basename(__file__))[0]

# Define output folder for logs
output_folder = os.path.join(current_dir, "logs")
os.makedirs(output_folder, exist_ok=True)

#%% ======================= PROBLEM SETUP =======================
k = 3.0                # Wave number
n = 15                 # Boundary elements
n_dom = 2 * 40         # Domain sampling points per axis
r_exclude = np.pi / 4  # Obstacle radius
l_se = np.pi           # Half-length of computational domain
n_grid = 501           # Grid points

#%% ======================= BOUNDARY MESH =======================
aVertex, aElement = Circle_n(n=n, radius=r_exclude)
num_elements = aElement.shape[0]
aCenters = 0.5 * (aVertex[aElement[:, 0]] + aVertex[aElement[:, 1]])
theta = np.arctan2(aCenters[:, 1], aCenters[:, 0])  # Normal angle

#%% ======================= BOUNDARY CONDITIONS =======================
alpha = np.full(num_elements, 0.0, dtype=complex)
beta = np.full(num_elements, 1.0, dtype=complex)
f = np.empty(num_elements, dtype=complex)
phi = np.full(num_elements, 0.0, dtype=complex)
v = np.full(num_elements, 0.0, dtype=complex)

# Incident field + derivative
kx = k * aCenters[:, 0]
phi_inc = np.exp(1j * kx)
f = -1j * k * np.cos(theta) * phi_inc

#%% ======================= INTERIOR POINTS =======================
points_outside, points_inside = generateInteriorPoints_excluding_circle(
    Nx=n_dom, Ny=n_dom,
    xmin=-2 * l_se, xmax=2 * l_se,
    ymin=-2 * l_se, ymax=2 * l_se,
    r_exclude=r_exclude
)
interiorIncidentPhi = np.zeros(points_outside.shape[0], dtype=complex)

#%% ======================= SOLVE EXTERIOR BEM =======================
c, density = None, None
v, phi = solveExteriorBoundary(
    k, alpha, beta, f, phi, v,
    aVertex, aElement,
    c, density,
    'exterior'
)

#%% ======================= EVALUATE FIELD =======================
interiorPhi = solveExterior(
    k, v, phi,
    interiorIncidentPhi,
    points_outside,
    aVertex, aElement,
    'exterior'
)

#%% ======================= INTERPOLATION TO GRID =======================
Y, X = np.mgrid[-2*l_se:2*l_se:n_grid*1j, -2*l_se:2*l_se:n_grid*1j]
grid_z = griddata(points_outside, interiorPhi, (X, Y), method='cubic')
grid_z = np.ma.masked_where(np.sqrt(X**2 + Y**2) < r_exclude, grid_z)

u_scn_amp = grid_z.real   # amplitude
u_scn_phase = grid_z.imag # phase

#%% ======================= ANALYTICAL SOLUTION =======================
R_exact = np.sqrt(X**2 + Y**2)
u_inc_exact, u_scn_exact, u_exact = sound_hard_circle_calc(k, r_exclude, X, Y, n_terms=None)

# Mask solutions
u_inc_exact = mask_displacement(R_exact, r_exclude, l_se, u_inc_exact)
u_scn_exact = mask_displacement(R_exact, r_exclude, l_se, u_scn_exact)
u_exact = mask_displacement(R_exact, r_exclude, l_se, u_exact)

#%% ======================= ERROR COMPUTATION =======================
R_grid = np.sqrt(X**2 + Y**2)
u_scn_exact_masked = np.copy(u_scn_exact)
u_scn_amp_masked = np.copy(u_scn_amp)
u_scn_exact_masked[R_grid < r_exclude] = 0
u_scn_amp_masked[R_grid < r_exclude] = 0

relative_error = np.linalg.norm(
    u_scn_exact_masked.real - u_scn_amp_masked.real, 2
) / np.linalg.norm(u_scn_exact_masked.real, 2)
print(f"Relative L2 error: {relative_error:.2e}")

#%% ======================= RADIAL LINE PROFILE =======================
# Línea radial desde el centro hacia +x, con y=0
x_line = np.linspace(np.pi, 10 * np.pi, 500)
y_line = np.zeros_like(x_line)
points_line = np.vstack((x_line, y_line)).T

# Evaluar campo BEM en puntos de la línea
interiorIncidentPhi_line = np.zeros(points_line.shape[0], dtype=complex)
phi_bem_line = solveExterior(
    k, v, phi,
    interiorIncidentPhi_line,
    points_line,
    aVertex, aElement,
    'exterior'
)

# Solución exacta
X_line, Y_line = x_line, y_line
u_inc_line, u_scn_exact_line, u_tot_exact_line = sound_hard_circle_calc(
    k, r_exclude,
    X_line, Y_line
)

# Error relativo
error_line = np.abs(np.real(u_scn_exact_line) - np.real(phi_bem_line))
rel_error_line = error_line / np.max(np.real(u_scn_exact_line))


#%% ======================= PLOTTING =======================
 
plot_bem_displacements_errors(
    X, Y,
    u_scn_amp,
    np.real(u_inc_exact) + u_scn_amp,
    np.abs(np.real(u_scn_exact) - u_scn_amp),
    u_scn_phase,
    u_scn_phase + np.real(u_inc_exact),
    np.abs(np.imag(u_scn_exact) - u_scn_phase),
    x_line,
    rel_error_line
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

# %%
