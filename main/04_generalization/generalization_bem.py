#%% ======================= IMPORTS =======================
# Standard library imports
from datetime import datetime
import sys
import os
import time
import numpy as np
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

#%% ======================= PATH SETUP =======================
# Set the current directory and utilities path
current_dir = os.path.dirname(os.path.abspath(__file__))
#current_dir = os.getcwd()
utilities_dir = os.path.join(current_dir, '../../utilities')

# Change the working directory to the notebook's directory
os.chdir(current_dir)

# Modify the module search path to include utilities directory
sys.path.insert(0, utilities_dir)

#%% ======================= FUNCTION IMPORTS =======================
# Import necessary functions from utility modules
from analytical_solution_functions import sound_hard_circle_calc 
from analytical_solution_functions import mask_displacement
from bem_solution_functions import Circle_n
from bem_solution_functions import solveExteriorBoundary
from bem_solution_functions import solveExterior
from bem_solution_functions import generateInteriorPoints_excluding_circle
from plotting_functions import plot_bem_displacements_errors

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

#%% ======================= PROBLEM SETUP =======================
"""
Problem setup
"""
k = 3.0               # Wave number
n = 15                # Number of boundary elements (circular discretization)
n_dom = 2*40            # Number of domain sampling points (per axis)
r_exclude = np.pi / 4 # Radius of excluded circular region
l_se = np.pi          # Half-length of the computational domain
n_grid = 501          # Number of grid points along x and y

#%% ======================= BOUNDARY MESH =======================
"""
Generate circular boundary mesh
"""
aVertex, aElement = Circle_n(n=n, radius=r_exclude)
num_elements = aElement.shape[0]
aCenters = 0.5 * (aVertex[aElement[:, 0]] + aVertex[aElement[:, 1]])
theta = np.arctan2(aCenters[:, 1], aCenters[:, 0])  # Normal angle at each element center

#%% ======================= BOUNDARY CONDITIONS =======================
"""
Apply Neumann boundary conditions
"""
alpha = np.full(num_elements, 0.0, dtype=complex)
beta  = np.full(num_elements, 1.0, dtype=complex)
f     = np.empty(num_elements, dtype=complex)
phi   = np.full(num_elements, 0.0, dtype=complex)
v     = np.full(num_elements, 0.0, dtype=complex)

# Compute incident field and its normal derivative
kx = k * aCenters[:, 0]
phi_inc = np.exp(1j * kx)
f = -1j * k * np.cos(theta) * phi_inc  # ∂φ_inc/∂n = i k cos(θ) φ_inc

#%% ======================= INTERIOR POINTS =======================
"""
Generate interior points, excluding the circular obstacle
"""
points_outside, points_inside = generateInteriorPoints_excluding_circle(
    Nx=n_dom, Ny=n_dom,
    xmin=-2*l_se, xmax=2*l_se,
    ymin=-2*l_se, ymax=2*l_se,
    r_exclude=r_exclude
)
interiorIncidentPhi = np.zeros(points_outside.shape[0], dtype=complex)

#%% ======================= SOLVE EXTERIOR BEM =======================
"""
Solve boundary integral equation (exterior problem)
"""
c, density = None, None
v, phi = solveExteriorBoundary(
    k, alpha, beta, f, phi, v,
    aVertex, aElement,
    c, density,
    'exterior'
)

#%% ======================= EVALUATE FIELD =======================
"""
Evaluate scattered field at interior points
"""
interiorPhi = solveExterior(
    k, v, phi,
    interiorIncidentPhi,
    points_outside,
    aVertex, aElement,
    'exterior'
)

#%% ======================= INTERPOLATION TO GRID =======================
"""
Interpolate scattered field on a uniform grid
"""
Y, X = np.mgrid[-2*l_se:2*l_se:n_grid*1j, -2*l_se:2*l_se:n_grid*1j]
grid_z = griddata(points_outside, interiorPhi, (X, Y), method='cubic')
grid_z = np.ma.masked_where(np.sqrt(X**2 + Y**2) < r_exclude, grid_z)

# Extract real and imaginary parts
u_scn_amp = grid_z.real   # Scattered field amplitude
u_scn_phase = grid_z.imag # Scattered field phase

#%% ======================= ANALYTICAL SOLUTION =======================
"""
Compute analytical solution for a sound-hard circular obstacle
"""
R_exact = np.sqrt(X**2 + Y**2)  # Radial distance from origin
u_inc_exact, u_scn_exact, u_exact = sound_hard_circle_calc(k, r_exclude, X, Y, n_terms=None)

#%% ======================= MASKING =======================
"""
Mask exact solution inside the circular obstacle
"""
u_inc_exact = mask_displacement(R_exact, r_exclude, l_se, u_inc_exact)
u_scn_exact = mask_displacement(R_exact, r_exclude, l_se, u_scn_exact)
u_exact     = mask_displacement(R_exact, r_exclude, l_se, u_exact)

#%% ======================= ERROR COMPUTATION =======================
"""
Compute relative L2 error (real part of scattered field)
"""
R_grid = np.sqrt(X**2 + Y**2)
u_scn_exact_masked = np.copy(u_scn_exact)
u_scn_amp_masked   = np.copy(u_scn_amp)
u_scn_exact_masked[R_grid < r_exclude] = 0
u_scn_amp_masked[R_grid < r_exclude] = 0

relative_error = np.linalg.norm(u_scn_exact_masked.real - u_scn_amp_masked.real, 2) / \
                 np.linalg.norm(u_scn_exact_masked.real, 2)
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
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, FuncFormatter
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

def plot_bem_displacements_errors(X, Y, u_inc_amp, u_scn_amp, u_amp, u_inc_phase, u_scn_phase, u_phase, x_line, rel_error_line):
    """
    Plot the amplitude and phase of the incident, scattered, and total displacement, plus relative error along y = 0.
    """
    shrink = 0.75
    square_size = 2 * np.pi
    square_xy = (-square_size / 2, -square_size / 2)
    square_props = dict(edgecolor="gray", facecolor="none", lw=0.8)

    fig = plt.figure(figsize=(3.6, 6.0))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 0.7], hspace=0.5, wspace=0.05)
    # Update GridSpec to 4 rows: [amp, amp, phase, rel_error+zoom]
    # fig = plt.figure(figsize=(3.4, 6.5))
    # gs = gridspec.GridSpec(4, 2, height_ratios=[1, 1, 1, 1.2], hspace=0.4, wspace=0.05)

    axs = np.empty((4, 2), dtype=object)

    # Subplot 1: Amplitude of the incident wave
    axs[0, 0] = fig.add_subplot(gs[0, 0])
    c1 = axs[0, 0].pcolormesh(X, Y, u_inc_amp, cmap="RdYlBu", rasterized=True, vmin=-1.5, vmax=1.5)
    cb1 = fig.colorbar(c1, ax=axs[0, 0], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb1.set_label(r"$u_{\rm{sct}}$", fontsize=8)
    cb1.set_ticks([-1.5, 1.5])
    cb1.set_ticklabels(['-1.5', '1.5'], fontsize=8)
    axs[0, 0].add_patch(Rectangle(square_xy, square_size, square_size, **square_props))
    axs[0, 0].axis("off")
    axs[0, 0].set_aspect("equal")

    # Subplot 2: Amplitude of the total wave
    axs[0, 1] = fig.add_subplot(gs[0, 1])
    amp_ratio = np.abs(u_amp) / np.abs(u_scn_amp).max()
    c2 = axs[0, 1].pcolormesh(X, Y, amp_ratio, cmap="magma", rasterized=True)
    cb2 = fig.colorbar(c2, ax=axs[0, 1], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb2.set_label(r"|Error| / max($u$)", fontsize=8)
    cb2.set_ticks([0, np.max(amp_ratio)])
    cb2.set_ticklabels([f'{0:.1f}', f'{np.max(amp_ratio):.4f}'], fontsize=8)
    axs[0, 1].add_patch(Rectangle(square_xy, square_size, square_size, **square_props))
    axs[0, 1].axis("off")
    axs[0, 1].set_aspect("equal")
    # Add horizontal line from x = π to x = 10π at y = 0 (or center)
    y_center = 0  # or e.g., y_center = Y.mean() or another value of interest
    line1 = Line2D([np.pi, 10*np.pi], [y_center, y_center], color="#00a2ff", linewidth=1.0, linestyle='-')
    axs[0, 1].add_line(line1)

    # Subplot 3: Phase of the incident wave
    axs[1, 0] = fig.add_subplot(gs[1, 0])
    c3 = axs[1, 0].pcolormesh(X, Y, u_inc_phase, cmap="twilight_shifted", rasterized=True, vmin=-np.pi, vmax=np.pi)
    cb3 = fig.colorbar(c3, ax=axs[1, 0], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb3.set_label(r"$u_{\rm{sct}}$", fontsize=8)
    cb3.set_ticks([-np.pi, np.pi])
    cb3.set_ticklabels([r'-$\pi$', r'$\pi$'], fontsize=8)
    axs[1, 0].add_patch(Rectangle(square_xy, square_size, square_size, **square_props))


    axs[1, 0].axis("off")
    axs[1, 0].set_aspect("equal")

    # Subplot 4: Phase of the total wave
    axs[1, 1] = fig.add_subplot(gs[1, 1])
    phase_ratio = np.abs(u_phase) / np.abs(u_scn_phase).max()
    c4 = axs[1, 1].pcolormesh(X, Y, phase_ratio, cmap="magma", rasterized=True)
    cb4 = fig.colorbar(c4, ax=axs[1, 1], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb4.set_label(r"|Error| / max($u$)", fontsize=8)
    cb4.set_ticks([0, np.max(phase_ratio)])
    cb4.set_ticklabels([f'{0:.1f}', f'{np.max(phase_ratio):.4f}'], fontsize=8)
    axs[1, 1].add_patch(Rectangle(square_xy, square_size, square_size, **square_props))
    axs[1, 1].axis("off")
    axs[1, 1].set_aspect("equal")

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
    # Subplot 5: Relative error along y = 0 (full width)
    ax_err = fig.add_subplot(gs[2, :])
    ax_err.plot(x_line, rel_error_line, label='Relative error', color="#00a2ff", linewidth=1.0)
    #ax_err.axvline(x=np.pi, color="#acacac", linestyle='-', linewidth=1)
    #ax_err.axvline(x=2*np.pi, color="#acacac", linestyle='-', linewidth=1)  
    # Sombrear la región entre π y 2π
    ax_err.axvspan(np.pi, 2*np.pi, color='gray', alpha=0.2)  # alpha controla la transparencia

    ax_err.set_xlabel(r'$x$')
    ax_err.set_ylabel(r"$|$Error$|$ / max($u$)")
    ax_err.set_ylim(0, 0.6)
    # zm = ax_err.inset_axes([0.2, 0.5, 0.75, 0.45])
    # zm.plot(x_line, rel_error_line, color="#00a2ff")
    # zm.set_yticks([0, 0.03])
    # zm.set_yticklabels([0, 0.03], fontsize=7)

    # Zoomed inset
    zm = ax_err.inset_axes([0.0, -1.1, 1.0, 0.60])
    zm.plot(x_line, rel_error_line, color="#00a2ff", linewidth=1.0)
    zm.set_xlim(np.pi, 10*np.pi)
    zm.set_xlabel(r'$x$', fontsize=6)
    zm.set_ylabel(r"$|$Error$|$ / max($u$)", fontsize=6)
    zm.set_ylim(0, 0.04)   # adjust zoom y-limits
    #zm.axvline(x=np.pi, color="#acacac", linestyle='-', linewidth=1)
    #zm.axvline(x=2*np.pi, color="#acacac", linestyle='-', linewidth=1)   
    zm.axvspan(np.pi, 2*np.pi, color='gray', alpha=0.2)  # alpha controla la transparencia

    zm.xaxis.set_major_locator(MultipleLocator(2*np.pi))
    zm.xaxis.set_major_locator(MultipleLocator(base=np.pi))
    zm.xaxis.set_major_formatter(FuncFormatter(format_func))
    zm.set_yticks([0, 0.04])
    zm.set_yticklabels([0, 0.04], fontsize=6)
    zm.tick_params(axis="x", labelsize=6)

    # Connect inset to main plot
    # mark_inset(ax_err, zm, loc1=2, loc2=4, fc="none", ec="0.5")
    #zm = inset_axes(ax_err, [0.24, 0.5, 0.75, 0.45])
    #zm.plot(x_line, rel_error_line, color="#00a2ff", linewidth=1.0)
    # configure limits, ticks, etc. here
    mark_inset(ax_err, zm, loc1=1, loc2=2, fc="none", ec="0.5", lw=1)


    #ax_err.set_ylim(0, 10)#
    #ax_err.set_yticks([0, np.max(amp_ratio)])
    #ax_err.set_yticklabels([f'{0:.1f}', f'{np.max(amp_ratio):.4f}'], fontsize=8)
    ax_err.xaxis.set_major_locator(MultipleLocator(base=np.pi))
    ax_err.xaxis.set_major_formatter(FuncFormatter(format_func))
    ax_err.set_title('BEM - Amplitude', fontsize=8)
    #ax_err.grid(True)

    # Add rotated labels on the left
    fig.text(0.08, 0.79, r'BEM - Amplitude', fontsize=8, va='center', ha='center', rotation='vertical')
    fig.text(0.08, 0.48, r'BEM - Phase', fontsize=8, va='center', ha='center', rotation='vertical')

    plt.tight_layout()
    plt.savefig("figures/generalization_bem.svg", dpi=300, bbox_inches='tight')
    #plt.show()



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

# Define log filename inside the logs folder (with date)
log_filename = os.path.join(output_folder, f"{script_name}_log_{date_str}.txt")

# Write log file
with open(log_filename, "w") as f:
    f.write(log_text)

print(f"Log saved to: {log_filename}")