# utf-8

"""
Scattering of a plane wave by a rigid cylinder using the Mie series expansion.

This script calculates the displacement field for the scattering of a plane wave by a rigid cylinder
using the Mie series expansion. The displacement field is calculated as the sum of the incident and
scattered waves. The incident wave is a plane wave impinging on the cylinder, and the scattered wave
is the wave scattered by the cylinder. The displacement field is calculated in polar coordinates
(r, theta) and plotted in polar coordinates.

"""
import subprocess
import timeit
from scipy.special import hankel1,hankel2,jv,jn
import numpy as np
from numpy import pi, exp, cos, zeros_like, ma, real, round, min, max, std, mean
import matplotlib.pyplot as plt
#import gmsh
#import meshio
import matplotlib as mpl
from scipy.interpolate import griddata
import matplotlib.patches as patches
from matplotlib.patches import Rectangle


# Configuración de LaTeX para matplotlib
pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "xelatex",        # change this if using xetex or lautex
    "text.usetex": False,                # use LaTeX to write all text
    "font.family": "sans-serif",
    # "font.serif": [],
    "font.sans-serif": ["DejaVu Sans"], # specify the sans-serif font
    "font.monospace": [],
    "axes.labelsize": 8,               # LaTeX default is 10pt font.
    "font.size": 0,
    "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    # "figure.figsize": (3.15, 2.17),     # default fig size of 0.9 textwidth
    "pgf.preamble": r'\usepackage{amsmath},\usepackage{amsthm},\usepackage{amssymb},\usepackage{mathspec},\renewcommand{\familydefault}{\sfdefault},\usepackage[italic]{mathastext}'
    }
mpl.rcParams.update(pgf_with_latex)

# Function to compute the exact solution
def sound_hard_circle_calc(k0, a, X, Y, n_terms=None):
    """
    Calculate the scattered and total sound field for a sound-hard circular obstacle.

    Parameters:
    -----------
    k0 : float
        Wave number of the incident wave.
    a : float
        Radius of the circular obstacle.
    X : ndarray
        X-coordinates of the grid points where the field is calculated.
    Y : ndarray
        Y-coordinates of the grid points where the field is calculated.
    n_terms : int, optional
        Number of terms in the series expansion. If None, it is calculated based on k0 and a.

    Returns:
    --------
    u_sc : ndarray
        Scattered sound field at the grid points.
    u : ndarray
        Total sound field (incident + scattered) at the grid points.
    """
    points = np.column_stack((X.ravel(), Y.ravel()))
    fem_xx = points[:, 0:1]
    fem_xy = points[:, 1:2]
    r = np.sqrt(fem_xx * fem_xx + fem_xy * fem_xy)
    theta = np.arctan2(fem_xy, fem_xx)
    npts = np.size(fem_xx, 0)
    if n_terms is None:
        n_terms = int(30 + (k0 * a)**1.01)
    u_scn = np.zeros((npts), dtype=np.complex128)
    for n in range(-n_terms, n_terms):
        bessel_deriv = jv(n-1, k0*a) - n/(k0*a) * jv(n, k0*a)
        hankel_deriv = n/(k0*a)*hankel1(n, k0*a) - hankel1(n+1, k0*a)
        u_scn += (-(1j)**(n) * (bessel_deriv/hankel_deriv) * hankel1(n, k0*r) * \
            np.exp(1j*n*theta)).ravel()
    u_scn = np.reshape(u_scn, X.shape)
    u_inc = np.exp(1j*k0*X)
    u = u_inc + u_scn
    return u_inc, u_scn, u


def mask_displacement(R_exact, r_i, r_e, u):
    """
    Mask the displacement outside the scatterer.

    Parameters:
    R_exact (numpy.ndarray): Radial coordinates.
    r_i (float): Inner radius.
    r_e (float): Outer radius.
    u_amp_exact (numpy.ma.core.MaskedArray): Exact displacement amplitude.
    u_scn_amp_exact (numpy.ma.core.MaskedArray): Exact scattered displacement amplitude.

    Returns:
    u_amp_exact (numpy.ma.core.MaskedArray): Masked exact displacement amplitude.
    u_scn_amp_exact (numpy.ma.core.MaskedArray): Masked exact scattered displacement amplitude.
    """
    u = np.ma.masked_where(R_exact < r_i, u)
    #u_scn_amp_exact = np.ma.masked_where(R_exact > r_e, u_scn_amp_exact)
    return u

def plot_exact_displacement(X, Y, u_inc_amp, u_scn_amp, u_amp, u_inc_phase, u_scn_phase, u_phase):
    """
    Plot the amplitude and phase of the incident, scattered, and total displacement.

    Parameters:
    X (numpy.ndarray): X-coordinates of the grid.
    Y (numpy.ndarray): Y-coordinates of the grid.
    u_inc (numpy.ndarray): Incident displacement field.
    u_scn (numpy.ndarray): Scattered displacement field.
    u (numpy.ndarray): Total displacement field.
    """

    fig, axs = plt.subplots(2, 3, figsize=(6.5, 3.5))
    decimales = 1e+4  # Number of decimals for the color bar
    shrink = 0.5  # Shrink factor for the color bar

    # Amplitude of the incident wave
    c1 = axs[0, 0].pcolormesh(X, Y, u_inc_amp, cmap="RdYlBu", rasterized=True, vmin=-1.5, vmax=1.5)
    cb1 = fig.colorbar(c1, ax=axs[0, 0], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb1.set_label(r"$u_{\rm{inc}}$")
    cb1.set_ticks([-1.5, 1.5])
    cb1.set_ticklabels([f'{-1.5}', f'{1.5}'], fontsize=7)
    axs[0, 0].axis("off")
    axs[0, 0].set_aspect("equal")

    # Amplitude of the scattered wave
    c2 = axs[0, 1].pcolormesh(X, Y, u_scn_amp, cmap="RdYlBu", rasterized=True, vmin=-1.5, vmax=1.5)
    cb2 = fig.colorbar(c2, ax=axs[0, 1], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb2.set_label(r"$u_{\rm{sct}}$")
    cb2.set_ticks([-1.5, 1.5])
    cb2.set_ticklabels([f'{-1.5}', f'{1.5}'], fontsize=7)
    axs[0, 1].axis("off")
    axs[0, 1].set_aspect("equal")

    # Amplitude of the total wave
    c3 = axs[0, 2].pcolormesh(X, Y, u_amp, cmap="RdYlBu", rasterized=True, vmin=-1.5, vmax=1.5)
    cb3 = fig.colorbar(c3, ax=axs[0, 2], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb3.set_label(r"$u$")
    cb3.set_ticks([-1.5, 1.5])
    cb3.set_ticklabels([f'{-1.5}', f'{1.5}'], fontsize=7)
    axs[0, 2].axis("off")
    axs[0, 2].set_aspect("equal")

    # Phase of the incident wave
    c4 = axs[1, 0].pcolormesh(X, Y, u_inc_phase, cmap="twilight_shifted", rasterized=True, vmin=-(np.pi), vmax=(np.pi))
    cb4 = fig.colorbar(c4, ax=axs[1, 0], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb4.set_label(r"$u_{\rm{inc}}$")
    cb4.set_ticks([-(np.pi),(np.pi)])
    cb4.set_ticklabels([r'-$\pi$', r'$\pi$'], fontsize=7)
    axs[1, 0].axis("off")
    axs[1, 0].set_aspect("equal")

    # Phase of the scattered wave
    c5 = axs[1, 1].pcolormesh(X, Y, u_scn_phase, cmap="twilight_shifted", rasterized=True, vmin=-(np.pi), vmax=(np.pi))
    cb5 = fig.colorbar(c5, ax=axs[1, 1], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb5.set_label(r"$u_{\rm{sct}}$")
    cb5.set_ticks([-(np.pi),(np.pi)])
    cb5.set_ticklabels([r'-$\pi$', r'$\pi$'], fontsize=7)
    axs[1, 1].axis("off")
    axs[1, 1].set_aspect("equal")

    # Phase of the total wave
    c6 = axs[1, 2].pcolormesh(X, Y, u_phase, cmap="twilight_shifted", rasterized=True, vmin=-(np.pi), vmax=(np.pi))
    cb6 = fig.colorbar(c6, ax=axs[1, 2], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb6.set_label(r"$u$")
    cb6.set_ticks([-(np.pi),(np.pi)])
    cb6.set_ticklabels([r'-$\pi$', r'$\pi$'], fontsize=7)
    axs[1, 2].axis("off")
    axs[1, 2].set_aspect("equal")

    # Add rotated labels "Amplitude" and "Phase"
    fig.text(0.05, 0.80, r'Exact - Amplitude', fontsize=8, fontweight='regular', va='center', ha='center', rotation='vertical')
    fig.text(0.05, 0.30, r'Exact - Phase', fontsize=8, fontweight='regular', va='center', ha='center', rotation='vertical')

    # Adjust space between rows (increase 'hspace' for more space between rows)
    plt.subplots_adjust(hspace=1.1)  # You can tweak this value (e.g., 0.5, 0.6) as needed

    # Tight layout
    plt.tight_layout()

    # Save the figure
    plt.savefig("figs/displacement_exact.pdf", dpi=300, bbox_inches='tight')

 
def plot_fem_displacements(X, Y, u_inc_amp, u_scn_amp, u_amp, u_inc_phase, u_scn_phase, u_phase):
    """
    Plot the amplitude and phase of the incident, scattered, and total displacement.

    Parameters:
    X (numpy.ndarray): X-coordinates of the grid.
    Y (numpy.ndarray): Y-coordinates of the grid.
    u_inc (numpy.ndarray): Incident displacement field.
    u_scn (numpy.ndarray): Scattered displacement field.
    u (numpy.ndarray): Total displacement field.
    """

    fig, axs = plt.subplots(2, 3, figsize=(6.5, 3.5))
    decimales = 1e+4  # Number of decimals for the color bar
    shrink = 0.5  # Shrink factor for the color bar

    # Amplitude of the incident wave
    c1 = axs[0, 0].pcolormesh(X, Y, u_inc_amp, cmap="RdYlBu", rasterized=True, vmin=-1.5, vmax=1.5)
    cb1 = fig.colorbar(c1, ax=axs[0, 0], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb1.set_label(r"$u_{\rm{sct}}$")
    cb1.set_ticks([-1.5, 1.5])
    cb1.set_ticklabels([f'{-1.5}', f'{1.5}'], fontsize=7)
    axs[0, 0].axis("off")
    axs[0, 0].set_aspect("equal")

    # Amplitude of the scattered wave
    c2 = axs[0, 1].pcolormesh(X, Y, u_scn_amp, cmap="RdYlBu", rasterized=True, vmin=-1.5, vmax=1.5)
    cb2 = fig.colorbar(c2, ax=axs[0, 1], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb2.set_label(r"$u$")
    cb2.set_ticks([-1.5, 1.5])
    cb2.set_ticklabels([f'{-1.5}', f'{1.5}'], fontsize=7)
    axs[0, 1].axis("off")
    axs[0, 1].set_aspect("equal")

    # Amplitude of the total wave
    c3 = axs[0, 2].pcolormesh(X, Y, np.abs(u_amp)/np.abs(u_scn_amp).max(), cmap="magma", rasterized=True)
    cb3 = fig.colorbar(c3, ax=axs[0, 2], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb3.set_label(r"|Error| / max($u$)")
    cb3.set_ticks([0, np.max(np.abs(u_amp)/np.abs(u_scn_amp).max())])
    cb3.set_ticklabels([f'{0:.1f}', f'{np.max(np.abs(u_amp)/np.abs(u_scn_amp).max()):.4f}'], fontsize=7)
    axs[0, 2].axis("off")
    axs[0, 2].set_aspect("equal")

    # Phase of the incident wave
    c4 = axs[1, 0].pcolormesh(X, Y, u_inc_phase, cmap="twilight_shifted", rasterized=True, vmin=-(np.pi), vmax=(np.pi))
    cb4 = fig.colorbar(c4, ax=axs[1, 0], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb4.set_label(r"$u_{\rm{sct}}$")
    cb4.set_ticks([-(np.pi),(np.pi)])
    cb4.set_ticklabels([r'-$\pi$', r'$\pi$'], fontsize=7)
    axs[1, 0].axis("off")
    axs[1, 0].set_aspect("equal")

    # Phase of the scattered wave
    c5 = axs[1, 1].pcolormesh(X, Y, u_scn_phase, cmap="twilight_shifted", rasterized=True, vmin=-np.pi, vmax=(np.pi))
    cb5 = fig.colorbar(c5, ax=axs[1, 1], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb5.set_label(r"$u$")
    cb5.set_ticks([-(np.pi),(np.pi)])
    cb5.set_ticklabels([r'-$\pi$', r'$\pi$'], fontsize=7)
    axs[1, 1].axis("off")
    axs[1, 1].set_aspect("equal")

    # Phase of the total wave
    c6 = axs[1, 2].pcolormesh(X, Y, np.abs(u_phase)/np.abs(u_scn_phase).max(), cmap="magma", rasterized=True)
    cb6 = fig.colorbar(c6, ax=axs[1, 2], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb6.set_label(r"|Error| / max($u$)")
    cb6.set_ticks([0, np.max(np.abs(u_phase)/np.abs(u_scn_phase).max())])
    cb6.set_ticklabels([f'{0:.1f}', f'{np.max(np.abs(u_phase)/np.abs(u_scn_phase).max()):.4f}'], fontsize=7)
    axs[1, 2].axis("off")
    axs[1, 2].set_aspect("equal")

    # Add rotated labels "Amplitude" and "Phase"
    fig.text(0.05, 0.80, r'FEM - Amplitude', fontsize=8, fontweight='regular', va='center', ha='center', rotation='vertical')
    fig.text(0.05, 0.30, r'FEM - Phase', fontsize=8, fontweight='regular', va='center', ha='center', rotation='vertical')

    # Adjust space between rows (increase 'hspace' for more space between rows)
    plt.subplots_adjust(hspace=1.1)  # You can tweak this value (e.g., 0.5, 0.6) as needed

    # Tight layout
    plt.tight_layout()

    # Save the figure
    plt.savefig("figs/displacement_fem.svg", dpi=300, bbox_inches='tight')


 
def plot_pinns_displacements(X, Y, u_inc_amp, u_scn_amp, u_amp, u_inc_phase, u_scn_phase, u_phase):
    """
    Plot the amplitude and phase of the incident, scattered, and total displacement.

    Parameters:
    X (numpy.ndarray): X-coordinates of the grid.
    Y (numpy.ndarray): Y-coordinates of the grid.
    u_inc (numpy.ndarray): Incident displacement field.
    u_scn (numpy.ndarray): Scattered displacement field.
    u (numpy.ndarray): Total displacement field.
    """
 

    # Coordenadas de la esquina inferior izquierda del cuadrado (ajusta según sea necesario)
 
    # Definiciones generales
    square_size = 2*np.pi
    square_xy = (-square_size/2, -square_size/2)  # centrado en el origen
    square_props = dict(edgecolor="gray", facecolor="none", lw=0.8)


    fig, axs = plt.subplots(2, 3, figsize=(6.5, 3.5))
    decimales = 1e+4  # Number of decimals for the color bar
    shrink = 0.5  # Shrink factor for the color bar

    # Amplitude of the incident wave
    c1 = axs[0, 0].pcolormesh(X, Y, u_inc_amp, cmap="RdYlBu", rasterized=True, vmin=-1.5, vmax=1.5)
    cb1 = fig.colorbar(c1, ax=axs[0, 0], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb1.set_label(r"$u_{\rm{sct}}$")
    cb1.set_ticks([-1.5, 1.5])
    cb1.set_ticklabels([f'{-1.5}', f'{1.5}'], fontsize=7)
    axs[0, 0].add_patch(Rectangle(square_xy, square_size, square_size, **square_props))   
    axs[0, 0].axis("off")
    axs[0, 0].set_aspect("equal")

    # Amplitude of the scattered wave
    c2 = axs[0, 1].pcolormesh(X, Y, u_scn_amp, cmap="RdYlBu", rasterized=True, vmin=-1.5, vmax=1.5)
    cb2 = fig.colorbar(c2, ax=axs[0, 1], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb2.set_label(r"$u$")
    cb2.set_ticks([-1.5, 1.5])
    cb2.set_ticklabels([f'{-1.5}', f'{1.5}'], fontsize=7)
    axs[0, 1].add_patch(Rectangle(square_xy, square_size, square_size, **square_props))   
    axs[0, 1].axis("off")
    axs[0, 1].set_aspect("equal")

    # Amplitude of the total wave
    c3 = axs[0, 2].pcolormesh(X, Y, np.abs(u_amp)/np.abs(u_scn_amp).max(), cmap="magma", rasterized=True)
    cb3 = fig.colorbar(c3, ax=axs[0, 2], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb3.set_label(r"|Error| / max($u$)")
    cb3.set_ticks([0, np.max(np.abs(u_amp)/np.abs(u_scn_amp).max())])
    cb3.set_ticklabels([f'{0:.1f}', f'{np.max(np.abs(u_amp)/np.abs(u_scn_amp).max()):.4f}'], fontsize=7)
    axs[0, 2].add_patch(Rectangle(square_xy, square_size, square_size, **square_props))
    axs[0, 2].axis("off")
    axs[0, 2].set_aspect("equal")

    # Phase of the incident wave
    c4 = axs[1, 0].pcolormesh(X, Y, u_inc_phase, cmap="twilight_shifted", rasterized=True, vmin=-(np.pi), vmax=(np.pi))
    cb4 = fig.colorbar(c4, ax=axs[1, 0], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb4.set_label(r"$u_{\rm{sct}}$")
    cb4.set_ticks([-(np.pi),(np.pi)])
    cb4.set_ticklabels([r'-$\pi$', r'$\pi$'], fontsize=7)
    axs[1, 0].add_patch(Rectangle(square_xy, square_size, square_size, **square_props))
    axs[1, 0].axis("off")
    axs[1, 0].set_aspect("equal")

    # Phase of the scattered wave
    c5 = axs[1, 1].pcolormesh(X, Y, u_scn_phase, cmap="twilight_shifted", rasterized=True, vmin=-(np.pi), vmax=(np.pi))
    cb5 = fig.colorbar(c5, ax=axs[1, 1], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb5.set_label(r"$u$")
    cb5.set_ticks([-(np.pi),(np.pi)])
    cb5.set_ticklabels([r'-$\pi$', r'$\pi$'], fontsize=7)
    axs[1, 1].add_patch(Rectangle(square_xy, square_size, square_size, **square_props))
    axs[1, 1].axis("off")
    axs[1, 1].set_aspect("equal")

    # Phase of the total wave
    c6 = axs[1, 2].pcolormesh(X, Y, np.abs(u_phase)/np.abs(u_scn_phase).max(), cmap="magma", rasterized=True)
    cb6 = fig.colorbar(c6, ax=axs[1, 2], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb6.set_label(r"|Error| / max($u$)")
    cb6.set_ticks([0, np.max(np.abs(u_phase))/(np.abs(u_scn_phase).max())])
    cb6.set_ticklabels([f'{0:.1f}', f'{np.max(np.abs(u_phase)/np.abs(u_scn_phase).max()):.4f}'], fontsize=7)
    axs[1, 2].add_patch(Rectangle(square_xy, square_size, square_size, **square_props))
    axs[1, 2].axis("off")
    axs[1, 2].set_aspect("equal")

    # Add rotated labels "Amplitude" and "Phase"
    fig.text(0.05, 0.80, r'PINNs - Amplitude', fontsize=8, fontweight='regular', va='center', ha='center', rotation='vertical')
    fig.text(0.05, 0.30, r'PINNs - Phase', fontsize=8, fontweight='regular', va='center', ha='center', rotation='vertical')

    # Adjust space between rows (increase 'hspace' for more space between rows)
    plt.subplots_adjust(hspace=1.1)  # You can tweak this value (e.g., 0.5, 0.6) as needed

    # Tight layout
    plt.tight_layout()

    # Save the figure
    plt.savefig("figs/displacement_pinns.svg", dpi=150, bbox_inches='tight')


def plot_mesh_from_file(file_path_msh):
    """
    Reads a mesh file using meshio, extracts the points and cells, and plots the mesh.
    Also calculates the number of connections and nodes in the physical domain 
    and absorbing layer, and returns these values.

    Parameters:
    file_path_msh (str): Path to the mesh file.

    Returns:
    tuple: (num_nodes, num_connections_P, num_connections_A)
        - num_nodes: The total number of unique nodes in the mesh.
        - num_connections_P: The total number of unique connections in the physical domain.
        - num_connections_A: The total number of unique connections in the absorbing layer.
    """
    
    # Read the mesh using meshio
    mesh = meshio.read(file_path_msh)
    points = mesh.points 
    cells = mesh.cells

    # Extract the triangles from the mesh
    triangles_P = cells[10].data  # Connections for physical domain
    triangles_A = cells[9].data   # Connections for absorbing layer

    # Calculate the number of nodes (unique points)
    num_nodes = len(points)

    # Calculate the number of connections (unique edges)
    def count_unique_edges(triangles):
        edges = set()  # To store unique edges as tuples (sorted to avoid duplicates)
        for triangle in triangles:
            # Create all 3 edges of the triangle and sort each edge
            edges.add(tuple(sorted([triangle[0], triangle[1]])))
            edges.add(tuple(sorted([triangle[1], triangle[2]])))
            edges.add(tuple(sorted([triangle[2], triangle[0]])))
        return len(edges)

    # Count the number of unique edges in physical domain and absorbing layer
    num_connections_P = count_unique_edges(triangles_P)
    num_connections_A = count_unique_edges(triangles_A)

    # Print the results (optional)
    # print(f"Number of nodes: {num_nodes}")
    # print(f"Number of connections in physical domain: {num_connections_P}")
    # print(f"Number of connections in absorbing layer: {num_connections_A}")

    # Plot the mesh
    plt.figure(figsize=(2.0, 2.0))
    plt.triplot(points[:, 0], points[:, 1], triangles_P, color='#cbcbcbff', lw=0.3)
    plt.triplot(points[:, 0], points[:, 1], triangles_A, color='#989898ff', lw=0.3)

    # Surface connections (edges for physical and absorbing layers)
    connections_ri = np.concatenate([cells[i].data for i in range(0, 4)])   
    connections_re = np.concatenate([cells[i].data for i in range(4, 8)])   
    start_points_ri = points[connections_ri[:, 0]]
    end_points_ri = points[connections_ri[:, 1]]
    start_points_re = points[connections_re[:, 0]]
    end_points_re = points[connections_re[:, 1]]

    # Plot connections (edges)
    plt.plot(np.vstack([start_points_ri[:, 0], end_points_ri[:, 0]]), 
             np.vstack([start_points_ri[:, 1], end_points_ri[:, 1]]), color='#000146ff', lw=0.3)
    plt.plot(np.vstack([start_points_re[:, 0], end_points_re[:, 0]]), 
             np.vstack([start_points_re[:, 1], end_points_re[:, 1]]), color='#003e23ff', lw=0.3)

    # Final plot adjustments
    plt.axis('off')
    plt.savefig("figs/mesh.pdf", dpi=300)
    plt.show()

    return None

def process_onelab_data(file_path):
    """
    Process GMSH data from a given file and extract node coordinates and displacement values.

    Parameters:
    file_path (str): The path to the GMSH file to be processed.

    Returns:
    tuple: A tuple containing three numpy arrays:
        - X (numpy.ndarray): The x-coordinates of the nodes.
        - Y (numpy.ndarray): The y-coordinates of the nodes.
        - u_amp (numpy.ndarray): The displacement values at the nodes.
    """
    # Verificar si GMSH ya está inicializado, si no lo está, inicializarlo
    gmsh.initialize()

    # Abrir el archivo de GMSH
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.open(file_path)

    # Extraer nodos y elementos
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    element_types, element_tags, element_node_tags = gmsh.model.mesh.getElements()

    # Reshape de los elementos
    elements = np.array(element_node_tags[0]).reshape(-1, 3)

    # Reshape de las coordenadas de los nodos
    node_coords = node_coords.reshape(-1, 3)

    # Obtener los datos del campo
    try:
        field = gmsh.view.getHomogeneousModelData(0, 0) # (0, 1) to get phase data - (0, 0) to get amplitude data
    except:
        field = gmsh.view.getHomogeneousModelData(1, 0) # (1, 1) to get phase data - (0, 0) to get amplitude data
    field_id = field[1]
    field_data = np.array(field[2]).reshape(-1, 3)

    # Crear el diccionario
    node_u_data_dict = {}
    for elem, u_vals in zip(elements, field_data):
        for node, u_val in zip(elem, u_vals):
            node_u_data_dict[node] = u_val

    # Crear el diccionario node_tags -> node_coords
    node_coords_dict = {}
    for tag, coord in zip(node_tags, node_coords):
        node_coords_dict[tag] = coord

    # Añadir los valores de desplazamiento a las coordenadas de los nodos
    for tag, coord in zip(node_tags, node_coords):
        node_coords_dict[tag][2] = node_u_data_dict[tag]

    # Crear un nuevo arreglo de nodos con los valores de desplazamiento
    pts = np.array([coord for coord in node_coords_dict.values()])

    # Create an array to store the reordered pts
    reordered_pts = np.zeros((int(np.max(node_tags)), 3))

    # Reorder pts based on the indices of node_tags
    for i, tag in enumerate(node_tags-1):
        reordered_pts[int(tag)] = pts[i]

    X = reordered_pts[:, 0]
    Y = reordered_pts[:, 1]
    u_sc_amp = reordered_pts[:, 2]

    # Obtener los datos del campo
    try:
        field = gmsh.view.getHomogeneousModelData(0, 1) # (0, 1) to get phase data - (0, 0) to get amplitude data
    except:
        field = gmsh.view.getHomogeneousModelData(1, 1) # (1, 1) to get phase data - (0, 0) to get amplitude data
    field_id = field[1]
    field_data = np.array(field[2]).reshape(-1, 3)

    # Cerrar GMSH
    gmsh.finalize()

    # Crear el diccionario
    node_u_data_dict = {}
    for elem, u_vals in zip(elements, field_data):
        for node, u_val in zip(elem, u_vals):
            node_u_data_dict[node] = u_val

    # Crear el diccionario node_tags -> node_coords
    node_coords_dict = {}
    for tag, coord in zip(node_tags, node_coords):
        node_coords_dict[tag] = coord

    # Añadir los valores de desplazamiento a las coordenadas de los nodos
    for tag, coord in zip(node_tags, node_coords):
        node_coords_dict[tag][2] = node_u_data_dict[tag]

    # Crear un nuevo arreglo de nodos con los valores de desplazamiento
    pts = np.array([coord for coord in node_coords_dict.values()])

    # Create an array to store the reordered pts
    reordered_pts = np.zeros((int(np.max(node_tags)), 3))

    # Reorder pts based on the indices of node_tags
    for i, tag in enumerate(node_tags-1):
        reordered_pts[int(tag)] = pts[i]

 
    u_sc_phase = reordered_pts[:, 2]

    return X, Y, elements-1, u_sc_amp, u_sc_phase

def plot_fem_results(X_fem, Y_fem, elements_fem, uscn_amp_fem, uscn_phase_fem):
    """
    Plots the FEM results for the scattered and total displacement amplitudes.

    Parameters:
    X_fem (numpy.ndarray): X coordinates of the FEM mesh.
    Y_fem (numpy.ndarray): Y coordinates of the FEM mesh.
    elements_fem (numpy.ndarray): Elements of the FEM mesh.
    uscn_amp_fem (numpy.ndarray): Scattered displacement amplitude from FEM.
    uscn_phase_fem (numpy.ndarray): Scattered displacement phase from FEM.
    """
    decimales = 1e+4  # Number of decimals for the color bar
    shrink = 0.7  # Shrink factor for the color bar

    fig, axs = plt.subplots(1, 2, figsize=(7, 3))

    # Plot the first result
    c1 = axs[0].tricontourf(X_fem, Y_fem, elements_fem, uscn_amp_fem, cmap="RdYlBu", levels=200,
                            vmin=(np.trunc(np.min(uscn_amp_fem) * decimales) / decimales), 
                            vmax=(np.trunc(np.max(uscn_amp_fem) * decimales) / decimales))
    axs[0].axis('off')
    cbar1 = plt.colorbar(c1, ax=axs[0], shrink=shrink, orientation="horizontal", pad=0.07)
    cbar1.set_ticks([(np.trunc(np.min(uscn_amp_fem) * decimales) / decimales), 
                     (np.trunc(np.max(uscn_amp_fem) * decimales) / decimales)])
    cbar1.set_ticklabels([f'{(np.trunc(np.min(uscn_amp_fem) * decimales) / decimales):.4f}', 
                          f'{(np.trunc(np.max(uscn_amp_fem) * decimales) / decimales):.4f}'])
    cbar1.set_label(r"Amplitude")
    axs[0].set_aspect('equal', adjustable='box')

    # Plot the second result
    c2 = axs[1].tricontourf(X_fem, Y_fem, elements_fem, uscn_phase_fem, cmap="RdYlBu", levels=200,
                            vmin=(np.trunc(np.min(uscn_phase_fem) * decimales) / decimales), 
                            vmax=(np.trunc(np.max(uscn_phase_fem) * decimales) / decimales))
    axs[1].axis('off')
    cbar2 = plt.colorbar(c2, ax=axs[1], shrink=shrink, orientation="horizontal", pad=0.07)
    cbar2.set_ticks([(np.trunc(np.min(uscn_phase_fem) * decimales) / decimales), 
                     (np.trunc(np.max(uscn_phase_fem) * decimales) / decimales)])
    cbar2.set_ticklabels([f'{(np.trunc(np.min(uscn_phase_fem) * decimales) / decimales):.4f}', 
                          f'{(np.trunc(np.max(uscn_phase_fem) * decimales) / decimales):.4f}'])
    cbar2.set_label(r"Phase")
    axs[1].set_aspect('equal', adjustable='box')

    plt.show()

def interpolate_fem_data(X_fem, Y_fem, uscn_amp_fem, uscn_phase_fem, r_i, r_e, n_grid):
    """
    Interpolates FEM data onto a regular grid and masks the displacement outside the scatterer.

    Parameters:
    X_fem (numpy.ndarray): X coordinates of FEM data points.
    Y_fem (numpy.ndarray): Y coordinates of FEM data points.
    u_amp_fem (numpy.ndarray): Amplitude of the total displacement from FEM.
    uscn_amp_fem (numpy.ndarray): Amplitude of the scattered displacement from FEM.
    r_i (float): Inner radius of the scatterer.
    r_e (float): Outer radius of the scatterer.
    n_grid (int): Number of grid points for interpolation.

    Returns:
    X_grid (numpy.ndarray): X coordinates of the regular grid.
    Y_grid (numpy.ndarray): Y coordinates of the regular grid.
    u_amp_interp_fem (numpy.ma.core.MaskedArray): Interpolated and masked total displacement amplitude.
    uscn_amp_interp_fem (numpy.ma.core.MaskedArray): Interpolated and masked scattered displacement amplitude.
    """
    # Scale the data
    #X_fem, Y_fem = X_fem*r_e, Y_fem*r_e

    # Create a regular grid where you want to compare the exact solution
    x_grid = np.linspace(min(X_fem), max(X_fem), n_grid)
    y_grid = np.linspace(min(Y_fem), max(Y_fem), n_grid)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

    # Interpolate onto the grid without NaN values
    uscn_phase_interp_fem = griddata((X_fem, Y_fem), uscn_phase_fem, (X_grid, Y_grid), method='cubic')
    uscn_amp_interp_fem = griddata((X_fem, Y_fem), uscn_amp_fem, (X_grid, Y_grid), method='cubic')

    # Create R_grid and Theta_grid for the new grid
    R_grid = np.sqrt(X_grid**2 + Y_grid**2)
    Theta_grid = np.arctan2(Y_grid, X_grid)

    # Mask the displacement outside the scatterer
    uscn_phase_interp_fem = np.ma.masked_where(R_grid < r_i, uscn_phase_interp_fem)
    #u_amp_interp_fem = np.ma.masked_where(R_grid > r_e, u_amp_interp_fem)
    uscn_amp_interp_fem = np.ma.masked_where(R_grid < r_i, uscn_amp_interp_fem)
    #uscn_amp_interp_fem = np.ma.masked_where(R_grid > r_e, uscn_amp_interp_fem)

    return X_grid, Y_grid, uscn_amp_interp_fem, uscn_phase_interp_fem

def extract_fem_displacements(file_path, r_i, l_se, k, n_grid, u_scn_exact, u_exact):

    X, Y, elements, u_sc_amp, u_sc_phase = process_onelab_data(file_path)
    X_grid, Y_grid, uscn_amp_interp_fem, uscn_phase_interp_fem = interpolate_fem_data(X, Y, u_sc_amp, u_sc_phase, r_i, l_se, n_grid)
    u_inc_amp_fem = np.real(np.exp(1j*k*X_grid))
    u_inc_phase_fem = np.imag(np.exp(1j*k*X_grid))
    R_grid = np.sqrt(X_grid**2 + Y_grid**2)
    u_inc_amp_fem = mask_displacement(R_grid, r_i, l_se, u_inc_amp_fem)
    u_inc_phase_fem = mask_displacement(R_grid, r_i, l_se, u_inc_phase_fem)
    uscn_amp_interp_fem = mask_displacement(R_grid, r_i, l_se, uscn_amp_interp_fem)
    uscn_phase_interp_fem = mask_displacement(R_grid, r_i, l_se, uscn_phase_interp_fem)
    u_amp_fem = u_inc_amp_fem + uscn_amp_interp_fem
    u_phase_fem = u_inc_phase_fem + uscn_phase_interp_fem
    diff_uscn_amp, diff_u_amp = uscn_amp_interp_fem - np.real(u_scn_exact), u_amp_fem - np.real(u_exact)
    diff_u_scn_phase, diff_u_phase = uscn_phase_interp_fem - np.imag(u_scn_exact), u_phase_fem - np.imag(u_exact)
    return uscn_amp_interp_fem, uscn_phase_interp_fem, u_amp_fem, u_phase_fem, diff_uscn_amp, diff_u_amp, diff_u_scn_phase, diff_u_phase




def calc_error(X, Y, u_scn_amp_exact, u_amp_exact, uscn_amp_interp, u_amp_interp, r_i, r_e):
    """
    Calculate the relative error between the exact and interpolated FEM solutions.

    Parameters:
    X (numpy.ndarray): X-coordinates of the grid.
    Y (numpy.ndarray): Y-coordinates of the grid.
    u_scn_amp_exact (numpy.ma.core.MaskedArray): Exact scattered displacement amplitude.
    u_amp_exact (numpy.ma.core.MaskedArray): Exact total displacement amplitude.
    uscn_amp_interp (numpy.ma.core.MaskedArray): Interpolated scattered displacement amplitude from FEM.
    u_amp_interp (numpy.ma.core.MaskedArray): Interpolated total displacement amplitude from FEM.
    R_exact (numpy.ndarray): Radial coordinates of the exact solution.
    r_i (float): Inner radius of the scatterer.
    r_e (float): Outer radius of the scatterer.

    Returns:
    tuple: Relative errors for scattered and total displacement amplitudes.
    """
    # Create R_grid for the new grid
    R_grid = np.sqrt(X**2 + Y**2)

    # Mask the displacement outside the scatterer
    u_scn_amp_exact_data = u_scn_amp_exact.data
    u_scn_amp_exact_data[(R_grid < r_i)] = 0
    u_amp_exact_data = u_amp_exact.data
    u_amp_exact_data[(R_grid < r_i)] = 0
    u_amp_interp_data = u_amp_interp.data
    u_amp_interp_data[(R_grid < r_i)] = 0
    uscn_amp_interp_data = uscn_amp_interp.data
    uscn_amp_interp_data[(R_grid < r_i)] = 0

    # Calculate the difference between the interpolated results and the exact results
    diff_uscn_amp_data, diff_u_amp_data = uscn_amp_interp_data - u_scn_amp_exact_data, u_amp_interp_data - u_amp_exact_data
    diff_uscn_amp, diff_u_amp = uscn_amp_interp - u_scn_amp_exact, u_amp_interp - u_amp_exact

    # Calculate the L2 norm of the differences
    norm_diff_uscn = np.linalg.norm(diff_uscn_amp_data, 2)
    norm_diff_u = np.linalg.norm(diff_u_amp_data, 2)

    # Calculate the L2 norm of the exact solutions
    norm_u_scn_exact = np.linalg.norm(u_scn_amp_exact_data, 2)
    norm_u_exact = np.linalg.norm(u_amp_exact_data, 2)

    # Calculate the relative errors using L2 norms
    rel_error_uscn = norm_diff_uscn / norm_u_scn_exact
    rel_error_u = norm_diff_u / norm_u_exact

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    # Plot u_scn_amp
    order = 1e+4
    c1 = axs[0].pcolormesh(X, Y, diff_uscn_amp, cmap="RdYlBu", vmin=(np.trunc(np.min(diff_uscn_amp) *order) / order), vmax=(np.trunc(np.max(diff_uscn_amp) * order) /order))
    cb1 = fig.colorbar(c1, ax=axs[0], shrink=0.7, orientation="horizontal", pad=0.07)
    cb1.set_label(r"Error $u_{\rm{sct}}$ Amplitude")
    cb1.set_ticks([(np.trunc(np.min(diff_uscn_amp) *order) / order), (np.trunc(np.max(diff_uscn_amp) * order) /order)])
    cb1.set_ticklabels([f'{(np.trunc(np.min(diff_uscn_amp) * order) / order)}', f'{(np.trunc(np.max(diff_uscn_amp) * order) / order)}']) 
    axs[0].axis("off")
    axs[0].set_aspect("equal")

    # Plot u_amp
    
    c2 = axs[1].pcolormesh(X, Y, diff_u_amp, cmap="RdYlBu", vmin=(np.trunc(np.min(diff_u_amp) * order) / order), vmax=(np.trunc(np.max(diff_u_amp) * order) / order))
    cb2 = fig.colorbar(c2, ax=axs[1], shrink=0.7, orientation="horizontal", pad=0.07)
    cb2.set_label(r"Error $u_{\rm{sct}}$ Phase")
    cb2.set_ticks([(np.trunc(np.min(diff_u_amp) * order) / order), (np.trunc(np.max(diff_u_amp) * order) / order)])
    cb2.set_ticklabels([f'{(np.trunc(np.min(diff_u_amp) * order) / order)}', f'{(np.trunc(np.max(diff_u_amp) *order) /order)}'])
    axs[1].axis("off")
    axs[1].set_aspect("equal")

    plt.show()

    return rel_error_uscn, rel_error_u

def calculate_relative_errors(u_scn_exact, u_exact, diff_uscn_amp, diff_u_scn_phase, R_exact, r_i):
    # Mask the displacement for the inner radius
    u_scn_exact[R_exact < r_i] = 0
    u_exact[R_exact < r_i] = 0
    diff_uscn_amp[R_exact < r_i] = 0
    diff_u_scn_phase[R_exact < r_i] = 0

    # Calculate the L2 norm of the differences for u_scn
    norm_diff_uscn = np.linalg.norm(diff_uscn_amp, 2)
    norm_usc_exact = np.linalg.norm(np.real(u_scn_exact), 2)
    rel_error_uscn_amp = norm_diff_uscn / norm_usc_exact

    # Calculate the L2 norm of the differences for u
    norm_diff_u = np.linalg.norm(diff_u_scn_phase, 2)
    norm_u_exact = np.linalg.norm(np.imag(u_scn_exact), 2)
    rel_error_uscn_phase = norm_diff_u / norm_u_exact

    # Calculate the max and min differences for u_scn
    max_diff_uscn_amp = np.max(diff_uscn_amp)
    min_diff_uscn_amp = np.min(diff_uscn_amp)

    # Calculate the max and min differences for u
    max_diff_u_phase = np.max(diff_u_scn_phase)
    min_diff_u_phase = np.min(diff_u_scn_phase)

    return rel_error_uscn_amp, rel_error_uscn_phase, max_diff_uscn_amp, min_diff_uscn_amp, max_diff_u_phase, min_diff_u_phase

def measure_execution_time(getdp_path, command_args, num_runs=10):
    """
    Measures the execution time of a command run by the GetDP software.

    Parameters:
    getdp_path (str): The path to the GetDP executable.
    command_args (str): The command line arguments to pass to GetDP.
    num_runs (int, optional): The number of times to run the command for measurement. Default is 10.

    Returns:
    tuple: A tuple containing the average execution time, standard deviation of the execution time,
           minimum execution time, and maximum execution time.
    """
    def run_getdp():
        subprocess.run(f"{getdp_path} {command_args}", 
                       shell=True, 
                       stdout=subprocess.DEVNULL, 
                       stderr=subprocess.DEVNULL)

    times = timeit.repeat(run_getdp, repeat=num_runs, number=1)
    average_time = round(mean(times), 3)
    std_dev_time = round(std(times), 3)
    min_time = round(min(times), 3)
    max_time = round(max(times), 3)

    return average_time, std_dev_time, min_time, max_time 