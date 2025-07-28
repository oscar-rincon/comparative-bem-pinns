# utf-8

"""
Scattering of a plane wave by a rigid cylinder using the Mie series expansion.

This script calculates the displacement field for the scattering of a plane wave by a rigid cylinder
using the Mie series expansion. The displacement field is calculated as the sum of the incident and
scattered waves. The incident wave is a plane wave impinging on the cylinder, and the scattered wave
is the wave scattered by the cylinder. The displacement field is calculated in polar coordinates
(r, theta) and plotted in polar coordinates.

"""

from scipy.special import hankel1,jv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


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
    plt.savefig("figs/displacement_exact.svg", dpi=300, bbox_inches='tight')


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
    c5 = axs[1, 1].pcolormesh(X, Y, u_scn_phase, cmap="twilight_shifted", rasterized=True, vmin=-(np.pi), vmax=(np.pi))
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
    cb6.set_ticks([0, np.max(np.abs(u_phase))/(np.abs(u_scn_phase).max())])
    cb6.set_ticklabels([f'{0:.1f}', f'{np.max(np.abs(u_phase)/np.abs(u_scn_phase).max()):.4f}'], fontsize=7)
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