
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
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
    plt.savefig("figures/displacement_exact.svg", dpi=300, bbox_inches='tight')



def plot_bem_displacements(X, Y, u_inc_amp, u_scn_amp, u_amp, u_inc_phase, u_scn_phase, u_phase):
    """
    Plot the amplitude and phase of the incident, scattered, and total displacement.

    Parameters:
    X (numpy.ndarray): X-coordinates of the grid.
    Y (numpy.ndarray): Y-coordinates of the grid.
    u_inc (numpy.ndarray): Incident displacement field.
    u_scn (numpy.ndarray): Scattered displacement field.
    u (numpy.ndarray): Total displacement field.
    """

    # Square patch properties
    square_size = 2 * np.pi
    square_xy = (-square_size / 2, -square_size / 2)
    square_props = dict(edgecolor="gray", facecolor="none", lw=0.8)

    fig, axs = plt.subplots(2, 2, figsize=(4.5, 3.5))
    decimales = 1e+4  # Number of decimals for the color bar
    shrink = 0.4  # Shrink factor for the color bar

    # Amplitude of the incident wave
    c1 = axs[0, 0].pcolormesh(X, Y, u_inc_amp, cmap="RdYlBu", rasterized=True, vmin=-1.5, vmax=1.5)
    cb1 = fig.colorbar(c1, ax=axs[0, 0], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb1.set_label(r"$u_{\rm{sct}}$")
    cb1.set_ticks([-1.5, 1.5])
    cb1.set_ticklabels([f'{-1.5}', f'{1.5}'], fontsize=7)
    axs[0, 0].add_patch(Rectangle(square_xy, square_size, square_size, **square_props))
    axs[0, 0].axis("off")
    axs[0, 0].set_aspect("equal")

    # Amplitude of the total wave
    c3 = axs[0, 1].pcolormesh(X, Y, np.abs(u_amp)/np.abs(u_scn_amp).max(), cmap="magma", rasterized=True)
    cb3 = fig.colorbar(c3, ax=axs[0, 1], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb3.set_label(r"|Error| / max($u$)")
    cb3.set_ticks([0, np.max(np.abs(u_amp)/np.abs(u_scn_amp).max())])
    cb3.set_ticklabels([f'{0:.1f}', f'{np.max(np.abs(u_amp)/np.abs(u_scn_amp).max()):.4f}'], fontsize=7)
    axs[0, 1].add_patch(Rectangle(square_xy, square_size, square_size, **square_props))
    axs[0, 1].axis("off")
    axs[0, 1].set_aspect("equal")

    # Phase of the incident wave
    c4 = axs[1, 0].pcolormesh(X, Y, u_inc_phase, cmap="twilight_shifted", rasterized=True, vmin=-(np.pi), vmax=(np.pi))
    cb4 = fig.colorbar(c4, ax=axs[1, 0], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb4.set_label(r"$u_{\rm{sct}}$")
    cb4.set_ticks([-(np.pi),(np.pi)])
    cb4.set_ticklabels([r'-$\pi$', r'$\pi$'], fontsize=7)
    axs[1, 0].add_patch(Rectangle(square_xy, square_size, square_size, **square_props))
    axs[1, 0].axis("off")
    axs[1, 0].set_aspect("equal")

    # Phase of the total wave
    c6 = axs[1, 1].pcolormesh(X, Y, np.abs(u_phase)/np.abs(u_scn_phase).max(), cmap="magma", rasterized=True)
    cb6 = fig.colorbar(c6, ax=axs[1, 1], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb6.set_label(r"|Error| / max($u$)")
    cb6.set_ticks([0, np.max(np.abs(u_phase)/np.abs(u_scn_phase).max())])
    cb6.set_ticklabels([f'{0:.1f}', f'{np.max(np.abs(u_phase)/np.abs(u_scn_phase).max()):.4f}'], fontsize=7)
    axs[1, 1].add_patch(Rectangle(square_xy, square_size, square_size, **square_props))
    axs[1, 1].axis("off")
    axs[1, 1].set_aspect("equal")

    # Add rotated labels "Amplitude" and "Phase"
    fig.text(0.15, 0.80, r'BEM - Amplitude', fontsize=8, fontweight='regular', va='center', ha='center', rotation='vertical')
    fig.text(0.15, 0.30, r'BEM - Phase', fontsize=8, fontweight='regular', va='center', ha='center', rotation='vertical')

    # Adjust space between rows (increase 'hspace' for more space between rows)
    fig.subplots_adjust(wspace=-0.7)  # Reduce horizontal spacing, optionally adjust vertical spacing

    # Tight layout
    plt.tight_layout()

    # Save the figure
    plt.savefig("figures/generalization_bem.svg", dpi=300, bbox_inches='tight')

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

    # Square patch properties
    square_size = 2 * np.pi
    square_xy = (-square_size / 2, -square_size / 2)
    square_props = dict(edgecolor="gray", facecolor="none", lw=0.8)

    fig, axs = plt.subplots(2, 2, figsize=(4.5, 3.5))
    decimales = 1e+4  # Number of decimals for the color bar
    shrink = 0.4  # Shrink factor for the color bar

    # Amplitude of the incident wave
    c1 = axs[0, 0].pcolormesh(X, Y, u_inc_amp, cmap="RdYlBu", rasterized=True, vmin=-1.5, vmax=1.5)
    cb1 = fig.colorbar(c1, ax=axs[0, 0], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb1.set_label(r"$u_{\rm{sct}}$")
    cb1.set_ticks([-1.5, 1.5])
    cb1.set_ticklabels([f'{-1.5}', f'{1.5}'], fontsize=7)
    axs[0, 0].add_patch(Rectangle(square_xy, square_size, square_size, **square_props))
    axs[0, 0].axis("off")
    axs[0, 0].set_aspect("equal")

    # Amplitude of the total wave
    c3 = axs[0, 1].pcolormesh(X, Y, np.abs(u_amp)/np.abs(u_scn_amp).max(), cmap="magma", rasterized=True)
    cb3 = fig.colorbar(c3, ax=axs[0, 1], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb3.set_label(r"|Error| / max($u$)")
    cb3.set_ticks([0, np.max(np.abs(u_amp)/np.abs(u_scn_amp).max())])
    cb3.set_ticklabels([f'{0:.1f}', f'{np.max(np.abs(u_amp)/np.abs(u_scn_amp).max()):.4f}'], fontsize=7)
    axs[0, 1].add_patch(Rectangle(square_xy, square_size, square_size, **square_props))
    axs[0, 1].axis("off")
    axs[0, 1].set_aspect("equal")

    # Phase of the incident wave
    c4 = axs[1, 0].pcolormesh(X, Y, u_inc_phase, cmap="twilight_shifted", rasterized=True, vmin=-(np.pi), vmax=(np.pi))
    cb4 = fig.colorbar(c4, ax=axs[1, 0], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb4.set_label(r"$u_{\rm{sct}}$")
    cb4.set_ticks([-(np.pi),(np.pi)])
    cb4.set_ticklabels([r'-$\pi$', r'$\pi$'], fontsize=7)
    axs[1, 0].add_patch(Rectangle(square_xy, square_size, square_size, **square_props))
    axs[1, 0].axis("off")
    axs[1, 0].set_aspect("equal")

    # Phase of the total wave
    c6 = axs[1, 1].pcolormesh(X, Y, np.abs(u_phase)/np.abs(u_scn_phase).max(), cmap="magma", rasterized=True)
    cb6 = fig.colorbar(c6, ax=axs[1, 1], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb6.set_label(r"|Error| / max($u$)")
    cb6.set_ticks([0, np.max(np.abs(u_phase)/np.abs(u_scn_phase).max())])
    cb6.set_ticklabels([f'{0:.1f}', f'{np.max(np.abs(u_phase)/np.abs(u_scn_phase).max()):.4f}'], fontsize=7)
    axs[1, 1].add_patch(Rectangle(square_xy, square_size, square_size, **square_props))
    axs[1, 1].axis("off")
    axs[1, 1].set_aspect("equal")

    # Add rotated labels "Amplitude" and "Phase"
    fig.text(0.15, 0.80, r'PINNs - Amplitude', fontsize=8, fontweight='regular', va='center', ha='center', rotation='vertical')
    fig.text(0.15, 0.30, r'PINNs - Phase', fontsize=8, fontweight='regular', va='center', ha='center', rotation='vertical')

    # Adjust space between rows (increase 'hspace' for more space between rows)
    fig.subplots_adjust(wspace=-0.7)  # You can tweak this value (e.g., 0.5, 0.6) as needed

    # Tight layout
    plt.tight_layout()

    # Save the figure
    plt.savefig("figures/generalization_pinns.svg", dpi=300, bbox_inches='tight')


def plot_bem_error(X, Y, u_inc_amp, u_scn_amp, u_amp, u_inc_phase, u_scn_phase, u_phase):
    """
    Plot only the scattered amplitude and phase as a row of two figures.

    Parameters:
    X, Y : 2D ndarrays - Grid coordinates.
    u_scn_amp : 2D ndarray - Amplitude of the scattered field.
    u_scn_phase : 2D ndarray - Phase of the scattered field.
    """
    fig, axs = plt.subplots(1, 2, figsize=(3.9, 1.9))
    shrink = 0.5 
  
    c1 = axs[0].pcolormesh(X, Y, u_amp/np.abs(u_scn_amp).max(), cmap="magma", rasterized=True)
    cb1 = fig.colorbar(c1, ax=axs[0], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb1.set_label(r"|Error| / max($u$)", fontsize=8)
    cb1.set_ticks([0, np.max(np.abs(u_amp)/np.abs(u_scn_amp).max())])
    cb1.set_ticklabels([f'{0:.1f}', f'{np.max(np.abs(u_amp)/np.abs(u_scn_amp).max()):.4f}'], fontsize=7)
    axs[0].set_title("Amplitude", fontsize=8, pad=6)  
    axs[0].axis("off")
    axs[0].set_aspect("equal")

     
    c2 = axs[1].pcolormesh(X, Y, u_phase/np.abs(u_scn_phase).max(), cmap="magma", rasterized=True)
    cb2 = fig.colorbar(c2, ax=axs[1], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb2.set_label(r"|Error| / max($u$)", fontsize=8)
    
    cb2.set_ticks([0, np.max(u_phase)/np.abs(u_scn_phase).max()])
    cb2.set_ticklabels([f'{0:.1f}', f'{np.max(np.abs(u_phase)/np.abs(u_scn_phase).max()):.4f}'], fontsize=7)
    axs[1].set_title("Phase", fontsize=8, pad=6)  
    axs[1].axis("off")
    axs[1].set_aspect("equal")

    fig.text(0.1, 0.55, r'BEM', fontsize=8, va='center', ha='center', rotation='vertical')

    fig.subplots_adjust(wspace=-0.5)

    plt.tight_layout()
    plt.savefig("figures/bem_error.svg", dpi=150, bbox_inches='tight')
    plt.show()

def plot_pinns_error(X, Y, u_inc_amp, u_scn_amp, u_amp, u_inc_phase, u_scn_phase, u_phase):
    """
    Plot only the scattered amplitude and phase as a row of two figures.

    Parameters:
    X, Y : 2D ndarrays - Grid coordinates.
    u_scn_amp : 2D ndarray - Amplitude of the scattered field.
    u_scn_phase : 2D ndarray - Phase of the scattered field.
    """
    fig, axs = plt.subplots(1, 2, figsize=(3.9, 1.9))
    shrink = 0.5  
  
    c1 = axs[0].pcolormesh(X, Y, u_amp/np.abs(u_scn_amp).max(), cmap="magma", rasterized=True)
    cb1 = fig.colorbar(c1, ax=axs[0], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb1.set_label(r"|Error| / max($u$)", fontsize=8)
    cb1.set_ticks([0, np.max(np.abs(u_amp)/np.abs(u_scn_amp).max())])
    cb1.set_ticklabels([f'{0:.1f}', f'{np.max(np.abs(u_amp)/np.abs(u_scn_amp).max()):.4f}'], fontsize=7)
    axs[0].set_title("Amplitude", fontsize=8, pad=6)  
    axs[0].axis("off")
    axs[0].set_aspect("equal")

     
    c2 = axs[1].pcolormesh(X, Y, u_phase/np.abs(u_scn_phase).max(), cmap="magma", rasterized=True)
    cb2 = fig.colorbar(c2, ax=axs[1], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb2.set_label(r"|Error| / max($u$)", fontsize=8)
    
    cb2.set_ticks([0, np.max(u_phase)/np.abs(u_scn_phase).max()])
    cb2.set_ticklabels([f'{0:.1f}', f'{np.max(np.abs(u_phase)/np.abs(u_scn_phase).max()):.4f}'], fontsize=7)
    axs[1].set_title("Phase", fontsize=8, pad=6)  
    axs[1].axis("off")
    axs[1].set_aspect("equal")

    fig.text(0.1, 0.55, r'PINNs', fontsize=8, va='center', ha='center', rotation='vertical')

    fig.subplots_adjust(wspace=-0.5)

    plt.tight_layout()
    plt.savefig("figures/pinns_error.svg", dpi=150, bbox_inches='tight')
    plt.show()


