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