import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def load_A_interpolator(filename: str = "A_eta_table_alpha.csv") -> RegularGridInterpolator:
    """Load the pre-computed ``A(eta)`` table and return an interpolator.

    The stored table is four-dimensional with columns ``mu_DM``, ``beta_DM``,
    ``sigma_DM`` and ``alpha`` along with the normalisation ``A``.  This helper
    uses :func:`numpy.loadtxt` to avoid requiring :mod:`pandas`.
    """

    data = np.loadtxt(filename, delimiter=",", skiprows=1)
    mu_unique = np.unique(data[:, 0])
    beta_unique = np.unique(data[:, 1])
    sigma_unique = np.unique(data[:, 2])
    alpha_unique = np.unique(data[:, 3])

    shape = (len(mu_unique), len(beta_unique), len(sigma_unique), len(alpha_unique))
    values = data[:, 4].reshape(shape)

    return RegularGridInterpolator(
        (mu_unique, beta_unique, sigma_unique, alpha_unique),
        values,
        bounds_error=False,
        fill_value=None,
    )


# Load default interpolator ----------------------------------------------------
A_interp = load_A_interpolator(
    os.path.join(os.path.dirname(__file__), "A_eta_table_alpha.csv")
)


def cached_A_interp(mu0: float, sigmaDM: float, alpha: float, betaDM: float = 2.04) -> float:
    """Interpolation wrapper for the cached ``A(eta)`` table with fixed ``betaDM``."""

    return float(A_interp((mu0, betaDM, sigmaDM, alpha)))
