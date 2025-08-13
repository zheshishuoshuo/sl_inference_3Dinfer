import os
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


def load_A_interpolator(filename: str = "A_eta_table_alpha.csv") -> RegularGridInterpolator:
    """Load the pre-computed A(eta) table and return an interpolator.

    The table must contain the columns ``mu_DM``, ``beta_DM``, ``sigma_DM`` and
    ``alpha`` along with ``A``.  The grid is assumed to be regular in each
    dimension.
    """

    df = pd.read_csv(filename)
    mu_unique = np.sort(df["mu_DM"].unique())
    beta_unique = np.sort(df["beta_DM"].unique())
    sigma_unique = np.sort(df["sigma_DM"].unique())
    alpha_unique = np.sort(df["alpha"].unique())

    shape = (
        len(mu_unique),
        len(beta_unique),
        len(sigma_unique),
        len(alpha_unique),
    )
    values = (
        df.set_index(["mu_DM", "beta_DM", "sigma_DM", "alpha"])  # type: ignore[index]
        .sort_index()["A"]
        .values.reshape(shape)
    )

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


def cached_A_interp(mu0: float, betaDM: float, sigmaDM: float, alpha: float) -> float:
    """Interpolation wrapper for the cached A(eta) table."""

    return float(A_interp((mu0, betaDM, sigmaDM, alpha)))
