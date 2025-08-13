import sys
from pathlib import Path
import numpy as np

# Ensure parent directory (containing package) is on path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from sl_inference_4Dinfer.mock_generator.mock_generator import run_mock_simulation
from sl_inference_4Dinfer.make_tabulate import tabulate_likelihood_grids
from sl_inference_4Dinfer.likelihood import log_likelihood
from sl_inference_4Dinfer.mock_generator.mass_sampler import MODEL_PARAMS


def test_log_likelihood_runs():
    # Generate a small mock sample; using a large magnitude limit ensures detection
    _, mock_obs = run_mock_simulation(2, maximum_magnitude=99, process=0, seed=0)

    logMh_grid = np.linspace(10, 12, 3)
    grids = tabulate_likelihood_grids(mock_obs, logMh_grid)
    logM_obs = mock_obs["logM_star_sps_observed"].values

    model_p = MODEL_PARAMS["deVauc"]
    theta = (12.5, model_p["beta_h"], model_p["sigma_h"], 0.1)
    ll = log_likelihood(theta, grids, logM_obs)
    assert np.isfinite(ll)
