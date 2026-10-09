"""
Exoplanet Mass Estimator — inference.

Loads the trained model and estimates masses for planets where mass was
never measured. This is the real point of the project: the archive has
thousands of planets with a radius from the transit method but no mass
from radial velocity.
"""

import numpy as np
import pandas as pd
import joblib

from final_train import FEATURES   # same feature list, same order -- important

MODEL_PATH = "exoplanet_mass_model.pkl.gz"


def predict_one(model, **kwargs):
    """
    Estimate the mass of a single planet.

    Example:
        predict_one(model, pl_rade=2.4, pl_orbper=289.9, st_teff=5518)

    Any feature you leave out is filled with the training-set median by
    the pipeline's imputer -- so it still works, just less confidently.
    """
    row = pd.DataFrame([{f: kwargs.get(f, np.nan) for f in FEATURES}])
    log_mass = model.predict(row)[0]
    return 10 ** log_mass          # undo the log10 -> Earth masses


def predict_missing(model, csv_path, out_path="predicted_masses.csv"):
    """
    Fill in the archive's gaps: planets with a radius but no measured mass.
    These are the rows where pl_bmassprov is NOT "Mass" -- i.e. the mass
    was estimated or is a lower bound. We give our own estimate instead.
    """
    df = pd.read_csv(csv_path, comment="#")
    gaps = df[(df["pl_bmassprov"] != "Mass") & df["pl_rade"].notna()].copy()

    gaps["predicted_mass_earths"] = 10 ** model.predict(gaps[FEATURES])

    out = gaps[["pl_name", "pl_rade", "pl_bmassprov",
                "pl_bmasse", "predicted_mass_earths"]]
    out.to_csv(out_path, index=False)
    print(f"Estimated masses for {len(out)} planets -> {out_path}")
    return out


if __name__ == "__main__":
    model = joblib.load(MODEL_PATH)

    # --- single-planet inference (what your Streamlit app will do) ---
    mass = predict_one(model, pl_rade=2.4, pl_orbper=289.9,
                       pl_eqt=262, st_teff=5518, st_mass=0.97)
    print(f"Kepler-22b-like planet -> estimated mass: {mass:.1f} Earth masses")

    mass = predict_one(model, pl_rade=1.0, pl_orbper=365.25,
                       pl_eqt=255, st_teff=5772, st_mass=1.0)
    print(f"Earth-like planet      -> estimated mass: {mass:.2f} Earth masses")

    mass = predict_one(model, pl_rade=11.2, pl_orbper=4331,
                       pl_eqt=110, st_teff=5772, st_mass=1.0)
    print(f"Jupiter-like planet    -> estimated mass: {mass:.0f} Earth masses")
