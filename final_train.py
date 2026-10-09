"""
Exoplanet Mass Estimator -- FINAL training run.

What this does:
  1. Loads the NASA PSCompPars table, keeps only genuinely measured masses.
  2. Splits 70% train / 30% held-out test (every test planet has a true mass).
     The split is GROUPED BY HOST STAR so planets of the same system never
     straddle the split (they share stellar features -> would leak).
  3. Tunes the random forest's hyperparameters with 5-fold CV on the
     TRAINING SET ONLY. Feature "weighting" is learned by the model itself;
     tuning controls how it uses the features.
  4. Evaluates ONCE on the untouched 30%.
  5. Saves: the trained model, the held-out test set (with true masses),
     and a metrics report.
"""

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42

FEATURES = [
    "pl_rade",      # planet radius [Earth radii]
    "pl_orbper",    # orbital period [days]
    "pl_orbsmax",   # semi-major axis [AU]
    "pl_eqt",       # equilibrium temperature [K]
    "pl_insol",     # insolation flux [Earth flux]
    "st_teff",      # star temperature [K]
    "st_mass",      # star mass [solar masses]
    "st_rad",       # star radius [solar radii]
    "st_met",       # star metallicity [dex]
]
TARGET = "pl_bmasse"


def load_and_clean(path):
    df = pd.read_csv(path, comment="#")
    n0 = len(df)
    df = df[df["pl_bmassprov"] == "Mass"]          # measured masses only
    df = df.dropna(subset=[TARGET, "pl_rade"])
    df = df[df[TARGET] > 0]
    print(f"Loaded {n0} planets -> {len(df)} usable (measured mass + radius)")
    return df


def build_pipeline(model):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", model),
    ])


def report(name, y_true_log, y_pred_log, lines):
    mae = mean_absolute_error(y_true_log, y_pred_log)
    rmse = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    r2 = r2_score(y_true_log, y_pred_log)
    factor = 10 ** mae
    txt = (f"{name}\n"
           f"  R^2 (log space)  : {r2:.3f}\n"
           f"  MAE (log space)  : {mae:.3f} dex\n"
           f"  RMSE (log space) : {rmse:.3f} dex\n"
           f"  typical miss     : factor of {factor:.2f}x\n")
    print("\n" + txt)
    lines.append(txt)
    return r2


def main(path):
    lines = []
    df = load_and_clean(path)

    X = df[FEATURES]
    y = np.log10(df[TARGET])                 # predict log10(mass)
    groups = df["hostname"]                  # keep systems together

    # ---- 70/30 split, grouped by host star -------------------------------
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.30,
                                 random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    print(f"Train: {len(X_train)} planets   Held-out test: {len(X_test)} planets")
    print(f"(no host star appears on both sides of the split)")

    # ---- Baseline for reference ------------------------------------------
    lin = build_pipeline(LinearRegression())
    lin.fit(X_train, y_train)
    report("Linear regression (baseline)", y_test, lin.predict(X_test), lines)

    # ---- Hyperparameter tuning on TRAINING data only ---------------------
    # This is the legitimate version of "adjusting the weighting":
    # max_features controls how features compete within each tree;
    # min_samples_leaf controls smoothing; n_estimators controls averaging.
    grid = {
        "model__n_estimators": [300, 600],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": [0.5, 0.75, 1.0],
    }
    search = GridSearchCV(
        build_pipeline(RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
        grid, cv=KFold(5, shuffle=True, random_state=RANDOM_STATE),
        scoring="r2", n_jobs=-1,
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_
    print(f"\nBest hyperparameters: {search.best_params_}")
    print(f"CV R^2 during tuning : {search.best_score_:.3f}")
    lines.append(f"Best hyperparameters: {search.best_params_}\n"
                 f"CV R^2 during tuning: {search.best_score_:.3f}\n")

    # ---- Final, single evaluation on the untouched 30% -------------------
    report("TUNED RANDOM FOREST -- held-out 30% test set",
           y_test, best.predict(X_test), lines)

    # ---- Learned feature importances -------------------------------------
    imps = best.named_steps["model"].feature_importances_
    order = np.argsort(imps)[::-1]
    imp_txt = "Learned feature importances:\n" + "".join(
        f"  {FEATURES[i]:12s} {imps[i]:.3f}\n" for i in order)
    print("\n" + imp_txt)
    lines.append(imp_txt)

    # ---- Save everything --------------------------------------------------
    joblib.dump(best, "exoplanet_mass_model.pkl")

    holdout = df.iloc[test_idx][["pl_name", "hostname"] + FEATURES + [TARGET]].copy()
    holdout["predicted_mass_earths"] = 10 ** best.predict(X_test)
    holdout = holdout.rename(columns={TARGET: "true_mass_earths"})
    holdout.to_csv("holdout_test_set.csv", index=False)

    with open("training_report.txt", "w") as f:
        f.write("EXOPLANET MASS ESTIMATOR -- TRAINING REPORT\n")
        f.write("=" * 44 + "\n\n")
        f.write("\n".join(lines))

    print("Saved: exoplanet_mass_model.pkl  (predicts log10 mass; use 10**pred)")
    print(f"Saved: holdout_test_set.csv     ({len(holdout)} planets, true + predicted masses)")
    print("Saved: training_report.txt")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "PSCompPars.csv")
