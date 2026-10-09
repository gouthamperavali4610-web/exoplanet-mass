# Exoplanet Mass Estimator

A machine-learning model that estimates an exoplanet's mass (in Earth masses) from its transit-observable and host-star properties, trained on the NASA Exoplanet Archive.

## Why

Most exoplanets are discovered by the transit method, which measures a planet's **radius** but not its **mass**. Mass requires radial-velocity follow-up that many planets never get. This model fills that gap: given a planet's radius, orbit, and host-star properties, it estimates the mass — a machine-learned mass–radius relation.

## Results

Trained on 1,691 planets with genuinely measured masses; evaluated on a held-out 30% (689 planets, grouped by host star so no system appears on both sides of the split).

| Model | R² (log space) | Typical error |
|---|---|---|
| Linear regression (baseline) | 0.637 | factor of 2.81× |
| Tuned random forest | **0.824** | **factor of 1.91×** |

Radius is the dominant learned feature (importance 0.62), followed by orbital semi-major axis and insolation — consistent with the physics of mass–radius relations.

## Key design decisions

- **Provenance filter.** The archive lists a mass for ~99% of planets, but nearly half are *computed from radius* via an existing M–R relation (`pl_bmassprov = "M-R relationship"`). Training on those would just reverse-engineer someone else's formula. Only rows with `pl_bmassprov = "Mass"` (genuinely measured) are used.
- **Log-transformed target.** Masses span five orders of magnitude (0.02 to ~9,500 Earth masses). The model predicts log10(mass); predictions are converted back with `10**pred`.
- **Group-aware split.** Planets orbiting the same star share all stellar features, so the train/test split keeps each host star's planets together to prevent leakage.

## Files

- `final_train.py` — full training pipeline: cleaning, 70/30 group split, hyperparameter tuning (CV on train only), evaluation, artifact saving
- `predict.py` — load the trained model and estimate masses for new planets
- `exoplanet_mass_model.pkl.gz` — trained model (predicts log10 mass)
- `holdout_test_set.csv` — the 689 held-out planets with true and predicted masses
- `training_report.txt` — metrics from the final run

## Run it

```bash
pip install -r requirements.txt
python final_train.py PSCompPars_2026_08_03_09_40_35.csv
```

Data: NASA Exoplanet Archive, Planetary Systems Composite Parameters table (retrieved August 2026). Download from https://exoplanetarchive.ipac.caltech.edu

## Limitations

- The training sample is detection-biased (dominated by transiting planets — large, close-in) and the model inherits that bias.
- Planets with identical radii can have very different masses (e.g. inflated hot Jupiters); a factor-of-~2 typical error reflects real physical degeneracy, not just model error.
- Estimates for planets unlike the training distribution (extrapolation) are unreliable.

## Acknowledgment

This research has made use of the NASA Exoplanet Archive, operated by Caltech under contract with NASA's Exoplanet Exploration Program.
