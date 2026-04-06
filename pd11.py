# https://medium.com/@guyko81/stop-predicting-numbers-start-predicting-distributions-0d4975db52ae
# https://github.com/guyko81/DistributionRegressor

 

"""
Predicting Distributions - pd11
DistributionRegressor: Nonparametric Distributional Regression 
Lotto 7/39 probabilistic predictions
"""


"""
PD11 6 modela ALL (medijana ansambla) 
GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor, KNeighborsRegressor, LGBMRegressor
Samostalan ensemble fajl za Lotto 7/39.
"""

import time
import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False
    LGBMRegressor = None


# -----------------------------
# Konfiguracija
# -----------------------------
SEED = 39
np.random.seed(SEED)

CSV_PATH = "/Users/4c/Desktop/GHQ/data/loto7hh_4592_k27.csv"
COLS = ["Num1", "Num2", "Num3", "Num4", "Num5", "Num6", "Num7"]
FEATURE_COLS = [f"f{i+1}" for i in range(7)]

MIN_POS = np.array([1, 2, 3, 4, 5, 6, 7], dtype=int)
MAX_POS = np.array([33, 34, 35, 36, 37, 38, 39], dtype=int)

# Ukloni poznat warning koji zatrpava izlaz.
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"sklearn\.utils\.validation",
    message=r"X does not have valid feature names, but LGBMRegressor was fitted with feature names",
)


def load_draws(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    if all(c in df.columns for c in COLS):
        arr = df[COLS].values.astype(float)
    else:
        arr = pd.read_csv(csv_path, header=None).iloc[:, :7].values.astype(float)
    return arr


def enforce_loto_7_39(nums_float: np.ndarray) -> np.ndarray:
    nums = np.rint(np.asarray(nums_float, dtype=float)).astype(int)
    nums = np.clip(nums, MIN_POS, MAX_POS)
    nums = np.sort(nums)

    for i in range(7):
        low = MIN_POS[i] if i == 0 else max(MIN_POS[i], nums[i - 1] + 1)
        nums[i] = max(nums[i], low)

    for i in range(6, -1, -1):
        high = MAX_POS[i] if i == 6 else min(MAX_POS[i], nums[i + 1] - 1)
        nums[i] = min(nums[i], high)

    for i in range(7):
        low = MIN_POS[i] if i == 0 else max(MIN_POS[i], nums[i - 1] + 1)
        nums[i] = max(nums[i], low)

    return nums


def make_builders():
    """
    6 internih modela (pd5..pd10)
    """
    return [
        (
            "m1_boosted",
            lambda: GradientBoostingRegressor(
                n_estimators=450,
                learning_rate=0.04,
                max_depth=3,
                random_state=SEED,
                loss="huber",
            ),
        ),
        (
            "m2_cdf_median",
            lambda: GradientBoostingRegressor(
                n_estimators=500,
                learning_rate=0.035,
                max_depth=3,
                random_state=SEED,
                loss="quantile",
                alpha=0.5,
            ),
        ),
        (
            "m3_rf_single",
            lambda: RandomForestRegressor(
                n_estimators=500,
                max_depth=14,
                min_samples_leaf=2,
                random_state=SEED,
                n_jobs=-1,
            ),
        ),
        (
            "m4_rf_extra",
            lambda: ExtraTreesRegressor(
                n_estimators=500,
                max_depth=16,
                min_samples_leaf=2,
                random_state=SEED,
                n_jobs=-1,
            ),
        ),
        (
            "m5_soft_target",
            lambda: KNeighborsRegressor(
                n_neighbors=45,
                weights="distance",
                p=2,
            ),
        ),
        (
            "m6_legacy_lgbm",
            lambda: (
                LGBMRegressor(
                    n_estimators=700,
                    learning_rate=0.03,
                    max_depth=6,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    random_state=SEED,
                    verbosity=-1,
                )
                if HAS_LGBM
                else GradientBoostingRegressor(
                    n_estimators=520,
                    learning_rate=0.035,
                    max_depth=4,
                    random_state=SEED,
                    loss="huber",
                )
            ),
        ),
    ]


def fit_predict_safe(model_name, build, X_df, y_vec, X_next_df):
    t0 = time.time()
    try:
        model = build()
        model.fit(X_df, y_vec)
        pred = float(np.asarray(model.predict(X_next_df)).ravel()[0])
        dt = time.time() - t0
        print(f"  - {model_name}: {pred:.6f}  (t={dt:.1f}s)")
        return pred
    except Exception as e:
        dt = time.time() - t0
        print(f"  - {model_name}: SKIP ({e})  (t={dt:.1f}s)")
        return None


def main():
    draws = load_draws(CSV_PATH)
    X = pd.DataFrame(draws[:-1], columns=FEATURE_COLS)
    Y = draws[1:]
    X_next = pd.DataFrame(draws[-1:].astype(float), columns=FEATURE_COLS)

    builders = make_builders()
    model_names = [name for name, _ in builders]
    per_model_raw = {name: [] for name in model_names}

    print("=" * 72)
    print("PD11 6 ALL - Standalone 6-model prediction (Loto 7/39)")
    print("=" * 72)
    print(f"CSV: {CSV_PATH}")
    print(f"Uzoraka za trening: {len(X)}")
    print(f"Modela: {len(builders)}")
    print(f"LGBM dostupan: {'DA' if HAS_LGBM else 'NE (fallback aktivan)'}")
    print()

    # Trening i predikcija po poziciji
    for pos in range(7):
        y_pos = Y[:, pos]
        print(f"[pozicija {pos + 1}] trening + predikcija...")
        for name, build in builders:
            p = fit_predict_safe(name, build, X, y_pos, X_next)
            if p is not None:
                per_model_raw[name].append(p)
        print()

    # Posebna predikcija za svaki model
    print("=" * 72)
    print("POSEBNE PREDIKCIJE PO MODELU")
    print("=" * 72)
    per_model_final = {}
    for name in model_names:
        raw = per_model_raw.get(name, [])
        if len(raw) != 7:
            print(f"{name}: SKIP (nema svih 7 pozicija)")
            continue
        comb = enforce_loto_7_39(np.array(raw, dtype=float))
        per_model_final[name] = comb
        print(f"{name}: {comb}")

    # ALL predikcija: medijana modela po poziciji
    if not per_model_final:
        raise RuntimeError("Nijedan model nije dao kompletnih 7 pozicija.")

    stacked = np.array([per_model_raw[n] for n in model_names if len(per_model_raw[n]) == 7], dtype=float)
    all_raw = np.median(stacked, axis=0)
    all_pred = enforce_loto_7_39(all_raw)

    print()
    print("=" * 72)
    print("ALL PREDICTION:", all_pred)
    print("=" * 72)


if __name__ == "__main__":
    main()

"""
Uzoraka za trening: 4591
Modela: 6
LGBM dostupan: DA

[pozicija 1] trening + predikcija...
  - m1_boosted: 4.052099  (t=1.6s)
  - m2_cdf_median: 3.931348  (t=1.4s)
  - m3_rf_single: 4.611446  (t=0.5s)
  - m4_rf_extra: 4.435116  (t=0.4s)
  - m5_soft_target: 4.501884  (t=0.0s)
  - m6_legacy_lgbm: 4.605944  (t=1.8s)

[pozicija 2] trening + predikcija...
  - m1_boosted: 9.051870  (t=1.5s)
  - m2_cdf_median: 8.060547  (t=1.4s)
  - m3_rf_single: 9.230569  (t=0.5s)
  - m4_rf_extra: 9.592852  (t=0.3s)
  - m5_soft_target: 9.896982  (t=0.0s)
  - m6_legacy_lgbm: 8.710657  (t=1.7s)

[pozicija 3] trening + predikcija...
  - m1_boosted: 14.551470  (t=1.5s)
  - m2_cdf_median: 14.369185  (t=1.4s)
  - m3_rf_single: 14.926078  (t=0.5s)
  - m4_rf_extra: 15.723219  (t=0.3s)
  - m5_soft_target: 15.647430  (t=0.0s)
  - m6_legacy_lgbm: 14.627534  (t=1.7s)

[pozicija 4] trening + predikcija...
  - m1_boosted: 19.829642  (t=1.5s)
  - m2_cdf_median: 20.042729  (t=1.4s)
  - m3_rf_single: 19.665341  (t=0.5s)
  - m4_rf_extra: 20.238076  (t=0.3s)
  - m5_soft_target: 20.291073  (t=0.0s)
  - m6_legacy_lgbm: 18.838339  (t=1.9s)

[pozicija 5] trening + predikcija...
  - m1_boosted: 24.082444  (t=1.5s)
  - m2_cdf_median: 25.921422  (t=1.4s)
  - m3_rf_single: 25.053232  (t=0.5s)
  - m4_rf_extra: 25.126639  (t=0.3s)
  - m5_soft_target: 24.990751  (t=0.0s)
  - m6_legacy_lgbm: 23.444292  (t=2.1s)

[pozicija 6] trening + predikcija...
  - m1_boosted: 29.729726  (t=1.5s)
  - m2_cdf_median: 30.966038  (t=1.4s)
  - m3_rf_single: 29.897694  (t=0.5s)
  - m4_rf_extra: 30.822915  (t=0.3s)
  - m5_soft_target: 30.915698  (t=0.0s)
  - m6_legacy_lgbm: 29.891405  (t=2.0s)

[pozicija 7] trening + predikcija...
  - m1_boosted: 35.453766  (t=1.5s)
  - m2_cdf_median: 36.005452  (t=1.4s)
  - m3_rf_single: 35.424320  (t=0.5s)
  - m4_rf_extra: 35.325232  (t=0.3s)
  - m5_soft_target: 35.519957  (t=0.0s)
  - m6_legacy_lgbm: 34.633822  (t=2.0s)

========================================================================
POSEBNE PREDIKCIJE PO MODELU
========================================================================
m1_boosted: [ 4  9 15 20 24 30 35]
m2_cdf_median: [ 4  8 14 20 26 31 36]
m3_rf_single: [ 5  9 15 20 25 30 35]
m4_rf_extra: [ 4 10 16 20 25 31 35]
m5_soft_target: [ 5 10 16 20 25 31 36]
m6_legacy_lgbm: [ 5  9 15 19 23 30 35]

========================================================================
ALL PREDICTION: [ 4  9 15 20 25 30 35]
========================================================================
"""
