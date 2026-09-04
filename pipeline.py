"""
Hard-braking prediction pipeline.

Predicts label_hard_braking_3s (will a hard-braking event occur in the next 3s)
using only information available up to and including time t.

Run: python3 pipeline.py
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_recall_curve,
    confusion_matrix, precision_score, recall_score
)
import json

RAW_PATH = "driving_telemetry.csv"
HZ = 10.0
DT = 1.0 / HZ

# ---------------------------------------------------------------------------
# 1. LOAD + CLEAN
# ---------------------------------------------------------------------------

def load_and_clean(path):
    df = pd.read_csv(path)
    n_raw = len(df)

    # --- 1a. Exact duplicate (trip_id, t) rows -----------------------------
    # Investigation showed every trip has exactly 1800 unique timestamps
    # (0.0 to 179.9 at 0.1s) plus a handful of *exact repeat* rows appended
    # at the end of the file. These are not "the same instant measured twice"
    # with different sensor noise -- they are byte-identical duplicate rows.
    # Safe to drop, keeping the first occurrence (arbitrary, since identical).
    df = df.sort_values(["trip_id", "t"]).reset_index(drop=True)
    n_dupe = df.duplicated(subset=["trip_id", "t"]).sum()
    df = df.drop_duplicates(subset=["trip_id", "t"], keep="first").reset_index(drop=True)

    # --- 1b. Drop lat/lon -----------------------------------------------
    # GPS is logged at 1Hz vs 10Hz for everything else (confirmed: present
    # every 10th row exactly). >90% missing at the 10Hz grain. Raw position
    # carries no obvious causal signal for *braking* (as opposed to route
    # matching), and forward-filling it 10x would fabricate resolution that
    # doesn't exist. Dropped rather than imputed.
    df = df.drop(columns=["lat", "lon"])

    # --- 1c. Drop the leaky EMA column -----------------------------------
    # brake_pressure_ema_5s was verified (by grid-searching window sizes
    # against a pandas rolling-mean) to be an EXACTLY CENTERED 51-sample
    # (5.0s) rolling mean of brake_pressure, i.e. it mixes in brake_pressure
    # readings up to +2.5s in the future. Since the label horizon is only
    # 3s, this column leaks directly into the label -- using it produces a
    # model that looks excellent and is unusable in production, where the
    # future doesn't exist yet. It is dropped entirely and replaced with a
    # causal (trailing-only) EMA computed below.
    df = df.drop(columns=["brake_pressure_ema_5s"])

    # --- 1d. IMU sensor dropout -------------------------------------------
    # accel_x/y/z/gyro_z missing on ~2% of rows, as isolated single-row
    # dropouts (not sustained outages) in every trip. Forward-filled within
    # each trip with a short limit (2 samples = 0.2s) -- this is causal
    # (uses only past values) and matches the "brief sensor glitch" pattern
    # observed. Any leading NaNs at the very start of a trip (before any
    # real reading exists) are back-filled just for that opening fraction
    # of a second, since there is no legitimate causal value to use there.
    imu_cols = ["accel_x", "accel_y", "accel_z", "gyro_z"]
    df[imu_cols] = (
        df.groupby("trip_id")[imu_cols]
        .apply(lambda g: g.ffill(limit=2).bfill(limit=2))
        .reset_index(drop=True)
    )

    meta = dict(n_raw=n_raw, n_dupe_dropped=int(n_dupe), n_after_clean=len(df))
    return df, meta


# ---------------------------------------------------------------------------
# 2. CAUSAL FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def add_causal_features(df):
    df = df.sort_values(["trip_id", "t"]).reset_index(drop=True)
    out = []
    for tid, g in df.groupby("trip_id", sort=False):
        g = g.copy()

        # Trailing (causal) EMA of brake pressure at a couple of horizons.
        g["bp_ema_1s"] = g["brake_pressure"].ewm(span=10, adjust=False).mean()
        g["bp_ema_2s"] = g["brake_pressure"].ewm(span=20, adjust=False).mean()

        # Trailing rolling stats (windows look only backward: min_periods=1,
        # default pandas rolling() is trailing/causal).
        for w, label in [(5, "0_5s"), (10, "1s"), (20, "2s")]:
            g[f"bp_max_{label}"] = g["brake_pressure"].rolling(w, min_periods=1).max()
            g[f"bp_std_{label}"] = g["brake_pressure"].rolling(w, min_periods=1).std().fillna(0)
            g[f"speed_min_{label}"] = g["speed_kmph"].rolling(w, min_periods=1).min()

        # Rate of change / "jerk"-style causal derivatives (backward diff).
        g["bp_roc"] = g["brake_pressure"].diff().fillna(0) / DT
        g["accel_x_roc"] = g["accel_x"].diff().fillna(0) / DT
        g["speed_roc"] = g["speed_kmph"].diff().fillna(0) / DT  # longitudinal accel proxy

        # Deceleration over the last 1s / 2s (causal, backward-looking).
        g["speed_delta_1s"] = g["speed_kmph"] - g["speed_kmph"].shift(10).bfill()
        g["speed_delta_2s"] = g["speed_kmph"] - g["speed_kmph"].shift(20).bfill()

        # Rolling std of accel_x / gyro_z as a noisiness / swerve proxy.
        g["accel_x_std_1s"] = g["accel_x"].rolling(10, min_periods=1).std().fillna(0)
        g["gyro_z_std_1s"] = g["gyro_z"].rolling(10, min_periods=1).std().fillna(0)

        out.append(g)
    return pd.concat(out, ignore_index=True)


FEATURE_COLS = [
    "speed_kmph", "accel_x", "accel_y", "accel_z", "gyro_z",
    "steering_angle_deg", "brake_pressure",
    "bp_ema_1s", "bp_ema_2s",
    "bp_max_0_5s", "bp_std_0_5s", "speed_min_0_5s",
    "bp_max_1s", "bp_std_1s", "speed_min_1s",
    "bp_max_2s", "bp_std_2s", "speed_min_2s",
    "bp_roc", "accel_x_roc", "speed_roc",
    "speed_delta_1s", "speed_delta_2s",
    "accel_x_std_1s", "gyro_z_std_1s",
]


# ---------------------------------------------------------------------------
# 3. SPLIT
# ---------------------------------------------------------------------------

def trip_split(df, train_trips, val_trips, test_trips):
    return (
        df[df.trip_id.isin(train_trips)].copy(),
        df[df.trip_id.isin(val_trips)].copy(),
        df[df.trip_id.isin(test_trips)].copy(),
    )


# ---------------------------------------------------------------------------
# 4. TRAIN / EVAL
# ---------------------------------------------------------------------------

def fit_logreg(train, val):
    scaler = StandardScaler().fit(train[FEATURE_COLS])
    Xtr = scaler.transform(train[FEATURE_COLS])
    Xval = scaler.transform(val[FEATURE_COLS])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Xtr, train["label_hard_braking_3s"])
    return clf, scaler


def fit_gbt(train):
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=4,
        class_weight="balanced", random_state=0,
    )
    clf.fit(train[FEATURE_COLS], train["label_hard_braking_3s"])
    return clf


def eval_model(name, y_true, y_score, results):
    ap = average_precision_score(y_true, y_score)
    roc = roc_auc_score(y_true, y_score)
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    # recall achievable at >=0.5 precision
    mask = prec[:-1] >= 0.5
    recall_at_p50 = rec[:-1][mask].max() if mask.any() else 0.0
    results[name] = dict(pr_auc=float(ap), roc_auc=float(roc), recall_at_p50=float(recall_at_p50))
    print(f"{name:30s} PR-AUC={ap:.3f}  ROC-AUC={roc:.3f}  Recall@P>=0.5={recall_at_p50:.3f}")
    return ap, roc, recall_at_p50


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df, meta = load_and_clean(RAW_PATH)
    print("Cleaning summary:", meta)

    df = add_causal_features(df)
    assert df[FEATURE_COLS].isna().sum().sum() == 0, "NaNs remain in features!"

    # Trip-level split: 6 train / 1 val / 1 test.
    all_trips = sorted(df.trip_id.unique())
    train_trips, val_trips, test_trips = all_trips[:6], all_trips[6:7], all_trips[7:8]
    train, val, test = trip_split(df, train_trips, val_trips, test_trips)
    print(f"trips -> train {train_trips}, val {val_trips}, test {test_trips}")
    print(f"rows  -> train {len(train)}, val {len(val)}, test {len(test)}")
    print(f"positive rate -> train {train.label_hard_braking_3s.mean():.4f}, "
          f"val {val.label_hard_braking_3s.mean():.4f}, test {test.label_hard_braking_3s.mean():.4f}")

    results = {}

    # Baseline: raw brake_pressure alone (sanity floor).
    eval_model("baseline: brake_pressure only", test["label_hard_braking_3s"], test["brake_pressure"], results)

    # Logistic regression.
    lr, scaler = fit_logreg(train, val)
    val_score_lr = lr.predict_proba(scaler.transform(val[FEATURE_COLS]))[:, 1]
    eval_model("logreg (val)", val["label_hard_braking_3s"], val_score_lr, results)
    test_score_lr = lr.predict_proba(scaler.transform(test[FEATURE_COLS]))[:, 1]
    eval_model("logreg (test)", test["label_hard_braking_3s"], test_score_lr, results)

    # Gradient boosted trees.
    gbt = fit_gbt(train)
    val_score_gbt = gbt.predict_proba(val[FEATURE_COLS])[:, 1]
    eval_model("gbt (val)", val["label_hard_braking_3s"], val_score_gbt, results)
    test_score_gbt = gbt.predict_proba(test[FEATURE_COLS])[:, 1]
    eval_model("gbt (test)", test["label_hard_braking_3s"], test_score_gbt, results)

    # Leave-one-trip-out CV with GBT, since 8 trips / 8 events is too few
    # for a single split to be a trustworthy estimate.
    print("\nLeave-one-trip-out CV (GBT):")
    loto_aps = []
    for held_out in all_trips:
        tr = df[df.trip_id != held_out]
        te = df[df.trip_id == held_out]
        m = fit_gbt(tr)
        s = m.predict_proba(te[FEATURE_COLS])[:, 1]
        ap = average_precision_score(te["label_hard_braking_3s"], s)
        loto_aps.append(ap)
        print(f"  held-out trip {held_out}: PR-AUC={ap:.3f}  n_pos={int(te.label_hard_braking_3s.sum())}")
    results["loto_pr_auc_mean"] = float(np.mean(loto_aps))
    results["loto_pr_auc_std"] = float(np.std(loto_aps))
    results["loto_pr_auc_per_trip"] = [float(x) for x in loto_aps]
    print(f"LOTO PR-AUC: mean={np.mean(loto_aps):.3f} std={np.std(loto_aps):.3f}")

    # Feature importance (permutation-free: HGB gives no direct importances,
    # so use a quick GBT-on-all-but-test fit's built-in via sklearn's
    # `feature_importances_`-less HGB -> fall back to permutation importance).
    from sklearn.inspection import permutation_importance
    pi = permutation_importance(gbt, test[FEATURE_COLS], test["label_hard_braking_3s"],
                                 scoring="average_precision", n_repeats=10, random_state=0)
    importances = sorted(zip(FEATURE_COLS, pi.importances_mean), key=lambda x: -x[1])
    print("\nTop permutation importances (test, GBT, PR-AUC drop):")
    for feat, imp in importances[:10]:
        print(f"  {feat:20s} {imp:.4f}")
    results["top_features"] = [(f, float(i)) for f, i in importances[:10]]

    # ---- Error analysis on test trip (GBT) ----
    test = test.copy()
    test["score"] = test_score_gbt
    test["pred"] = (test["score"] >= 0.5).astype(int)
    cm = confusion_matrix(test["label_hard_braking_3s"], test["pred"])
    print("\nConfusion matrix (test, thr=0.5):\n", cm)
    results["test_confusion_matrix_thr0.5"] = cm.tolist()

    fn = test[(test.label_hard_braking_3s == 1) & (test.pred == 0)]
    fp = test[(test.label_hard_braking_3s == 0) & (test.pred == 1)]
    print(f"\nFalse negatives: {len(fn)} rows, t range within trip: "
          f"{fn.t.min() if len(fn) else None} - {fn.t.max() if len(fn) else None}")
    print(f"False positives: {len(fp)} rows, t range within trip: "
          f"{fp.t.min() if len(fp) else None} - {fp.t.max() if len(fp) else None}")

    results["n_false_negatives"] = int(len(fn))
    results["n_false_positives"] = int(len(fp))

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    df.to_csv("cleaned_features.csv", index=False)
    test.to_csv("test_predictions.csv", index=False)
    print("\nSaved results.json, cleaned_features.csv, test_predictions.csv")
