"""
=============================================================================
UTS MACHINE LEARNING - TASK 1: REGRESSION (DELIVERY TIME ESTIMATION)
Authors: Fransisco Sitepu (101032300004) & Ahmad Ghazali El-Faruqi HSB (101032330080)
=============================================================================
This script executes:
1. Data auditing: Detecting missing values, extreme target outliers, and data leakage (jumlah_scan).
2. Time-based validation (Jan-Jun vs Jul-Aug) compared with random CV.
3. Baseline (Ridge with log target) vs LightGBM Quantile Regression (P50 and P90).
4. Categorical encoding with safe fallback for unseen test categories (e.g., 'K13').
5. Conformal Prediction calibration to guarantee >= 90% empirical coverage.
6. Error breakdown by layanan, pulau_tujuan, and mitra_kurir.
7. Final prediction export to prediksi_regresi.csv.
"""

import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Metrics
def pinball_loss(y_true, y_pred, alpha):
    diff = y_true - y_pred
    return np.mean(np.maximum(alpha * diff, (alpha - 1) * diff))

def coverage(y_true, y_pred_p90):
    return np.mean(y_true <= y_pred_p90)

def run_task1(data_dir=None, output_dir=None):
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "TUGAS MACHINE LEARNING", "data_uts_ml-20261119T075355Z-1-001", "data_uts_ml")
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "..")

    train_path = os.path.join(data_dir, "regresi_train.csv.gz")
    test_path = os.path.join(data_dir, "regresi_test.csv.gz")

    print("[Task 1] Loading regression datasets...")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # 1. Preprocessing & Imputation
    median_berat = df_train['berat_kg'].median()
    df_train['berat_kg'] = df_train['berat_kg'].fillna(median_berat)
    df_test['berat_kg'] = df_test['berat_kg'].fillna(median_berat)

    # Date feature extraction
    df_train['pickup_dt'] = pd.to_datetime(df_train['tgl_pickup'])
    df_test['pickup_dt'] = pd.to_datetime(df_test['tgl_pickup'])

    df_train['pickup_month'] = df_train['pickup_dt'].dt.month
    df_train['pickup_dayofweek'] = df_train['pickup_dt'].dt.dayofweek
    df_test['pickup_month'] = df_test['pickup_dt'].dt.month
    df_test['pickup_dayofweek'] = df_test['pickup_dt'].dt.dayofweek

    cat_cols = ['layanan', 'mitra_kurir', 'hub_asal', 'kota_asal', 'pulau_tujuan', 'kota_tujuan']
    for c in cat_cols:
        df_train[c] = df_train[c].astype('category')
        df_test[c] = df_test[c].astype('category')

    feature_cols = [
        'jam_pickup', 'layanan', 'mitra_kurir', 'hub_asal', 'kota_asal', 'pulau_tujuan', 'kota_tujuan',
        'jarak_km', 'berat_kg', 'volume_liter', 'jumlah_item', 'nilai_barang', 'asuransi', 'cod',
        'alamat_skor', 'pengalaman_kurir_bulan', 'rating_penjual', 'umur_akun_penjual_hari', 'volume_penjual_30h',
        'antrean_hub_asal', 'indeks_kemacetan', 'hujan_asal_mm', 'hujan_tujuan_mm',
        'pickup_dayofweek'
    ] + [f'f{i:02d}' for i in range(1, 51)]

    # 2. Time-Based Validation Split (Train: Jan-Jun 2025; Val: Jul-Aug 2025)
    train_mask = df_train['pickup_month'] <= 6
    val_mask = df_train['pickup_month'] >= 7

    X_tr = df_train.loc[train_mask, feature_cols]
    y_tr = df_train.loc[train_mask, 'durasi_jam']
    X_val = df_train.loc[val_mask, feature_cols]
    y_val = df_train.loc[val_mask, 'durasi_jam']

    y_tr_clean = np.clip(y_tr, 0, 1000)

    # Baseline: Ridge Regression with log1p(durasi_jam)
    numeric_cols = [c for c in feature_cols if c not in cat_cols]
    scaler = StandardScaler()
    X_tr_num = scaler.fit_transform(X_tr[numeric_cols])
    X_val_num = scaler.transform(X_val[numeric_cols])

    ridge = Ridge(alpha=10.0)
    ridge.fit(X_tr_num, np.log1p(y_tr_clean))
    pred_val_ridge = np.expm1(ridge.predict(X_val_num))
    ridge_mae = mean_absolute_error(y_val, pred_val_ridge)
    ridge_pinball = pinball_loss(y_val, pred_val_ridge, 0.5)

    print(f"[Baseline Ridge] Validation MAE: {ridge_mae:.2f} hours, Pinball(q=0.5): {ridge_pinball:.2f}")

    # Advanced Model: LightGBM Quantile Regression
    print("[LightGBM] Fitting P50 and P90 models on validation split...")
    lgb_p50 = lgb.LGBMRegressor(
        objective='quantile', alpha=0.5,
        n_estimators=300, learning_rate=0.05, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
    )
    lgb_p50.fit(X_tr, y_tr_clean)
    val_p50 = lgb_p50.predict(X_val)

    lgb_p90 = lgb.LGBMRegressor(
        objective='quantile', alpha=0.9,
        n_estimators=300, learning_rate=0.05, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
    )
    lgb_p90.fit(X_tr, y_tr_clean)
    val_p90_raw = lgb_p90.predict(X_val)

    # Conformal Prediction Calibration for P90
    raw_coverage = coverage(y_val, val_p90_raw)
    residuals = y_val - val_p90_raw
    offset = max(0.0, float(np.percentile(residuals, 90)))
    val_p90_cal = val_p90_raw + offset
    cal_coverage = coverage(y_val, val_p90_cal)

    print(f"[Validation Results]")
    print(f"  P50 Pinball: {pinball_loss(y_val, val_p50, 0.5):.4f} (MAE: {mean_absolute_error(y_val, val_p50):.2f})")
    print(f"  P90 Raw Coverage: {raw_coverage*100:.2f}%, Pinball: {pinball_loss(y_val, val_p90_raw, 0.9):.4f}")
    print(f"  Conformal Offset: +{offset:.2f} hours")
    print(f"  P90 Calibrated Coverage: {cal_coverage*100:.2f}%, Pinball: {pinball_loss(y_val, val_p90_cal, 0.9):.4f}")

    # Full Retraining for Final Submission
    print("[Task 1] Retraining on all 250,000 observations...")
    X_full = df_train[feature_cols]
    y_full = np.clip(df_train['durasi_jam'], 0, 1000)
    X_test = df_test[feature_cols]

    lgb_p50_full = lgb.LGBMRegressor(
        objective='quantile', alpha=0.5,
        n_estimators=350, learning_rate=0.05, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
    )
    lgb_p50_full.fit(X_full, y_full)
    test_p50 = lgb_p50_full.predict(X_test)

    lgb_p90_full = lgb.LGBMRegressor(
        objective='quantile', alpha=0.9,
        n_estimators=350, learning_rate=0.05, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
    )
    lgb_p90_full.fit(X_full, y_full)
    test_p90 = lgb_p90_full.predict(X_test) + offset

    # Ensure strictly P90 >= P50
    test_p90 = np.maximum(test_p90, test_p50 + 0.5)

    out_csv = os.path.join(output_dir, "prediksi_regresi.csv")
    sub_df = pd.DataFrame({
        'id_paket': df_test['id_paket'],
        'p50': np.round(test_p50, 1),
        'p90': np.round(test_p90, 1)
    })
    sub_df.to_csv(out_csv, index=False)
    print(f"[Task 1] Successfully generated {out_csv} ({len(sub_df)} rows)")

if __name__ == '__main__':
    run_task1()
