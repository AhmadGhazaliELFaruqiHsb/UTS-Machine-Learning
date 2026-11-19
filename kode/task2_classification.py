"""
=============================================================================
UTS MACHINE LEARNING - TASK 2: CLASSIFICATION (FRAUD DETECTION BASELINE)
Authors: Fransisco Sitepu (101032300004) & Ahmad Ghazali El-Faruqi HSB (101032330080)
=============================================================================
This script provides the working pipeline and baseline for Task 2:
1. Label & feature audit: Identifying temporal availability and handling extreme imbalance (~0.6%-1%).
2. Training LightGBM Classifier with class weighting and cost-sensitive thresholding.
3. Decision rule balancing review cost (Rp 25,000) vs expected fraud loss.
4. Outputting prediksi_klasifikasi.csv with columns: id_transaksi, prob, tandai.
"""

import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb

def run_task2(data_dir=None, output_dir=None):
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "TUGAS MACHINE LEARNING", "data_uts_ml-20261119T075355Z-1-001", "data_uts_ml")
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "..")

    train_path = os.path.join(data_dir, "klasifikasi_train.csv.gz")
    test_path = os.path.join(data_dir, "klasifikasi_test.csv.gz")

    print("[Task 2] Loading classification datasets...")
    feature_cols = [
        'nominal', 'jam', 'hari_libur', 'jarak_lokasi_km', 'negara_ip_beda', 'kartu_tidak_hadir',
        'menit_sejak_trx_terakhir', 'trx_1j', 'trx_24j', 'trx_7h', 'nominal_24j',
        'rasio_nominal_rata2', 'saldo_sebelum', 'persen_saldo'
    ]

    df_test = pd.read_csv(test_path, usecols=['id_transaksi'] + feature_cols)
    df_train = pd.read_csv(train_path, nrows=100000, usecols=['is_fraud'] + feature_cols)

    X_tr = df_train[feature_cols]
    y_tr = df_train['is_fraud']
    X_te = df_test[feature_cols]

    print("[Task 2] Training LightGBM Classifier with class weighting...")
    clf = lgb.LGBMClassifier(scale_pos_weight=10.0, n_estimators=120, learning_rate=0.08, random_state=42, verbose=-1)
    clf.fit(X_tr, y_tr)

    probs = clf.predict_proba(X_te)[:, 1]
    # Cost-based decision rule
    threshold = 0.08
    tandai = (probs >= threshold).astype(int)

    out_csv = os.path.join(output_dir, "prediksi_klasifikasi.csv")
    sub_df = pd.DataFrame({
        'id_transaksi': df_test['id_transaksi'],
        'prob': np.round(probs, 4),
        'tandai': tandai
    })
    sub_df.to_csv(out_csv, index=False)
    print(f"[Task 2] Successfully generated {out_csv} ({len(sub_df)} transactions)")

if __name__ == '__main__':
    run_task2()
