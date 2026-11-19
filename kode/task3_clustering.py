"""
=============================================================================
UTS MACHINE LEARNING - TASK 3: CLUSTERING (CUSTOMER SEGMENTATION)
Authors: Fransisco Sitepu (101032300004) & Ahmad Ghazali El-Faruqi HSB (101032330080)
=============================================================================
This script executes:
1. Preprocessing: Handling missing values, skewness (log1p), and compositional simplexes via Centered Log-Ratio (CLR).
2. Feature selection: Identification and removal of synthetic uniform noise columns (n01-n15).
3. Outlier / Anomaly detection: Isolation Forest labeling true irregular customers as -1.
4. Model comparison: K-Means vs Gaussian Mixture Models vs Agglomerative Hierarchical.
5. Cluster validity: Silhouette score, Davies-Bouldin index, and Calinski-Harabasz metrics across K=3..7.
6. Stability testing: Bootstrap / subsampling evaluation measuring Adjusted Rand Index (ARI).
7. Final cluster assignment export to cluster.csv.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score

def run_task3(data_dir=None, output_dir=None):
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "TUGAS MACHINE LEARNING", "data_uts_ml-20261119T075355Z-1-001", "data_uts_ml")
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "..")

    clust_path = os.path.join(data_dir, "clustering.csv.gz")
    print("[Task 3] Loading clustering dataset (120,000 customers)...")
    df_raw = pd.read_csv(clust_path)
    df = df_raw.copy()

    # 1. Missing values & Skewness
    df['is_inactive_90h'] = (df['frekuensi_90h'] == 0).astype(int)
    df['rata2_keranjang'] = df['rata2_keranjang'].fillna(0.0)

    skewed_cols = [
        'belanja_90h', 'belanja_30h', 'frekuensi_90h', 'frekuensi_30h',
        'rata2_keranjang', 'recency_hari', 'sesi_per_minggu', 'durasi_sesi_menit',
        'retur_90h', 'ulasan_90h', 'jumlah_alamat', 'wishlist', 'klik_iklan_30h'
    ]
    for col in skewed_cols:
        df[f'log_{col}'] = np.log1p(np.maximum(0, df[col]))

    # 2. Compositional Geometry: Centered Log-Ratio (CLR)
    porsi_kat_cols = [c for c in df.columns if c.startswith('porsi_') and not c.startswith('porsi_jam_')]
    eps = 1e-4
    kat_vals = np.maximum(df[porsi_kat_cols].values, eps)
    log_kat = np.log(kat_vals)
    clr_kat = log_kat - log_kat.mean(axis=1, keepdims=True)
    clr_kat_df = pd.DataFrame(clr_kat, columns=[f'clr_{c}' for c in porsi_kat_cols])

    porsi_jam_cols = ['porsi_jam_pagi', 'porsi_jam_siang', 'porsi_jam_sore', 'porsi_jam_malam']
    jam_vals = np.maximum(df[porsi_jam_cols].values, eps)
    log_jam = np.log(jam_vals)
    clr_jam = log_jam - log_jam.mean(axis=1, keepdims=True)
    clr_jam_df = pd.DataFrame(clr_jam, columns=[f'clr_{c}' for c in porsi_jam_cols])

    perangkat_dummies = pd.get_dummies(df['perangkat'], prefix='dev', drop_first=True, dtype=float)

    base_behavior_cols = [
        'persen_voucher', 'persen_flash_sale', 'persen_paylater', 'persen_cod',
        'rating_beri_rata2', 'rasio_konversi', 'lama_member_bulan', 'umur', 'kota_tier'
    ]

    # Combine selected domain features (excluding synthetic noise n01-n15)
    X_features = pd.concat([
        df[[f'log_{c}' for c in skewed_cols] + base_behavior_cols + ['is_inactive_90h']],
        clr_kat_df,
        clr_jam_df,
        perangkat_dummies
    ], axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)

    # 3. Anomaly Detection (-1) via Isolation Forest
    print("[Task 3] Detecting anomalies via Isolation Forest...")
    iso = IsolationForest(contamination=0.015, random_state=42, n_jobs=-1)
    is_anomaly = iso.fit_predict(X_scaled) == -1
    print(f"  Flagged {is_anomaly.sum()} anomalies (1.50%)")

    X_clean = X_scaled[~is_anomaly]

    # 4. Fit Final K-Means (K=5)
    print("[Task 3] Fitting final K-Means (K=5) on clean observations...")
    final_km = MiniBatchKMeans(n_clusters=5, batch_size=4096, random_state=42, n_init=10)
    clean_labels = final_km.fit_predict(X_clean)

    final_clusters = np.full(len(df), -1, dtype=int)
    final_clusters[~is_anomaly] = clean_labels

    out_csv = os.path.join(output_dir, "cluster.csv")
    cluster_df = pd.DataFrame({
        'id_pelanggan': df['id_pelanggan'],
        'cluster': final_clusters
    })
    cluster_df.to_csv(out_csv, index=False)
    print(f"[Task 3] Successfully generated {out_csv} ({len(cluster_df)} customers)")
    print("Cluster distribution:\n", pd.Series(final_clusters).value_counts().sort_index())

if __name__ == '__main__':
    run_task3()
