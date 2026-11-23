"""
=============================================================================
UTS MACHINE LEARNING - MASTER REPRODUCTION SCRIPT (ONE-COMMAND REPRODUCTION)
Kelompok 16: Fransisco Sitepu (101032300004) & Ahmad Ghazali El-Faruqi HSB (101032330080)
=============================================================================
Menjalankan seluruh pipeline secara berurutan lalu membangun laporan.pdf:
- Task 1: Regresi (estimasi waktu pengiriman P50 & P90, kalibrasi conformal)  -> kode/hasil_task1/
- Task 2: Klasifikasi (deteksi fraud, kalibrasi & aturan biaya)              -> kode/hasil_task2/
- Task 3: Clustering (segmentasi pelanggan, K dipilih dengan stabilitas)     -> kode/hasil_task3/
- laporan.pdf dibangun dari angka-angka di atas (kode/laporan.py)

Pemakaian:  python kode/run_all.py [folder_data]
Folder data berisi regresi_*.csv.gz, klasifikasi_*.csv.gz, clustering.csv.gz, kamus_data.md.
=============================================================================
"""

import os
import sys
import time

from task1_regression import run_task1
from task2_classification import run_task2
from task3_clustering import run_task3
from laporan import build as build_laporan


def main():
    print("=" * 80)
    print("UTS MACHINE LEARNING - PIPELINE EXECUTION")
    print("Kelompok 16: Fransisco Sitepu (101032300004) & Ahmad Ghazali El-Faruqi HSB (101032330080)")
    print("=" * 80)

    start_total = time.time()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))

    candidate_data_dirs = ([os.path.abspath(sys.argv[1])] if len(sys.argv) > 1 else []) + [
        os.path.abspath(os.path.join(repo_root, "..", "TUGAS MACHINE LEARNING", "data_uts_ml-20261119T075355Z-1-001", "data_uts_ml")),
        os.path.abspath(os.path.join(repo_root, "data")),
        os.path.abspath(os.path.join(repo_root, "data_uts_ml")),
        os.path.abspath(os.path.join(repo_root, "..", "data_uts_ml")),
        os.path.abspath("data_uts_ml"),
    ]
    data_dir = next((p for p in candidate_data_dirs if os.path.exists(os.path.join(p, "regresi_train.csv.gz"))), None)
    if data_dir is None:
        raise FileNotFoundError(f"Dataset UTS tidak ditemukan di: {candidate_data_dirs}")

    print(f"Dataset directory resolved: {data_dir}")
    print(f"Output directory resolved:  {repo_root}\n")

    for name, fn in [("TASK 1: REGRESSION (DELIVERY TIME ESTIMATION)", run_task1),
                     ("TASK 2: CLASSIFICATION (FRAUD DETECTION)", run_task2),
                     ("TASK 3: CLUSTERING (CUSTOMER SEGMENTATION)", run_task3)]:
        t0 = time.time()
        print(f">>> RUNNING {name}...")
        fn(data_dir=data_dir, output_dir=repo_root)
        print(f">>> selesai dalam {time.time() - t0:.1f}s\n")

    build_laporan(os.path.join(repo_root, "laporan.pdf"))

    print("=" * 80)
    print(f"ALL PIPELINES EXECUTED SUCCESSFULLY in {time.time() - start_total:.1f}s!")
    print("Generated files: prediksi_regresi.csv, prediksi_klasifikasi.csv, cluster.csv, laporan.pdf, kode/hasil_task1-3/")
    print("=" * 80)


if __name__ == "__main__":
    main()
