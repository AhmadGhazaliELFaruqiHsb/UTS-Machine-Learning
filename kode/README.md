# Panduan Reproduksi Kode - UTS Machine Learning

**Kelompok 16:**
- **Fransisco Sitepu** (NIM: 101032300004) - Task 1 & Task 3
- **Ahmad Ghazali El-Faruqi HSB** (NIM: 101032330080) - Task 2, perbaikan Task 1 & Task 3, generator laporan

---

## 1. Persyaratan Lingkungan

- **Python** 3.10+
- **Library** (versi yang dipakai untuk semua angka di laporan: LightGBM 4.7.0, scikit-learn 1.9.1, pandas 3.0.5, numpy 2.5.3):
  ```bash
  pip install pandas numpy scipy scikit-learn lightgbm matplotlib fpdf2
  ```
- **Data**: folder `data_uts_ml` (berisi `regresi_*.csv.gz`, `klasifikasi_*.csv.gz`, `clustering.csv.gz`, `kamus_data.md`). Folder dicari otomatis di `../TUGAS MACHINE LEARNING/data_uts_ml-.../data_uts_ml`, `data/`, `data_uts_ml/`, atau `../data_uts_ml`; bisa juga diberikan sebagai argumen.

---

## 2. Satu Perintah Reproduksi

Sesuai instruksi ujian: *"One command must reproduce every number in the report."*

```bash
python kode/run_all.py                 # atau: python kode/run_all.py /path/ke/data_uts_ml
```

Perintah ini menjalankan berurutan:
1. **Task 1** (`task1_regression.py`, ±13–21 menit pada 2 CPU): audit data (kebocoran `jumlah_scan`, label bersatuan menit, sentinel `jarak_km`, `berat_kg` hilang tidak acak, `f30` proksi waktu), validasi waktu Jan–Jun → Jul–Agu vs random 5-fold CV, tuning di dalam lipatan latih, baseline + Ridge log-target + LightGBM quantile, uji kategori baru, kalibrasi conformal (kalibrasi Juli, evaluasi Agustus), galat per layanan/pulau/kurir, SHAP & partial dependence → `prediksi_regresi.csv`, `kode/hasil_task1/`.
2. **Task 2** (`task2_classification.py`, ±7 menit): audit label lag & kebocoran `flag_review_manual`, validasi waktu, fitur perilaku tanpa informasi masa depan, imbalance, pergeseran pola, kalibrasi out-of-time, aturan biaya, analisis galat → `prediksi_klasifikasi.csv`, `kode/hasil_task2/`.
3. **Task 3** (`task3_clustering.py`, ±2–4 menit): CLR/logit/log1p, bukti noise `n01–n15`, seleksi fitur dengan η², anomali Isolation Forest (ambang knee), pemilihan K (silhouette, CH, DB, BIC, stabilitas bootstrap), perbandingan K-Means/GMM/Agglomerative, profil & aksi → `cluster.csv`, `kode/hasil_task3/`.
4. **Laporan** (`laporan.py` + `laporan_task2.py`): membangun ulang seluruh `laporan.pdf` dari `kode/hasil_task1-3/`. Tidak ada angka yang diketik manual.

---

## 3. Menjalankan per Task

```bash
python kode/task1_regression.py   [folder_data]
python kode/task2_classification.py [folder_data]
python kode/task3_clustering.py   [folder_data]
python kode/laporan.py            # bangun laporan.pdf dari hasil yang sudah ada
```

| Task | Output prediksi | Semua angka laporan |
| --- | --- | --- |
| 1 | `prediksi_regresi.csv` (`id_paket`, `p50`, `p90`) | `kode/hasil_task1/ringkasan_task1.json` + `*.csv`, grafik `fig_task1_*.png` |
| 2 | `prediksi_klasifikasi.csv` (`id_transaksi`, `prob`, `tandai`) | `kode/hasil_task2/ringkasan_task2.json` + `*.csv`, grafik `fig_task2_*.png` |
| 3 | `cluster.csv` (`id_pelanggan`, `cluster`) | `kode/hasil_task3/ringkasan_task3.json` + `*.csv`, grafik `fig_task3_*.png` |

---

## 4. Seed & Determinisme

- `random_state = 42` (atau turunan tetapnya) di semua model, split, subsampling, dan bootstrap.
- Hasil deterministik: menjalankan ulang menghasilkan file prediksi dan angka yang sama (diverifikasi dengan md5).
- Validasi berbasis waktu menjamin tidak ada *look-ahead bias*.
