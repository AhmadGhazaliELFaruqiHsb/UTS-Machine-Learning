"""
=============================================================================
UTS MACHINE LEARNING - TASK 3: CLUSTERING (CUSTOMER SEGMENTATION)
Kelompok 16: Fransisco Sitepu (101032300004) & Ahmad Ghazali El-Faruqi HSB (101032330080)
=============================================================================
Pipeline lengkap Task 3 (poin a-g soal UTS). Semua angka laporan Task 3 ditulis ke
kode/hasil_task3/ (ringkasan_task3.json + tabel CSV + grafik PNG).

a. Prapemrosesan: log1p untuk variabel miring (uang & hitungan), logit untuk persen (0-1),
   CLR (Aitchison) untuk porsi kategori & porsi jam, one-hot perangkat, imputasi
   rata2_keranjang (NaN hanya bila frekuensi_90h = 0), winsorizing 0,5/99,5 persen
   lalu standardisasi; pelanggan ekstrem ditangani sebagai anomali (poin g).
b. Seleksi fitur dengan bukti: (1) n01-n15 independen dari semua fitur domain (|rho| maks
   ~0,01) -> noise; (2) relevansi eta^2 tiap fitur terhadap klaster awal K=5..8, fitur dipakai
   bila eta^2 MINIMUM >= 0,01 (relevan konsisten, bukan membentuk klaster sendiri); (3) redundansi
   (belanja = frekuensi x keranjang; versi 30 hari vs 90 hari); (4) dampak ke silhouette &
   stabilitas.
c. Metode: K-Means, Gaussian Mixture, Agglomerative (Ward) dibandingkan.
d. Jumlah klaster: silhouette, Calinski-Harabasz, Davies-Bouldin, BIC (GMM), dan stabilitas
   bootstrap untuk K = 3..10. Aturan: K terbesar dengan ARI stabilitas >= 0,95.
e. Stabilitas: bootstrap subsampling, ARI antar-run pada holdout yang sama.
f. Profil & aksi: profil tiap segmen (median fitur asli), nama & aksi pemasaran ditentukan
   dari fitur pembeda terkuat (z-score rata-rata segmen).
g. Anomali: Isolation Forest, ambang ditentukan dari titik siku (knee) skor anomali -> -1.

Seed 42. Satu perintah: python kode/run_all.py (atau python kode/task3_clustering.py)
"""

import os
import json
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score

warnings.filterwarnings("ignore")
SEED = 42

LOG_COLS = ["recency_hari", "frekuensi_90h", "belanja_90h", "rata2_keranjang", "item_per_pesanan", "retur_90h",
            "ulasan_90h", "sesi_per_minggu", "durasi_sesi_menit", "rasio_konversi", "lama_member_bulan",
            "jumlah_alamat", "wishlist", "klik_iklan_30h", "belanja_30h", "frekuensi_30h"]
PERSEN_COLS = ["persen_voucher", "persen_flash_sale", "persen_paylater", "persen_cod"]
PLAIN_COLS = ["rating_beri_rata2", "umur", "kota_tier"]
REDUNDAN = {"log_belanja_90h": "= frekuensi_90h x rata2_keranjang",
            "log_belanja_30h": "versi 30 hari dari belanja_90h",
            "log_frekuensi_30h": "versi 30 hari dari frekuensi_90h"}

# kamus nama & aksi untuk fitur pembeda (dipakai otomatis saat profiling)
FRASA = {
    "logit_persen_voucher": ("Pemburu Voucher", "kampanye voucher bertarget & gamifikasi kumpul voucher"),
    "logit_persen_flash_sale": ("Pemburu Flash Sale", "notifikasi flash sale terjadwal sesuai jam aktif"),
    "logit_persen_paylater": ("Pengguna Paylater", "promo cicilan paylater 0% untuk menaikkan nilai keranjang"),
    "logit_persen_cod": ("Pembeli COD", "insentif pindah ke pembayaran digital (cashback e-wallet)"),
    "log_rata2_keranjang": ("Keranjang Besar", "bundling produk premium & garansi/proteksi tambahan"),
    "log_frekuensi_90h": ("Pembeli Rutin", "program loyalitas berjenjang (poin & gratis ongkir)"),
    "log_recency_hari": ("Lama Tidak Belanja", "kampanye win-back: voucher tanpa minimum belanja"),
    "log_sesi_per_minggu": ("Sering Buka Aplikasi", "push notification personal & rekomendasi real-time"),
    "log_rasio_konversi": ("Konversi Tinggi", "rekomendasi cross-sell di halaman checkout"),
    "umur": ("Usia Matang", "konten edukasi produk & layanan prioritas"),
    "log_lama_member_bulan": ("Member Lama", "apresiasi loyalitas (anniversary reward)"),
    "log_ulasan_90h": ("Rajin Ulas", "program reviewer/affiliate dengan insentif"),
    "log_item_per_pesanan": ("Banyak Item per Pesanan", "diskon grosir/multi-buy"),
    "clr_porsi_fashion": ("Fashion", "lookbook & rekomendasi tren fashion"),
    "clr_porsi_makanan_minuman": ("Kebutuhan Harian", "langganan belanja bulanan (subscribe & save)"),
    "clr_porsi_kecantikan": ("Kecantikan", "sampel produk & bundling perawatan"),
    "clr_porsi_elektronik": ("Elektronik", "trade-in & proteksi gadget"),
    "clr_porsi_ibu_bayi": ("Ibu & Bayi", "paket kebutuhan bayi berlangganan"),
    "clr_porsi_voucher_digital": ("Produk Digital", "promo pulsa/token & tagihan otomatis"),
    "clr_porsi_rumah_tangga": ("Rumah Tangga", "bundling perlengkapan rumah"),
    "clr_porsi_gadget_aksesoris": ("Gadget", "aksesoris pelengkap saat beli gadget"),
    "clr_porsi_perabot": ("Perabot", "promo perabot & jasa pasang"),
    "clr_porsi_jam_malam": ("Belanja Malam", "promo jam malam (night sale)"),
    "clr_porsi_jam_pagi": ("Belanja Pagi", "promo pagi (morning deals)"),
    "dev_ios": ("Pengguna iOS", "fitur & promo khusus aplikasi iOS"),
    "log_retur_90h": ("Sering Retur", "panduan ukuran & fitur coba virtual untuk menekan retur"),
}
FRASA_RENDAH = {
    "log_frekuensi_90h": ("Jarang Belanja", "kampanye aktivasi ulang dengan voucher pembelian berikutnya"),
    "log_rata2_keranjang": ("Keranjang Kecil", "ambang gratis ongkir untuk menaikkan nilai keranjang"),
    "log_recency_hari": ("Baru Saja Belanja", "rekomendasi produk pelengkap pasca-pembelian"),
    "logit_persen_voucher": ("Tidak Bergantung Voucher", "jaga margin: fokus kualitas & layanan, bukan diskon"),
    "log_sesi_per_minggu": ("Jarang Buka Aplikasi", "email/WA reminder berkala"),
    "umur": ("Usia Muda", "konten media sosial & kolaborasi influencer"),
    "log_lama_member_bulan": ("Member Baru", "onboarding: voucher pembelian pertama & panduan aplikasi"),
    "log_rasio_konversi": ("Konversi Rendah", "pengingat keranjang & rekomendasi produk yang dilihat"),
}


def log(msg):
    print(f"[Task 3] {msg}", flush=True)


def resolve_data_dir(data_dir):
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, ".."))
    cands = [data_dir] if data_dir else []
    cands += [os.path.join(root, "..", "TUGAS MACHINE LEARNING", "data_uts_ml-20261119T075355Z-1-001", "data_uts_ml"),
              os.path.join(root, "data"), os.path.join(root, "data_uts_ml"), os.path.join(root, "..", "data_uts_ml")]
    for p in cands:
        if p and os.path.exists(os.path.join(p, "clustering.csv.gz")):
            return os.path.abspath(p)
    raise FileNotFoundError(f"clustering.csv.gz tidak ditemukan di: {cands}")


def transform(df, noise_cols):
    """Transformasi semua kandidat fitur (sebelum seleksi)."""
    X = pd.DataFrame(index=df.index)
    keranjang = df["rata2_keranjang"].fillna(df["rata2_keranjang"].median())
    for c in LOG_COLS:
        v = keranjang if c == "rata2_keranjang" else df[c]
        X[f"log_{c}"] = np.log1p(v.clip(lower=0))
    for c in PERSEN_COLS:
        p = df[c].clip(1e-3, 1 - 1e-3)
        X[f"logit_{c}"] = np.log(p / (1 - p))
    for c in PLAIN_COLS:
        X[c] = df[c]
    por = [c for c in df.columns if c.startswith("porsi_") and not c.startswith("porsi_jam_")]
    jam = [c for c in df.columns if c.startswith("porsi_jam_")]
    for group in (por, jam):
        M = np.maximum(df[group].to_numpy(float), 1e-3)          # penggantian nol multiplikatif
        M = M / M.sum(1, keepdims=True)
        L = np.log(M)
        C = L - L.mean(1, keepdims=True)
        for i, c in enumerate(group):
            X[f"clr_{c}"] = C[:, i]
    for v in ("ios", "web"):
        X[f"dev_{v}"] = (df["perangkat"] == v).astype(float)
    for c in noise_cols:
        X[c] = df[c]
    return X


def winsor_scale(X):
    lo, hi = X.quantile(0.005), X.quantile(0.995)
    Xw = X.clip(lo, hi, axis=1)
    return StandardScaler().fit_transform(Xw)


def stability(Z, k, n_runs=5, sub=20000, hold=None, method="kmeans"):
    """ARI antar-run: model dilatih pada subsample bootstrap berbeda, dinilai pada holdout sama."""
    labs = []
    for b in range(n_runs):
        idx = np.random.RandomState(1000 + b).choice(len(Z), sub, replace=True)
        if method == "kmeans":
            m = KMeans(k, n_init=3, random_state=b).fit(Z[idx]); labs.append(m.predict(Z[hold]))
        else:
            m = GaussianMixture(k, covariance_type="diag", random_state=b, n_init=1).fit(Z[idx]); labs.append(m.predict(Z[hold]))
    aris = [adjusted_rand_score(labs[i], labs[j]) for i in range(n_runs) for j in range(i + 1, n_runs)]
    return float(np.mean(aris)), float(np.std(aris)), float(np.min(aris))


def knee(scores):
    """Titik siku pada skor anomali terurut (metode jarak maksimum ke garis)."""
    s = np.sort(scores)[::-1]
    top = s[: int(0.10 * len(s))]                      # cari siku di 10% skor tertinggi
    x = np.arange(len(top)); x1, y1, x2, y2 = 0, top[0], len(top) - 1, top[-1]
    d = np.abs((y2 - y1) * x - (x2 - x1) * top + x2 * y1 - y2 * x1) / np.hypot(y2 - y1, x2 - x1)
    i = int(np.argmax(d))
    return float(top[i]), i + 1


def run_task3(data_dir=None, output_dir=None):
    t0 = time.time()
    data_dir = resolve_data_dir(data_dir)
    here = os.path.dirname(os.path.abspath(__file__))
    output_dir = output_dir or os.path.abspath(os.path.join(here, ".."))
    res_dir = os.path.join(here, "hasil_task3")
    os.makedirs(res_dir, exist_ok=True)
    R = {}

    log(f"Memuat data dari {data_dir}")
    df = pd.read_csv(os.path.join(data_dir, "clustering.csv.gz"))
    noise_cols = [c for c in df.columns if c.startswith("n") and c[1:].isdigit()]
    R["n_pelanggan"], R["n_kolom"] = int(len(df)), int(df.shape[1])

    # ======================================================================
    # (a) AUDIT & PRAPEMROSESAN
    # ======================================================================
    log("(a) Audit & prapemrosesan")
    por = [c for c in df.columns if c.startswith("porsi_") and not c.startswith("porsi_jam_")]
    jam = [c for c in df.columns if c.startswith("porsi_jam_")]
    inactive = df["frekuensi_90h"] == 0
    R["audit"] = dict(
        n_porsi_kategori=len(por), n_porsi_jam=len(jam),
        jumlah_porsi_kategori_min_max=[round(float(df[por].sum(axis=1).min()), 4), round(float(df[por].sum(axis=1).max()), 4)],
        pct_nol_porsi_kategori=round(100 * float((df[por] == 0).to_numpy().mean()), 2),
        keranjang_nan=int(df["rata2_keranjang"].isna().sum()),
        keranjang_nan_saat_frekuensi_nol=int((df["rata2_keranjang"].isna() & inactive).sum()),
        frekuensi_nol_tapi_recency_lt_90=int((inactive & (df["recency_hari"] < 90)).sum()),
        skew_sebelum={c: round(float(df[c].skew()), 2) for c in ["belanja_90h", "recency_hari", "sesi_per_minggu", "rata2_keranjang", "klik_iklan_30h"]},
        skew_sesudah_log={c: round(float(np.log1p(df[c].fillna(df[c].median())).skew()), 2) for c in ["belanja_90h", "recency_hari", "sesi_per_minggu", "rata2_keranjang", "klik_iklan_30h"]},
        perangkat=df["perangkat"].value_counts().to_dict(),
    )

    # ======================================================================
    # (b) SELEKSI FITUR DENGAN BUKTI
    # ======================================================================
    log("(b) Seleksi fitur: bukti noise, relevansi eta^2, redundansi")
    Xall = transform(df, noise_cols)
    domain = [c for c in Xall.columns if c not in noise_cols]
    rho = Xall[domain + noise_cols].sample(30000, random_state=SEED).corr(method="spearman")
    noise_tab = []
    for c in noise_cols:
        x = df[c]
        u = (x - x.min()) / (x.max() - x.min())
        noise_tab.append(dict(fitur=c, maks_abs_rho_domain=round(float(rho.loc[c, domain].abs().max()), 4),
                              maks_abs_rho_noise_lain=round(float(rho.loc[c, [n for n in noise_cols if n != c]].abs().max()), 4),
                              ks_uniform_p=round(float(stats.kstest(u, "uniform").pvalue), 4),
                              mean=round(float(x.mean()), 3), std=round(float(x.std()), 3)))
    pd.DataFrame(noise_tab).to_csv(os.path.join(res_dir, "b_bukti_noise.csv"), index=False)
    R["noise"] = dict(maks_rho_domain=max(r["maks_abs_rho_domain"] for r in noise_tab),
                      seragam_p_gt_005=[r["fitur"] for r in noise_tab if r["ks_uniform_p"] > 0.05])
    dom_rho = rho.loc[domain, domain].abs().where(~np.eye(len(domain), dtype=bool))
    R["rho_domain_median_maks"] = round(float(dom_rho.max().median()), 3)
    R["redundansi_rho"] = {
        "log_belanja_90h~log_rata2_keranjang+log_frekuensi_90h": round(float(np.corrcoef(
            Xall["log_belanja_90h"], np.log(df["rata2_keranjang"].fillna(0) * df["frekuensi_90h"] + 1))[0, 1]), 3),
        "log_belanja_30h~log_belanja_90h": round(float(rho.loc["log_belanja_30h", "log_belanja_90h"]), 3),
        "log_frekuensi_30h~log_frekuensi_90h": round(float(rho.loc["log_frekuensi_30h", "log_frekuensi_90h"]), 3),
    }

    cand = [c for c in domain if c not in REDUNDAN]
    Zc = winsor_scale(Xall[cand])
    rng = np.random.RandomState(SEED)
    samp = rng.choice(len(Zc), 20000, replace=False)
    eta = pd.DataFrame(index=cand)
    for k in (5, 6, 7, 8):
        lab = KMeans(k, n_init=3, random_state=SEED).fit_predict(Zc[samp])
        Zs = Zc[samp]
        tot = ((Zs - Zs.mean(0)) ** 2).sum(0)
        btw = sum(((Zs[lab == l].mean(0) - Zs.mean(0)) ** 2) * (lab == l).sum() for l in range(k))
        eta[f"K{k}"] = btw / tot
    kcols = [c for c in eta.columns]
    eta["eta2_rata2"] = eta[kcols].mean(1)
    eta["eta2_min"] = eta[kcols].min(1)
    eta = eta.sort_values("eta2_rata2", ascending=False).round(4)
    eta.to_csv(os.path.join(res_dir, "b_relevansi_eta2.csv"))
    ETA_MIN = 0.01
    # fitur harus relevan secara KONSISTEN di semua K (eta^2 minimum), bukan hanya di satu K
    selected = [c for c in eta.index if eta.loc[c, "eta2_min"] >= ETA_MIN]
    dropped = [c for c in eta.index if eta.loc[c, "eta2_min"] < ETA_MIN]
    R["seleksi"] = dict(eta_min=ETA_MIN, n_kandidat=len(cand), n_terpilih=len(selected), terpilih=selected,
                        dibuang_eta=dropped, dibuang_redundan=REDUNDAN, dibuang_noise=noise_cols)

    # ======================================================================
    # (g) ANOMALI - dilakukan sebelum pemilihan K agar klaster tidak terdistorsi
    # ======================================================================
    log("(g) Deteksi anomali (Isolation Forest + knee)")
    Z = winsor_scale(Xall[selected])
    # skor anomali dihitung pada fitur asli (tanpa winsor) agar ekstrem tetap terlihat
    Zraw = StandardScaler().fit_transform(Xall[selected])
    iso = IsolationForest(n_estimators=300, max_samples=4096, random_state=SEED, n_jobs=-1).fit(Zraw)
    score = -iso.score_samples(Zraw)
    thr, n_anom = knee(score)
    is_anom = score >= thr
    R["anomali"] = dict(ambang_skor=round(thr, 4), n=int(is_anom.sum()), pct=round(100 * float(is_anom.mean()), 2))
    clean_idx = np.where(~is_anom)[0]
    Zcl = Z[clean_idx]

    # ======================================================================
    # (d) JUMLAH KLASTER
    # ======================================================================
    log("(d) Memilih K: silhouette, CH, DB, BIC, stabilitas (K=3..10)")
    rs = np.random.RandomState(SEED)
    s_idx = rs.choice(len(Zcl), 20000, replace=False)
    hold = rs.choice(len(Zcl), 10000, replace=False)
    krows = []
    for k in range(3, 11):
        km = KMeans(k, n_init=4, random_state=SEED).fit(Zcl[s_idx])
        lab = km.labels_
        gm = GaussianMixture(k, covariance_type="diag", random_state=SEED, n_init=2).fit(Zcl[s_idx])
        st_m, st_s, st_min = stability(Zcl, k, n_runs=5, hold=hold)
        krows.append(dict(K=k, silhouette=round(float(silhouette_score(Zcl[s_idx], lab, sample_size=8000, random_state=SEED)), 4),
                          calinski_harabasz=round(float(calinski_harabasz_score(Zcl[s_idx], lab)), 1),
                          davies_bouldin=round(float(davies_bouldin_score(Zcl[s_idx], lab)), 3),
                          bic_gmm=round(float(gm.bic(Zcl[s_idx])), 0),
                          stabilitas_ari=round(st_m, 3), stabilitas_ari_min=round(st_min, 3)))
        log(f"   K={k}: stabilitas {st_m:.3f}")
    ktab = pd.DataFrame(krows)
    ktab.to_csv(os.path.join(res_dir, "d_pemilihan_k.csv"), index=False)
    STAB_MIN = 0.95
    K = int(ktab.loc[ktab.stabilitas_ari >= STAB_MIN, "K"].max())
    R["pemilihan_k"] = dict(aturan=f"K terbesar dengan ARI stabilitas >= {STAB_MIN}", K=K, tabel=krows)

    # Dampak seleksi fitur (bukti b): semua fitur+noise vs terpilih, pada K terpilih
    imp_rows = []
    for name, cols in [("Semua kolom (domain + noise n01-n15)", domain + noise_cols),
                       ("Domain tanpa noise (belum diseleksi)", domain),
                       ("Fitur terpilih (dipakai)", selected)]:
        Zx = winsor_scale(Xall[cols])[clean_idx]
        lab = KMeans(K, n_init=4, random_state=SEED).fit_predict(Zx[s_idx])
        st = stability(Zx, K, n_runs=4, hold=hold)
        imp_rows.append(dict(set_fitur=name, n_fitur=len(cols),
                             silhouette=round(float(silhouette_score(Zx[s_idx], lab, sample_size=8000, random_state=SEED)), 4),
                             stabilitas_ari=round(st[0], 3)))
    pd.DataFrame(imp_rows).to_csv(os.path.join(res_dir, "b_dampak_seleksi.csv"), index=False)
    R["dampak_seleksi"] = imp_rows

    # ======================================================================
    # (c) PERBANDINGAN METODE pada K terpilih
    # ======================================================================
    log(f"(c) Perbandingan metode pada K={K}")
    ev = rs.choice(len(Zcl), 10000, replace=False)
    km_full = KMeans(K, n_init=10, random_state=SEED).fit(Zcl)
    lab_km = km_full.labels_
    gm_samp = rs.choice(len(Zcl), 40000, replace=False)
    gm = GaussianMixture(K, covariance_type="diag", random_state=SEED, n_init=3).fit(Zcl[gm_samp])
    lab_gm = gm.predict(Zcl)
    ag_samp = rs.choice(len(Zcl), 15000, replace=False)
    ag = AgglomerativeClustering(n_clusters=K, linkage="ward").fit(Zcl[ag_samp])
    cent = np.vstack([Zcl[ag_samp][ag.labels_ == l].mean(0) for l in range(K)])
    lab_ag = np.argmin(((Zcl[:, None, :] - cent[None]) ** 2).sum(-1), axis=1)
    meth = {"K-Means": lab_km, "Gaussian Mixture (diag)": lab_gm, "Agglomerative Ward": lab_ag}
    mrows = []
    for name, lab in meth.items():
        st = stability(Zcl, K, n_runs=4, hold=hold, method="gmm") if name.startswith("Gaussian") else (
            stability(Zcl, K, n_runs=4, hold=hold) if name == "K-Means" else (None, None, None))
        mrows.append(dict(metode=name, silhouette=round(float(silhouette_score(Zcl[ev], lab[ev])), 4),
                          davies_bouldin=round(float(davies_bouldin_score(Zcl[ev], lab[ev])), 3),
                          calinski_harabasz=round(float(calinski_harabasz_score(Zcl[ev], lab[ev])), 1),
                          stabilitas_ari=None if st[0] is None else round(st[0], 3),
                          ukuran_terkecil_pct=round(100 * np.bincount(lab, minlength=K).min() / len(lab), 2)))
    # stabilitas agglomerative: 4 subsample berbeda
    ag_labs = []
    for b in range(4):
        sb = np.random.RandomState(2000 + b).choice(len(Zcl), 12000, replace=False)
        a = AgglomerativeClustering(n_clusters=K, linkage="ward").fit(Zcl[sb])
        c_ = np.vstack([Zcl[sb][a.labels_ == l].mean(0) for l in range(K)])
        ag_labs.append(np.argmin(((Zcl[hold][:, None, :] - c_[None]) ** 2).sum(-1), axis=1))
    mrows[2]["stabilitas_ari"] = round(float(np.mean([adjusted_rand_score(ag_labs[i], ag_labs[j]) for i in range(4) for j in range(i + 1, 4)])), 3)
    agree = {f"{a} vs {b}": round(float(adjusted_rand_score(meth[a], meth[b])), 3)
             for i, a in enumerate(meth) for b in list(meth)[i + 1:]}
    pd.DataFrame(mrows).to_csv(os.path.join(res_dir, "c_perbandingan_metode.csv"), index=False)
    R["metode"] = dict(tabel=mrows, kesepakatan_ari=agree)
    best_m = max(mrows, key=lambda r: (r["stabilitas_ari"], r["silhouette"]))["metode"]
    R["metode_terpilih"] = best_m
    final_lab = meth[best_m]

    # ======================================================================
    # (e) STABILITAS model final (10 bootstrap)
    # ======================================================================
    log("(e) Uji stabilitas bootstrap model final")
    meth_key = "gmm" if best_m.startswith("Gaussian") else "kmeans"
    st_m, st_s, st_min = stability(Zcl, K, n_runs=10, sub=30000, hold=hold, method=meth_key)
    R["stabilitas_final"] = dict(n_bootstrap=10, ukuran_subsample=30000, ukuran_holdout=len(hold),
                                 ari_mean=round(st_m, 3), ari_std=round(st_s, 3), ari_min=round(st_min, 3))

    # ======================================================================
    # (f) PROFIL & AKSI
    # ======================================================================
    log("(f) Profil segmen")
    clusters = np.full(len(df), -1, dtype=int)
    # urutkan nomor klaster berdasarkan ukuran (0 = terbesar) agar deterministik
    order = np.argsort(-np.bincount(final_lab, minlength=K))
    remap = {old: new for new, old in enumerate(order)}
    clusters[clean_idx] = np.vectorize(remap.get)(final_lab)
    prof_cols = ["recency_hari", "frekuensi_90h", "belanja_90h", "rata2_keranjang", "persen_voucher", "persen_flash_sale",
                 "persen_paylater", "persen_cod", "sesi_per_minggu", "rasio_konversi", "umur", "lama_member_bulan",
                 "ulasan_90h", "porsi_fashion", "porsi_makanan_minuman", "porsi_elektronik", "porsi_kecantikan",
                 "porsi_ibu_bayi", "porsi_jam_malam"]
    d = df.assign(cluster=clusters)
    prof = d.groupby("cluster")[prof_cols].median().round(3)
    prof.insert(0, "n", d.groupby("cluster").size())
    prof.insert(1, "pct", (100 * prof["n"] / len(d)).round(2))
    Zdf = pd.DataFrame(Z, columns=selected).assign(cluster=clusters)
    zmean = Zdf[Zdf.cluster >= 0].groupby("cluster")[selected].mean()
    names, actions, pembeda = {}, {}, {}
    for c in zmean.index:
        s = zmean.loc[c].sort_values()
        cand_f = [(abs(v), FRASA[f]) for f, v in s.items() if v > 0.4 and f in FRASA]
        cand_f += [(abs(v), FRASA_RENDAH[f]) for f, v in s.items() if v < -0.4 and f in FRASA_RENDAH]
        cand_f.sort(key=lambda t: -t[0])
        top = [t[1] for t in cand_f[:2]]
        names[c] = " / ".join(t[0] for t in top) if top else "Pelanggan Rata-rata"
        actions[c] = top[0][1] if top else "newsletter umum & rekomendasi personal"
        pembeda[c] = {f: round(float(s[f]), 2) for f in list(s.index[::-1][:3]) + list(s.index[:2])}
    names[-1], actions[-1] = "Anomali (tidak cocok segmen mana pun)", "tinjau manual: kemungkinan reseller/bot/akun bermasalah"
    prof["nama"] = [names[c] for c in prof.index]
    prof["aksi"] = [actions[c] for c in prof.index]
    prof.to_csv(os.path.join(res_dir, "f_profil_segmen.csv"))
    zmean.round(3).to_csv(os.path.join(res_dir, "f_zscore_segmen.csv"))
    R["profil"] = prof.reset_index().to_dict(orient="records")
    R["pembeda"] = {str(k): v for k, v in pembeda.items()}
    an = d[d.cluster == -1]
    R["anomali"]["profil_median_vs_normal"] = {c: [round(float(an[c].median()), 2), round(float(d[d.cluster >= 0][c].median()), 2)]
                                              for c in ["belanja_90h", "frekuensi_90h", "sesi_per_minggu", "recency_hari", "rasio_konversi", "lama_member_bulan", "item_per_pesanan"]}
    R["anomali"]["pct_di_atas_p99_minimal_satu_fitur"] = round(100 * float((an[LOG_COLS[:14]] > d[LOG_COLS[:14]].quantile(0.99)).any(axis=1).mean()), 1)

    out = pd.DataFrame({"id_pelanggan": df["id_pelanggan"], "cluster": clusters})
    assert len(out) == len(df) and out.id_pelanggan.is_unique
    out.to_csv(os.path.join(output_dir, "cluster.csv"), index=False)
    R["distribusi"] = out["cluster"].value_counts().sort_index().to_dict()
    log(f"Tersimpan: cluster.csv ({len(out)} baris), K={K}, anomali={int(is_anom.sum())}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        ax[0].plot(ktab.K, ktab.stabilitas_ari, marker="o", label="stabilitas ARI")
        ax[0].plot(ktab.K, ktab.silhouette * 10, marker="s", label="silhouette x10")
        ax[0].axhline(STAB_MIN, color="r", ls="--", lw=1); ax[0].axvline(K, color="k", ls=":", lw=1)
        ax[0].set_title("(d) Pemilihan K"); ax[0].set_xlabel("K"); ax[0].legend()
        ax2 = ax[1]; ax2.plot(ktab.K, ktab.bic_gmm, marker="o", color="#8172B2"); ax2.set_title("(d) BIC GMM (rendah = baik)"); ax2.set_xlabel("K")
        im = ax[2].imshow(zmean.T.values, aspect="auto", cmap="RdBu_r", vmin=-1.5, vmax=1.5)
        ax[2].set_yticks(range(len(selected))); ax[2].set_yticklabels(selected, fontsize=5)
        ax[2].set_xticks(range(K)); ax[2].set_title("(f) z-score rata-rata per segmen"); fig.colorbar(im, ax=ax[2])
        plt.tight_layout(); plt.savefig(os.path.join(res_dir, "fig_task3_k_profil.png"), dpi=130); plt.close()
    except Exception as e:
        log(f"Grafik dilewati: {e}")

    import sklearn
    R["library"] = dict(sklearn=sklearn.__version__, pandas=pd.__version__, numpy=np.__version__, seed=SEED)
    R["durasi_detik"] = round(time.time() - t0, 1)
    with open(os.path.join(res_dir, "ringkasan_task3.json"), "w") as f:
        json.dump(R, f, indent=2, default=str, ensure_ascii=False)
    log(f"Selesai dalam {R['durasi_detik']} detik")
    return R


if __name__ == "__main__":
    import sys
    run_task3(sys.argv[1] if len(sys.argv) > 1 else None)
