"""
=============================================================================
UTS MACHINE LEARNING - TASK 1: REGRESSION (DELIVERY TIME ESTIMATION)
Kelompok 16: Fransisco Sitepu (101032300004) & Ahmad Ghazali El-Faruqi HSB (101032330080)
=============================================================================
Pipeline lengkap Task 1 (poin a-f soal UTS). Semua angka laporan Task 1 ditulis ke
kode/hasil_task1/ (ringkasan_task1.json + tabel CSV + grafik PNG).

a. Audit data: kebocoran `jumlah_scan` (dicatat setelah paket sampai), label tercatat
   dalam MENIT (durasi > 1000 jam, dibagi 60), sentinel `jarak_km` (0 dan 9999),
   `berat_kg` hilang tidak acak (dominan kargo), `f30` = proksi waktu. Dampak diukur.
b. Validasi: latih Jan-Jun -> uji Jul-Agu (horizon 2 bulan, sama dengan data uji asli);
   tuning hanya di dalam lipatan latih (Jan-Apr -> Mei-Jun). Dibandingkan random 5-fold CV.
c. Model: baseline kuantil empiris per (layanan, pulau_tujuan); model lanjutan
   (1) LightGBM quantile loss, (2) Ridge dengan target log + kuantil residual.
d. Kategori: `kota_tujuan` (405 kategori) -> kategori native LightGBM + regularisasi
   (cat_smooth, min_data_per_group) dan target encoding ber-smoothing untuk Ridge;
   kategori yang hanya ada di uji (mitra_kurir K13) -> missing, diuji dengan simulasi
   "kurir tak dikenal" (satu kurir dikeluarkan dari data latih).
e. Ketidakpastian: P50 & P90, pinball loss & cakupan P90, kalibrasi conformal (CQR)
   dengan data kalibrasi (Juli) TERPISAH dari data evaluasi (Agustus).
f. Analisis: galat per layanan, pulau_tujuan, mitra_kurir; SHAP (TreeSHAP bawaan LightGBM)
   dan partial dependence.

Seed 42. Satu perintah: python kode/run_all.py (atau python kode/task1_regression.py)
"""

import os
import json
import time
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

SEED = 42
CAT_COLS = ["layanan", "mitra_kurir", "hub_asal", "kota_asal", "pulau_tujuan", "kota_tujuan"]
LEAK_COLS = ["jumlah_scan"]       # dicatat sepanjang perjalanan paket -> tidak ada saat pickup
TIME_PROXY_COLS = ["f30"]         # naik linear tiap bulan, tanpa sinyal
UNIT_THRESHOLD = 1000.0           # durasi > 1000 jam = tercatat dalam menit


def log(msg):
    print(f"[Task 1] {msg}", flush=True)


def pinball(y, q, alpha):
    d = np.asarray(y) - np.asarray(q)
    return float(np.mean(np.maximum(alpha * d, (alpha - 1) * d)))


def coverage(y, q90):
    return float(np.mean(np.asarray(y) <= np.asarray(q90)))


def evaluate(y, p50, p90, name, **extra):
    out = dict(model=name, mae_p50=round(float(np.mean(np.abs(y - p50))), 3),
               pinball_50=round(pinball(y, p50, 0.5), 3),
               pinball_90=round(pinball(y, p90, 0.9), 3) if p90 is not None else None,
               coverage_90=round(100 * coverage(y, p90), 2) if p90 is not None else None)
    out.update(extra)
    return out


def resolve_data_dir(data_dir):
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, ".."))
    cands = [data_dir] if data_dir else []
    cands += [os.path.join(root, "..", "TUGAS MACHINE LEARNING", "data_uts_ml-20261119T075355Z-1-001", "data_uts_ml"),
              os.path.join(root, "data"), os.path.join(root, "data_uts_ml"), os.path.join(root, "..", "data_uts_ml")]
    for p in cands:
        if p and os.path.exists(os.path.join(p, "regresi_train.csv.gz")):
            return os.path.abspath(p)
    raise FileNotFoundError(f"regresi_train.csv.gz tidak ditemukan di: {cands}")


# ----------------------------------------------------------------------------
# Pembersihan & fitur (semua tersedia saat pickup)
# ----------------------------------------------------------------------------
def clean(df, is_train):
    df = df.copy()
    df["tgl_pickup"] = pd.to_datetime(df["tgl_pickup"])
    if is_train:
        unit = df["durasi_jam"] > UNIT_THRESHOLD
        df["durasi_jam"] = np.where(unit, df["durasi_jam"] / 60.0, df["durasi_jam"])
    df["jarak_km"] = df["jarak_km"].where((df["jarak_km"] > 0) & (df["jarak_km"] < 9999))
    df["berat_hilang"] = df["berat_kg"].isna().astype(int)
    df["umur_akun_penjual_hari"] = df["umur_akun_penjual_hari"].where(df["umur_akun_penjual_hari"] <= 20000)
    df["hari_minggu"] = df["tgl_pickup"].dt.dayofweek
    df["log_antrean"] = np.log1p(df["antrean_hub_asal"])
    df["berat_volumetrik"] = df["volume_liter"] / 6.0      # aturan kurir: 1 kg ~ 6 liter
    df["berat_tagih"] = np.fmax(df["berat_kg"], df["berat_volumetrik"])
    return df


def to_lgb(df, cols, levels):
    X = df[cols].copy()
    for c in CAT_COLS:
        if c in X:
            # kategori di luar data latih (mis. K13) -> NaN (diperlakukan sebagai missing)
            X[c] = pd.Categorical(X[c], categories=levels[c])
    return X


def target_encode(train_col, y, apply_col, k=50):
    """Target encoding (mean log-durasi) ber-smoothing; kategori baru -> rata-rata global."""
    g = pd.DataFrame({"c": train_col.values, "y": y}).groupby("c")["y"].agg(["mean", "count"])
    prior = float(np.mean(y))
    enc = (g["mean"] * g["count"] + prior * k) / (g["count"] + k)
    return apply_col.map(enc).astype(float).fillna(prior).to_numpy()


class RidgeLog:
    """Model linear dengan transformasi target log1p + kuantil residual untuk P90."""

    def __init__(self, num_cols, alpha=10.0):
        self.num_cols, self.alpha = num_cols, alpha

    def _design(self, df, fit=False):
        Z = df[self.num_cols].copy()
        if fit:
            self.med = Z.median()
        Z = Z.fillna(self.med)
        for c in ["jarak_km", "berat_kg", "volume_liter", "nilai_barang", "antrean_hub_asal",
                  "volume_penjual_30h", "umur_akun_penjual_hari", "berat_tagih", "berat_volumetrik"]:
            if c in Z:
                Z[c] = np.log1p(Z[c].clip(lower=0))
        oh = pd.get_dummies(df[["layanan", "pulau_tujuan", "hub_asal", "mitra_kurir"]].astype(str))
        if fit:
            self.oh_cols = oh.columns
        oh = oh.reindex(columns=self.oh_cols, fill_value=0)
        te = np.column_stack([target_encode(self.tr_cat[c], self.ylog, df[c]) for c in ["kota_tujuan", "kota_asal"]])
        A = np.hstack([Z.to_numpy(float), te, oh.to_numpy(float)])
        if fit:
            self.sc = StandardScaler().fit(A)
        return np.clip(self.sc.transform(A), -8, 8)

    def fit(self, df, y):
        self.ylog = np.log1p(y)
        self.tr_cat = {c: df[c] for c in ["kota_tujuan", "kota_asal"]}
        A = self._design(df, fit=True)
        self.m = Ridge(alpha=self.alpha).fit(A, self.ylog)
        res = self.ylog - self.m.predict(A)
        self.q50_res, self.q90_res = np.quantile(res, 0.5), np.quantile(res, 0.9)
        return self

    def predict(self, df):
        mu = self.m.predict(self._design(df))
        return np.expm1(mu + self.q50_res), np.expm1(mu + self.q90_res)


def run_task1(data_dir=None, output_dir=None):
    t0 = time.time()
    data_dir = resolve_data_dir(data_dir)
    here = os.path.dirname(os.path.abspath(__file__))
    output_dir = output_dir or os.path.abspath(os.path.join(here, ".."))
    res_dir = os.path.join(here, "hasil_task1")
    os.makedirs(res_dir, exist_ok=True)
    np.random.seed(SEED)
    R = {}

    log(f"Memuat data dari {data_dir}")
    raw_tr = pd.read_csv(os.path.join(data_dir, "regresi_train.csv.gz"))
    raw_te = pd.read_csv(os.path.join(data_dir, "regresi_test.csv.gz"))
    R["n_train"], R["n_test"] = len(raw_tr), len(raw_te)

    # ======================================================================
    # (a) AUDIT DATA
    # ======================================================================
    log("(a) Audit data")
    y_raw = raw_tr["durasi_jam"]
    big = raw_tr[y_raw > UNIT_THRESHOLD]
    normal = raw_tr[y_raw <= UNIT_THRESHOLD]
    unit_tab = pd.DataFrame({
        "median_normal_jam": normal.groupby("layanan")["durasi_jam"].median(),
        "median_outlier_dibagi_60": (big["durasi_jam"] / 60).groupby(big["layanan"]).median(),
        "n_outlier": big.groupby("layanan").size(),
    }).round(1)
    unit_tab.to_csv(os.path.join(res_dir, "a_label_satuan_menit.csv"))
    R["audit"] = dict(
        n_label_menit=int(len(big)), pct_label_menit=round(100 * len(big) / len(raw_tr), 3),
        max_label_raw=float(y_raw.max()), max_label_dibagi_60=round(float(big["durasi_jam"].max() / 60), 1),
        p99_label=round(float(y_raw.quantile(0.99)), 1),
        scan_mean_train=round(float(raw_tr.jumlah_scan.mean()), 2), scan_mean_test=round(float(raw_te.jumlah_scan.mean()), 2),
        scan_max_train=int(raw_tr.jumlah_scan.max()), scan_max_test=int(raw_te.jumlah_scan.max()),
        scan_spearman_target=round(float(raw_tr[["jumlah_scan", "durasi_jam"]].corr("spearman").iloc[0, 1]), 3),
        jarak_9999_train=int((raw_tr.jarak_km == 9999).sum()), jarak_9999_test=int((raw_te.jarak_km == 9999).sum()),
        jarak_0_train=int((raw_tr.jarak_km == 0).sum()), jarak_0_test=int((raw_te.jarak_km == 0).sum()),
        berat_missing_train=int(raw_tr.berat_kg.isna().sum()), berat_missing_test=int(raw_te.berat_kg.isna().sum()),
        berat_missing_pct=round(100 * raw_tr.berat_kg.isna().mean(), 2),
        share_kargo_saat_berat_hilang=round(100 * (raw_tr.loc[raw_tr.berat_kg.isna(), "layanan"] == "kargo").mean(), 1),
        share_kargo_total=round(100 * (raw_tr.layanan == "kargo").mean(), 1),
        umur_akun_mustahil=int((raw_tr.umur_akun_penjual_hari > 20000).sum()),
    )
    allp = pd.concat([raw_tr, raw_te])
    R["audit"]["f30_mean_per_bulan"] = allp.groupby(pd.to_datetime(allp.tgl_pickup).dt.month)["f30"].mean().round(3).to_dict()
    R["audit"]["f30_spearman_target"] = round(float(raw_tr[["f30", "durasi_jam"]].corr("spearman").iloc[0, 1]), 3)
    R["audit"]["kategori_baru_test"] = {c: sorted(set(raw_te[c]) - set(raw_tr[c])) for c in CAT_COLS}
    R["audit"]["n_baris_k13_test"] = int((raw_te.mitra_kurir == "K13").sum())
    R["audit"]["n_kota_tujuan"] = int(raw_tr.kota_tujuan.nunique())

    tr = clean(raw_tr, True)
    te = clean(raw_te, False)
    levels = {c: sorted(tr[c].astype(str).unique()) for c in CAT_COLS}
    tr["bulan"] = tr["tgl_pickup"].dt.month

    f_cols = [f"f{i:02d}" for i in range(1, 51) if f"f{i:02d}" not in TIME_PROXY_COLS]
    num_cols = ["jam_pickup", "jarak_km", "berat_kg", "volume_liter", "jumlah_item", "nilai_barang", "asuransi", "cod",
                "alamat_skor", "pengalaman_kurir_bulan", "rating_penjual", "umur_akun_penjual_hari", "volume_penjual_30h",
                "antrean_hub_asal", "indeks_kemacetan", "hujan_asal_mm", "hujan_tujuan_mm", "berat_hilang", "hari_minggu",
                "berat_volumetrik", "berat_tagih"]
    FEATS = CAT_COLS + num_cols + f_cols
    R["n_fitur"] = len(FEATS)

    # Lipatan waktu
    A_tr, A_va = tr[tr.bulan <= 6], tr[tr.bulan >= 7]
    I_tr, I_va = tr[tr.bulan <= 4], tr[(tr.bulan >= 5) & (tr.bulan <= 6)]
    jul, agu = A_va[A_va.bulan == 7], A_va[A_va.bulan == 8]
    R["lipatan"] = {k: int(len(d)) for k, d in [("latih_jan_jun", A_tr), ("uji_jul_agu", A_va), ("tuning_latih_jan_apr", I_tr),
                                               ("tuning_val_mei_jun", I_va), ("kalibrasi_jul", jul), ("evaluasi_agu", agu)]}
    yA, yJ, yG = A_va.durasi_jam.to_numpy(), jul.durasi_jam.to_numpy(), agu.durasi_jam.to_numpy()

    def lgb_q(alpha, params, trd, cols=FEATS, n=None):
        p = dict(objective="quantile", alpha=alpha, learning_rate=0.05, n_estimators=n or params.get("n_estimators", 600),
                 num_leaves=params["num_leaves"], min_child_samples=params["min_child_samples"], subsample=0.8,
                 subsample_freq=1, colsample_bytree=0.8, cat_smooth=20, min_data_per_group=100, cat_l2=10,
                 random_state=SEED, n_jobs=-1, verbose=-1)
        return lgb.LGBMRegressor(**p).fit(to_lgb(trd, cols, levels), trd.durasi_jam.to_numpy(),
                                         categorical_feature=[c for c in CAT_COLS if c in cols])

    # ======================================================================
    # (c) TUNING di dalam lipatan latih (Jan-Apr -> Mei-Jun), early stopping
    # ======================================================================
    log("(c) Tuning LightGBM di dalam skema validasi (Jan-Apr -> Mei-Jun)")
    grid = [dict(num_leaves=nl, min_child_samples=mc) for nl in (31, 127) for mc in (50, 300)]
    tune_rows, best = [], {}
    for alpha in (0.5, 0.9):
        for g in grid:
            m = lgb.LGBMRegressor(objective="quantile", alpha=alpha, learning_rate=0.05, n_estimators=1500,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, cat_smooth=20,
                                  min_data_per_group=100, cat_l2=10, random_state=SEED, n_jobs=-1, verbose=-1, **g)
            m.fit(to_lgb(I_tr, FEATS, levels), I_tr.durasi_jam, eval_set=[(to_lgb(I_va, FEATS, levels), I_va.durasi_jam)],
                  eval_metric="quantile", categorical_feature=CAT_COLS,
                  callbacks=[lgb.early_stopping(100, verbose=False)])
            p = m.predict(to_lgb(I_va, FEATS, levels))
            tune_rows.append(dict(alpha=alpha, **g, best_iter=int(m.best_iteration_), pinball=round(pinball(I_va.durasi_jam, p, alpha), 4)))
        cand = [r for r in tune_rows if r["alpha"] == alpha]
        b = min(cand, key=lambda r: r["pinball"])
        best[alpha] = dict(num_leaves=b["num_leaves"], min_child_samples=b["min_child_samples"],
                           n_estimators=int(b["best_iter"] * 1.1) + 10)
    pd.DataFrame(tune_rows).to_csv(os.path.join(res_dir, "c_tuning.csv"), index=False)
    R["tuning_terbaik"] = {str(k): v for k, v in best.items()}

    # ======================================================================
    # (a) DAMPAK KEBOCORAN jumlah_scan & PERBAIKAN LABEL
    # ======================================================================
    log("(a) Mengukur dampak kebocoran jumlah_scan & perbaikan label menit")
    rng = np.random.RandomState(SEED)
    m_leak = lgb_q(0.5, best[0.5], A_tr, FEATS + LEAK_COLS)
    p_leak = m_leak.predict(to_lgb(A_va, FEATS + LEAK_COLS, levels))
    va_real = A_va.copy()
    va_real["jumlah_scan"] = rng.choice(te.jumlah_scan.to_numpy(), size=len(va_real), replace=True)  # seperti saat pickup
    p_leak_real = m_leak.predict(to_lgb(va_real, FEATS + LEAK_COLS, levels))
    m50 = lgb_q(0.5, best[0.5], A_tr)
    p50_A = m50.predict(to_lgb(A_va, FEATS, levels))
    # perbaikan label: latih dengan label mentah (tanpa koreksi menit) vs terkoreksi
    A_tr_raw = A_tr.copy()
    A_tr_raw["durasi_jam"] = raw_tr.loc[A_tr.index, "durasi_jam"].to_numpy()
    p_rawlabel = lgb_q(0.5, best[0.5], A_tr_raw).predict(to_lgb(A_va, FEATS, levels))
    leak_rows = [
        evaluate(yA, p_leak, None, "Dengan jumlah_scan, val apa adanya (semu)"),
        evaluate(yA, p_leak_real, None, "Dengan jumlah_scan, scan seperti saat pickup"),
        evaluate(yA, p_rawlabel, None, "Tanpa jumlah_scan, label mentah (menit tidak dikoreksi)"),
        evaluate(yA, p50_A, None, "Tanpa jumlah_scan, label terkoreksi (dipakai)"),
    ]
    pd.DataFrame(leak_rows).to_csv(os.path.join(res_dir, "a_dampak_kebocoran.csv"), index=False)
    R["kebocoran"] = leak_rows

    # ======================================================================
    # (b) VALIDASI WAKTU vs RANDOM 5-FOLD CV
    # ======================================================================
    log("(b) Random 5-fold CV vs split waktu")
    pool = tr  # Jan-Agu
    kf = KFold(5, shuffle=True, random_state=SEED)
    cv_mae, cv_pin = [], []
    for k, (i_tr, i_va) in enumerate(kf.split(pool)):
        d_tr, d_va = pool.iloc[i_tr], pool.iloc[i_va]
        p = lgb_q(0.5, best[0.5], d_tr).predict(to_lgb(d_va, FEATS, levels))
        cv_mae.append(float(np.mean(np.abs(d_va.durasi_jam - p)))); cv_pin.append(pinball(d_va.durasi_jam, p, 0.5))
    # Validasi waktu bergulir (rolling-origin) sebagai pembanding kedua
    roll = []
    for end in (4, 6):
        d_tr, d_va = tr[tr.bulan <= end], tr[(tr.bulan > end) & (tr.bulan <= end + 2)]
        p = lgb_q(0.5, best[0.5], d_tr).predict(to_lgb(d_va, FEATS, levels))
        roll.append(dict(latih=f"Jan-{end}", uji=f"{end + 1}-{end + 2}", mae=round(float(np.mean(np.abs(d_va.durasi_jam - p))), 3),
                         pinball_50=round(pinball(d_va.durasi_jam, p, 0.5), 3)))
    R["validasi"] = dict(random_cv_mae=round(float(np.mean(cv_mae)), 3), random_cv_mae_std=round(float(np.std(cv_mae)), 3),
                         random_cv_pinball=round(float(np.mean(cv_pin)), 3),
                         waktu_mae=leak_rows[3]["mae_p50"], waktu_pinball=leak_rows[3]["pinball_50"], rolling=roll)
    pd.DataFrame(roll).to_csv(os.path.join(res_dir, "b_validasi_rolling.csv"), index=False)

    # ======================================================================
    # (c) BASELINE vs MODEL LANJUTAN (lipatan Jan-Jun -> Jul-Agu)
    # ======================================================================
    log("(c) Baseline vs model lanjutan")
    grp = A_tr.groupby(["layanan", "pulau_tujuan"])["durasi_jam"]
    q50_tab, q90_tab = grp.quantile(0.5), grp.quantile(0.9)
    key = list(zip(A_va.layanan, A_va.pulau_tujuan))
    b50 = np.array([q50_tab.get(k, A_tr.durasi_jam.median()) for k in key])
    b90 = np.array([q90_tab.get(k, A_tr.durasi_jam.quantile(0.9)) for k in key])
    rl = RidgeLog([c for c in num_cols] + f_cols).fit(A_tr, A_tr.durasi_jam.to_numpy())
    r50, r90 = rl.predict(A_va)
    m90 = lgb_q(0.9, best[0.9], A_tr)
    p90_A = m90.predict(to_lgb(A_va, FEATS, levels))
    p90_A = np.maximum(p90_A, p50_A)
    model_rows = [
        evaluate(yA, b50, b90, "Baseline: kuantil empiris per layanan x pulau"),
        evaluate(yA, r50, r90, "Lanjutan 1: Ridge target log + kuantil residual"),
        evaluate(yA, p50_A, p90_A, "Lanjutan 2: LightGBM quantile (tuned)"),
    ]
    pd.DataFrame(model_rows).to_csv(os.path.join(res_dir, "c_perbandingan_model.csv"), index=False)
    R["model"] = model_rows

    # ======================================================================
    # (d) KATEGORI: kardinalitas tinggi & kategori yang tak terlihat
    # ======================================================================
    log("(d) Kategori kardinalitas tinggi & simulasi kurir tak dikenal")
    no_kota = [c for c in FEATS if c != "kota_tujuan"]
    p_nokota = lgb_q(0.5, best[0.5], A_tr, no_kota).predict(to_lgb(A_va, no_kota, levels))
    A_tr_te = A_tr.copy(); A_va_te = A_va.copy()
    A_tr_te["kota_tujuan_te"] = target_encode(A_tr.kota_tujuan, np.log1p(A_tr.durasi_jam.to_numpy()), A_tr.kota_tujuan)
    A_va_te["kota_tujuan_te"] = target_encode(A_tr.kota_tujuan, np.log1p(A_tr.durasi_jam.to_numpy()), A_va.kota_tujuan)
    te_cols = no_kota + ["kota_tujuan_te"]
    p_te = lgb_q(0.5, best[0.5], A_tr_te, te_cols).predict(to_lgb(A_va_te, te_cols, levels))
    cat_rows = [evaluate(yA, p_nokota, None, "kota_tujuan dibuang"),
                evaluate(yA, p_te, None, "kota_tujuan target encoding (smoothing k=50)"),
                evaluate(yA, p50_A, None, "kota_tujuan kategori native LightGBM + cat_smooth (dipakai)")]
    # Simulasi kategori baru: kurir K12 dikeluarkan dari data latih lalu dievaluasi di Jul-Agu
    hold = "K12"
    m_hold = lgb_q(0.5, best[0.5], A_tr[A_tr.mitra_kurir != hold])
    va_h = A_va[A_va.mitra_kurir == hold]
    lv_h = dict(levels); lv_h["mitra_kurir"] = [x for x in levels["mitra_kurir"] if x != hold]
    p_h_unseen = m_hold.predict(to_lgb(va_h, FEATS, lv_h))
    p_h_seen = m50.predict(to_lgb(va_h, FEATS, levels))
    yh = va_h.durasi_jam.to_numpy()
    cat_rows += [evaluate(yh, p_h_seen, None, f"Kurir {hold} dikenal saat latih", n=len(va_h)),
                 evaluate(yh, p_h_unseen, None, f"Kurir {hold} tak dikenal -> missing (simulasi K13)", n=len(va_h)),
                 evaluate(yh, b50[(A_va.mitra_kurir == hold).to_numpy()], None, f"Kurir {hold}: baseline", n=len(va_h))]
    pd.DataFrame(cat_rows).to_csv(os.path.join(res_dir, "d_kategori.csv"), index=False)
    R["kategori"] = cat_rows

    # ======================================================================
    # (e) KETIDAKPASTIAN: CQR dengan kalibrasi Juli, evaluasi Agustus
    # ======================================================================
    log("(e) Kalibrasi conformal (CQR): kalibrasi Juli, evaluasi Agustus")
    is_jul = (A_va.bulan == 7).to_numpy()
    s_jul = yJ - p90_A[is_jul]
    n = len(s_jul)
    off_jul = float(np.quantile(s_jul, min(1.0, np.ceil((n + 1) * 0.9) / n)))
    p90_G_raw, p50_G = p90_A[~is_jul], p50_A[~is_jul]
    p90_G_cal = p90_G_raw + off_jul
    unc_rows = [evaluate(yG, p50_G, p90_G_raw, "LightGBM mentah (Agustus)"),
                evaluate(yG, p50_G, p90_G_cal, f"LightGBM + CQR offset Juli ({off_jul:+.2f} jam) (Agustus)"),
                evaluate(yG, b50[~is_jul], b90[~is_jul], "Baseline (Agustus)"),
                evaluate(yG, r50[~is_jul], r90[~is_jul], "Ridge log (Agustus)")]
    pd.DataFrame(unc_rows).to_csv(os.path.join(res_dir, "e_kalibrasi_conformal.csv"), index=False)
    # Offset final: residual out-of-time seluruh Jul-Agu dari model Jan-Jun
    s_all = yA - p90_A
    n2 = len(s_all)
    off_final = float(np.quantile(s_all, min(1.0, np.ceil((n2 + 1) * 0.9) / n2)))
    R["conformal"] = dict(offset_kalibrasi_jul=round(off_jul, 3), offset_final=round(off_final, 3),
                          coverage_raw_jul_agu=round(100 * coverage(yA, p90_A), 2), tabel=unc_rows)

    # ======================================================================
    # (f) ANALISIS GALAT, SHAP, PARTIAL DEPENDENCE
    # ======================================================================
    log("(f) Analisis galat per segmen, SHAP, partial dependence")
    ev = A_va[["layanan", "pulau_tujuan", "mitra_kurir", "durasi_jam"]].copy()
    ev["p50"] = p50_A; ev["p90"] = p90_A + off_jul   # interval terkalibrasi (offset dari Juli)

    def breakdown(col):
        def f(d):
            return pd.Series(dict(n=len(d), median_aktual=round(d.durasi_jam.median(), 1),
                                  mae_p50=round(float(np.mean(np.abs(d.durasi_jam - d.p50))), 2),
                                  pinball_50=round(pinball(d.durasi_jam, d.p50, 0.5), 2),
                                  pinball_90=round(pinball(d.durasi_jam, d.p90, 0.9), 2),
                                  coverage_90=round(100 * coverage(d.durasi_jam, d.p90), 1),
                                  mae_relatif_pct=round(100 * float(np.mean(np.abs(d.durasi_jam - d.p50))) / d.durasi_jam.median(), 1)))
        return ev.groupby(col).apply(f).sort_values("mae_p50", ascending=False)

    for col in ["layanan", "pulau_tujuan", "mitra_kurir"]:
        b = breakdown(col); b.to_csv(os.path.join(res_dir, f"f_galat_per_{col}.csv"))
        R[f"galat_{col}"] = b.reset_index().to_dict(orient="records")

    samp = A_va.sample(5000, random_state=SEED)
    Xs = to_lgb(samp, FEATS, levels)
    contrib = m50.booster_.predict(Xs, pred_contrib=True)[:, :-1]
    shap_imp = pd.DataFrame({"fitur": FEATS, "mean_abs_shap_jam": np.abs(contrib).mean(0)}).sort_values("mean_abs_shap_jam", ascending=False)
    shap_imp["mean_abs_shap_jam"] = shap_imp["mean_abs_shap_jam"].round(3)
    shap_imp.to_csv(os.path.join(res_dir, "f_shap_importance.csv"), index=False)
    R["shap_top"] = shap_imp.head(12).to_dict(orient="records")

    pdp = {}
    pdp_samp = A_va.sample(2000, random_state=SEED + 1)
    for feat in ["jarak_km", "log_antrean" if "log_antrean" in FEATS else "antrean_hub_asal", "alamat_skor"]:
        grid_v = np.nanquantile(pdp_samp[feat], np.linspace(0.02, 0.98, 15))
        vals = []
        for v in grid_v:
            d = pdp_samp.copy(); d[feat] = v
            vals.append(float(m50.predict(to_lgb(d, FEATS, levels)).mean()))
        pdp[feat] = dict(grid=[round(float(g), 3) for g in grid_v], p50=[round(x, 2) for x in vals])
    # PDP kategori layanan
    pd_lay = {}
    for v in levels["layanan"]:
        d = pdp_samp.copy(); d["layanan"] = v
        pd_lay[v] = round(float(m50.predict(to_lgb(d, FEATS, levels)).mean()), 2)
    pdp["layanan"] = pd_lay
    R["pdp"] = pdp

    # ======================================================================
    # MODEL FINAL & PREDIKSI UJI
    # ======================================================================
    log("Melatih model final pada Jan-Agu & memprediksi data uji")
    f50 = lgb_q(0.5, best[0.5], tr)
    f90 = lgb_q(0.9, best[0.9], tr)
    Xt = to_lgb(te, FEATS, levels)
    t50 = f50.predict(Xt)
    t90 = np.maximum(f90.predict(Xt) + off_final, t50)
    t50 = np.clip(t50, 0, None)
    sub = pd.DataFrame({"id_paket": te.id_paket, "p50": np.round(t50, 2), "p90": np.round(np.maximum(t90, t50), 2)})
    assert len(sub) == len(raw_te) and (sub.p90 >= sub.p50).all() and sub.notna().all().all()
    sub.to_csv(os.path.join(output_dir, "prediksi_regresi.csv"), index=False)
    R["final"] = dict(n=len(sub), p50_median=round(float(sub.p50.median()), 2), p90_median=round(float(sub.p90.median()), 2),
                      n_k13=int((te.mitra_kurir == "K13").sum()),
                      p50_median_k13=round(float(sub.p50[(te.mitra_kurir == "K13").to_numpy()].median()), 2))
    log(f"Tersimpan: prediksi_regresi.csv ({len(sub)} baris)")

    # ======================================================================
    # GRAFIK
    # ======================================================================
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        top = shap_imp.head(12)[::-1]
        ax[0].barh(top.fitur, top.mean_abs_shap_jam, color="#4C72B0"); ax[0].set_title("(f) mean |SHAP| P50 (jam)")
        for feat, c in [("jarak_km", "#C44E52")]:
            ax[1].plot(pdp[feat]["grid"], pdp[feat]["p50"], marker="o", color=c)
        ax[1].set_title("(f) Partial dependence: jarak_km"); ax[1].set_xlabel("jarak_km"); ax[1].set_ylabel("P50 rata-rata (jam)")
        b = pd.DataFrame(R["galat_pulau_tujuan"])
        ax[2].bar(b.pulau_tujuan, b.mae_p50, color="#55A868"); ax[2].set_title("(f) MAE P50 per pulau tujuan (Jul-Agu)")
        ax[2].tick_params(axis="x", rotation=45)
        plt.tight_layout(); plt.savefig(os.path.join(res_dir, "fig_task1_shap_pdp_galat.png"), dpi=130); plt.close()
    except Exception as e:
        log(f"Grafik dilewati: {e}")

    R["library"] = dict(lightgbm=lgb.__version__, pandas=pd.__version__, numpy=np.__version__,
                        sklearn=__import__("sklearn").__version__, seed=SEED)
    R["durasi_detik"] = round(time.time() - t0, 1)
    with open(os.path.join(res_dir, "ringkasan_task1.json"), "w") as f:
        json.dump(R, f, indent=2, default=str, ensure_ascii=False)
    log(f"Selesai dalam {R['durasi_detik']} detik")
    return R


if __name__ == "__main__":
    import sys
    run_task1(sys.argv[1] if len(sys.argv) > 1 else None)
