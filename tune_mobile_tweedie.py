"""
Tuning hyperparameter model Tweedie (sektor Mobile) dengan Optuna.

Tujuan : menemukan power / alpha / n_components PCA yang meminimalkan MAE
         prediksi Jumlah Karyawan, divalidasi dengan Leave-One-Out CV (LOOCV).

Model  : StandardScaler -> PCA -> TweedieRegressor  (target skala asli, link-log)
Fitur  : sama dengan data_prep mobile di app.py (UMR dipakai; HDI & PDRB dibuang).

Cara pakai:
    python tune_mobile_tweedie.py                 # 200 trial (default)
    python tune_mobile_tweedie.py --trials 500    # lebih banyak trial
    python tune_mobile_tweedie.py --no-nested     # lewati validasi jujur (lebih cepat)

Output:
    - parameter terbaik + metrik LOOCV (estimasi tuning, sedikit optimistik)
    - estimasi JUJUR via nested-CV (parameter dipilih ulang di tiap lipatan)
    - potongan kode pipeline siap-tempel ke app.py
    - plot_mobile_tweedie_optuna.png
"""

import argparse
import os
import re
import warnings

import numpy as np
import pandas as pd
import optuna
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, TweedieRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Konfigurasi ────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, "data", "mobile", "data-web.csv")
TARGET = "Jumlah Karyawan"
# fitur sesuai data_prep mobile (UMR dipakai; HDI & PDRB dibuang; + Gerai Density)
FEATURES = [
    "Revenue",
    "Luas WOK",
    "Jumlah Penduduk (Jiwa)",
    "Jumlah Gerai",
    "Jumlah Mitra",
    "Jumlah Customer Base",
    "Total Kecamatan",
    "UMR",
    "Gerai Density",
]
SEED = 42


# ── Data ────────────────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    # rapikan label WOK multi-baris jadi satu baris (sama seperti app.py)
    df["WOK"] = df["WOK"].apply(
        lambda v: " / ".join(s.strip() for s in re.split(r"[\r\n]+", str(v)) if s.strip())
    )
    # fitur turunan
    df["Gerai Density"] = df["Jumlah Gerai"] / df["Luas WOK"]
    return df


# ── Model ───────────────────────────────────────────────────────────────────────
def build_pipeline(power: float, alpha: float, n_components: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components)),
            ("model", TweedieRegressor(power=power, alpha=alpha, max_iter=10000)),
        ]
    )


def loocv_predict(pipe, X, y) -> np.ndarray:
    # prediksi LOOCV, di-clip minimal 1 (jumlah karyawan tidak boleh < 1)
    return np.maximum(1, cross_val_predict(pipe, X, y, cv=LeaveOneOut()))


def metrics(y_true, y_pred):
    return (
        mean_absolute_error(y_true, y_pred),
        r2_score(y_true, y_pred),
        float(np.sqrt(mean_squared_error(y_true, y_pred))),
    )


# ── Optuna objective ─────────────────────────────────────────────────────────────
def make_objective(X, y, cv):
    n_feat = X.shape[1]

    def objective(trial: optuna.Trial) -> float:
        power = trial.suggest_float("power", 1.0, 2.0)
        alpha = trial.suggest_float("alpha", 1e-4, 10.0, log=True)
        n_components = trial.suggest_int("n_components", 2, n_feat)
        pipe = build_pipeline(power, alpha, n_components)
        y_pred = np.maximum(1, cross_val_predict(pipe, X, y, cv=cv))
        return mean_absolute_error(y, y_pred)  # MAE = metrik yang dioptimalkan

    return objective


def tune(X, y, n_trials, cv):
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED)
    )
    study.optimize(make_objective(X, y, cv), n_trials=n_trials, show_progress_bar=False)
    return study


# ── Validasi jujur (nested-CV) ──────────────────────────────────────────────────
def nested_cv(X, y, outer_splits=5, inner_trials=40):
    """Outer KFold; tiap lipatan tuning ulang Optuna di data train -> estimasi tanpa bocor."""
    outer = KFold(n_splits=outer_splits, shuffle=True, random_state=SEED)
    inner = KFold(n_splits=3, shuffle=True, random_state=SEED)
    y_pred = np.zeros(len(y))
    for tr, te in outer.split(X):
        Xtr, ytr = X.iloc[tr], y[tr]
        study = tune(Xtr.reset_index(drop=True), ytr, inner_trials, inner)
        p = study.best_params
        pipe = build_pipeline(p["power"], p["alpha"], p["n_components"]).fit(Xtr, ytr)
        y_pred[te] = np.maximum(1, pipe.predict(X.iloc[te]))
    return metrics(y, y_pred)


# ── Plot ────────────────────────────────────────────────────────────────────────
def save_plot(study, X, y, best_params, df, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pipe = build_pipeline(**best_params)
    y_pred = loocv_predict(pipe, X, y)
    mae, r2, rmse = metrics(y, y_pred)

    fig, ax = plt.subplots(1, 2, figsize=(15, 6.5))

    # 1) riwayat optimasi
    vals = [t.value for t in study.trials if t.value is not None]
    running_best = np.minimum.accumulate(vals)
    ax[0].plot(vals, ".", alpha=0.4, label="MAE per trial")
    ax[0].plot(running_best, "-", color="#e74c3c", lw=2, label="MAE terbaik berjalan")
    ax[0].set_xlabel("Trial", fontweight="bold")
    ax[0].set_ylabel("MAE (LOOCV)", fontweight="bold")
    ax[0].set_title("Riwayat Optimasi Optuna", fontsize=13, fontweight="bold")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    # 2) aktual vs prediksi
    mx = max(y.max(), y_pred.max()) * 1.05
    ax[1].plot([0, mx], [0, mx], "r--", lw=2, label="Garis Ideal")
    ax[1].scatter(y, y_pred, c="#27ae60", s=90, alpha=0.78, edgecolors="white", linewidth=1)
    for i in np.where(y >= 20)[0]:
        ax[1].annotate(
            df["WOK"].iloc[i].split(" / ")[0],
            (y[i], y_pred[i]),
            xytext=(6, -4),
            textcoords="offset points",
            fontsize=8,
            color="#444",
        )
    ax[1].set_xlabel("Aktual", fontweight="bold")
    ax[1].set_ylabel("Prediksi (LOOCV)", fontweight="bold")
    ax[1].set_title(
        f"Tweedie tuned (Optuna)\nMAE={mae:.3f} | R²={r2:.4f} | RMSE={rmse:.2f}",
        fontsize=13,
        fontweight="bold",
    )
    ax[1].set_xlim(0, mx)
    ax[1].set_ylim(0, mx)
    ax[1].legend(loc="upper left")
    ax[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=200, help="jumlah trial Optuna")
    parser.add_argument("--no-nested", action="store_true", help="lewati validasi nested-CV")
    args = parser.parse_args()

    df = load_data()
    X = df[FEATURES]
    y = df[TARGET].to_numpy()
    print(f"Data: {len(df)} WOK, {len(FEATURES)} fitur -> {FEATURES}\n")

    # baseline OLS + Log(Y) (model saat ini)
    ols = TransformedTargetRegressor(
        regressor=Pipeline([("s", StandardScaler()), ("m", LinearRegression())]),
        func=np.log,
        inverse_func=np.exp,
    )
    mae0, r20, rmse0 = metrics(y, loocv_predict(ols, X, y))
    print(f"[Baseline] OLS+Log(Y)  : MAE={mae0:.3f} | R2={r20:.4f} | RMSE={rmse0:.2f}\n")

    # tuning Optuna (LOOCV)
    print(f"Menjalankan Optuna ({args.trials} trial, objektif=MAE LOOCV)...")
    study = tune(X, y, args.trials, LeaveOneOut())
    bp = study.best_params
    pipe = build_pipeline(**bp)
    mae, r2, rmse = metrics(y, loocv_predict(pipe, X, y))

    print("\n================  HASIL  ================")
    print(f"Parameter terbaik : power={bp['power']:.4f}, alpha={bp['alpha']:.5f}, "
          f"n_components={bp['n_components']}")
    print(f"LOOCV (tuning)    : MAE={mae:.3f} | R2={r2:.4f} | RMSE={rmse:.2f}   (sedikit optimistik)")

    if not args.no_nested:
        print("\nValidasi JUJUR (nested-CV, parameter dipilih ulang tiap lipatan)...")
        mae_n, r2_n, rmse_n = nested_cv(X, y)
        print(f"Nested-CV (jujur) : MAE={mae_n:.3f} | R2={r2_n:.4f} | RMSE={rmse_n:.2f}")

    # plot
    plot_path = os.path.join(BASE, "plot_mobile_tweedie_optuna.png")
    save_plot(study, X, y, bp, df, plot_path)
    print(f"\nPlot tersimpan: {plot_path}")

    # snippet siap-tempel
    print("\n--- Pipeline untuk app.py (cabang mobile train_loocv/train_full) ---")
    print(
        "pipeline = Pipeline(\n"
        "    [\n"
        '        ("scaler", StandardScaler()),\n'
        f'        ("pca", PCA(n_components={bp["n_components"]})),\n'
        '        (\n'
        '            "model",\n'
        f"            TweedieRegressor(power={bp['power']:.4f}, alpha={bp['alpha']:.5f}, max_iter=10000),\n"
        "        ),\n"
        "    ]\n"
        ")"
    )


if __name__ == "__main__":
    main()
