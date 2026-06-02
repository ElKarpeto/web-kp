import os
import re
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# TODO: misal ada library yang mau ditambah monggo ae iki
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.linear_model import LinearRegression, TweedieRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor

# ! const data
# TODO: masukin yang mobile boloo
MOBILE_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data/mobile/data-web.csv"
)
HH_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data/household/data-web.csv"
)

# TODO: masukin yang mobile boloo
MOBILE_WOK_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data/mobile/WOK.csv"
)
HH_WOK_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data/household/WOK.csv"
)

COOR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/KOTA.csv")

MOBILE_EXCEL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data/mobile/excel-web.xlsx"
)
HH_EXCEL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data/household/excel-web.xlsx"
)


# ! function
# TODO: misal ada yang mau ditambahin, monggo boloo
@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def load_excel(path: str) -> pd.DataFrame:
    return pd.read_excel(path)


def _clean_wok_label(series: pd.Series) -> pd.Series:
    # gabung daftar kota multi-baris jadi satu baris "A / B / C"
    return series.apply(
        lambda v: " / ".join(s.strip() for s in re.split(r"[\r\n]+", str(v)) if s.strip())
    )


def get_data(mode, uploaded_file):
    if mode == "📱 Mobile":
        if uploaded_file is not None:
            data = pd.read_excel(uploaded_file)
        else:
            data = load_csv(MOBILE_DATA_PATH)
        data["WOK"] = _clean_wok_label(data["WOK"])
        return data

    if uploaded_file is not None:
        return pd.read_excel(uploaded_file)

    return load_csv(HH_DATA_PATH)


@st.cache_data
def data_prep(mode, data: pd.DataFrame) -> pd.DataFrame:
    if mode == "📱 Mobile":
        # fitur turunan kepadatan gerai; UMR sengaja dibuang (analisis: UMR menurunkan performa)
        data["Gerai Density"] = data["Jumlah Gerai"] / data["Luas WOK"]
        data = data.drop(columns=["UMR"], errors="ignore")
    else:
        data["Kepadatan Gerai"] = np.log(data["Jumlah Gerai"] / data["Luas WOK"])
        data["Kepadatan Kepala Keluarga"] = np.log(
            data["Jumlah Kepala Keluarga"] / data["Luas WOK"]
        )
        data["revenue_cb"] = np.log(data["Revenue"]) * np.log(
            data["Jumlah Customer Base"]
        )
        data = data.drop(
            columns=[
                "Jumlah Gerai",
                "Jumlah Kepala Keluarga",
                "Revenue",
                "Jumlah Customer Base",
            ]
        )

    return data


# matrik score untuk ditampilkan di dashboard
@st.cache_data
def train_loocv(mode, data: pd.DataFrame):
    X = data.drop(columns=["WOK", "Jumlah Karyawan"])
    y = data["Jumlah Karyawan"]

    loo = LeaveOneOut()

    if mode == "📱 Mobile":
        # OLS + Log(Y): StandardScaler -> LinearRegression, target di-log lalu di-exp balik
        pipeline = TransformedTargetRegressor(
            regressor=Pipeline(
                [("scaler", StandardScaler()), ("model", LinearRegression())]
            ),
            func=np.log,
            inverse_func=np.exp,
        )
    else:
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=3)),
                (
                    "model",
                    TweedieRegressor(power=1.9854, alpha=0.0222),
                ),
            ]
        )

    y_pred_raw = cross_val_predict(pipeline, X, y, cv=loo)
    y_pred = np.maximum(1, y_pred_raw)

    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)

    return y.to_numpy(), np.array(y_pred), mae, rmse, r2


# model untuk prediksi
@st.cache_data
def train_full(mode, data: pd.DataFrame):
    X = data.drop(columns=["WOK", "Jumlah Karyawan"])
    y = data["Jumlah Karyawan"]

    if mode == "📱 Mobile":
        pipeline = TransformedTargetRegressor(
            regressor=Pipeline(
                [("scaler", StandardScaler()), ("model", LinearRegression())]
            ),
            func=np.log,
            inverse_func=np.exp,
        )
    else:
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=3)),
                (
                    "model",
                    TweedieRegressor(power=1.9854, alpha=0.0222),
                ),
            ]
        )

    model = pipeline.fit(X, y)
    return model


@st.cache_data
def _wok_coor(path_wok, coor_path):
    wok = load_csv(path_wok)
    coor = load_csv(coor_path)

    df = wok.merge(coor, left_on="Kota", right_on="Kota", how="left")
    group = df.groupby("WOK")[["Lintang", "Bujur"]].mean()

    wok_coor_dict = {}
    for w, c in group.iterrows():
        wok_coor_dict[w] = (c["Lintang"], c["Bujur"])

    return wok_coor_dict


# samakan ejaan kota mobile dengan KOTA.csv
_KOTA_FIX = {"KARANGASEM": "KARANG ASEM", "KOTA SURAKARTA (SOLO)": "KOTA SURAKARTA"}


def _norm_kota(s: str) -> str:
    s = str(s).strip().upper()
    return _KOTA_FIX.get(s, s)


@st.cache_data
def _mobile_wok_coor(data: pd.DataFrame, coor_path):
    # koordinat tiap WOK = rata-rata koordinat kota-kotanya (label WOK = "A / B / C")
    coor = load_csv(coor_path)
    canon = {
        _norm_kota(k): (lat, lon)
        for k, lat, lon in zip(coor["Kota"], coor["Lintang"], coor["Bujur"])
    }

    result = {}
    for cell in data["WOK"].unique():
        kotas = [_norm_kota(x) for x in str(cell).split(" / ") if x.strip()]
        pts = [canon[k] for k in kotas if k in canon]
        if pts:
            result[cell] = (
                float(np.mean([p[0] for p in pts])),
                float(np.mean([p[1] for p in pts])),
            )
    return result


def build_geo_df(labels, aktual, prediksi, wok_coor):
    rows = []
    for lbl, akt, pred in zip(labels, aktual, prediksi):
        if lbl in wok_coor:
            lat, lon = wok_coor[lbl]
            rows.append(
                {
                    "WOK": lbl,
                    "Jumlah Karyawan Saat Ini": int(akt),
                    "Jumlah Karyawan Prediksi": int(pred),
                    "Selisih": int(pred) - int(akt),
                    "lat": lat,
                    "lon": lon,
                }
            )

    return pd.DataFrame(rows)


def make_map(geo_df, color_col, colorscale="Blues"):
    fig = px.scatter_map(
        geo_df,
        lat="lat",
        lon="lon",
        size=color_col,
        color=color_col,
        color_continuous_scale=colorscale,
        size_max=35,
        zoom=5,
        center={"lat": -7.8, "lon": 117.5},
        hover_name="WOK",
        hover_data={
            "Jumlah Karyawan Saat Ini": True,
            "Jumlah Karyawan Prediksi": True,
            "Selisih": True,
            "lat": False,
            "lon": False,
        },
        map_style="basic",
    )
    fig.update_layout(
        height=480,
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        coloraxis_colorbar=dict(title="Prediksi Jumlah Karyawan"),
    )
    return fig


def download_excel(mode: str, path: str):
    with open(path, "rb") as f:
        return st.download_button(
            label="📥 Unduh Templat Excel",
            data=f,
            file_name=f"template-{mode}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


# page config
st.set_page_config(
    page_title="Workforce Planning — Telkomsel",
    page_icon="📡",
    layout="wide",
)

# title
st.title("📡 Workforce Planning — Telkomsel")
st.markdown("Prediksi kebutuhan karyawan per WOK (Wilayah Operasional)")

# select mode
mode = st.segmented_control(
    "Sektor", ["📱 Mobile", "🏠 Household"], default="🏠 Household", key="sektor_mode"
)

st.divider()

st.subheader("Kostomisasi Data")
st.markdown("Unduh dan Upload templat file excel di bawah ini")

# upload file untuk prediksi mandiri
if mode == "📱 Mobile":
    download_excel("mobile", MOBILE_EXCEL)
else:
    download_excel("household", HH_EXCEL)

uploaded_file = st.file_uploader("Unggah File Excel", type=["xlsx", "xls"])

if uploaded_file is not None:
    st.success(f"File '{uploaded_file.name}' berhasil di-_upload_!")

# data
df = get_data(mode, uploaded_file)
preped_data = data_prep(mode, df)

y_true, y_pred, mae, rmse, r2 = train_loocv(mode, preped_data)
model = train_full(mode, preped_data)

# tabs
tab1, tab2, tab3 = st.tabs(
    ["📊 Dashboard", "📋 Prediksi per WOK", "🎯 Prediksi Data Baru"]
)

with tab1:
    # persebaran karyawan
    karyawan_df = df.sort_values(by="Jumlah Karyawan", ascending=False)
    avg_val = karyawan_df["Jumlah Karyawan"].mean()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=karyawan_df["WOK"],
            y=karyawan_df["Jumlah Karyawan"],
            hovertemplate="<b>%{x}</b><br>"
            "Jumlah Karyawan: %{y} Orang<br>"
            "<extra></extra>",
        )
    )
    fig.add_hline(
        y=avg_val,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Rata-rata: {avg_val:.2f}",
        annotation_position="top right",
    )
    fig.update_layout(
        title="Jumlah Karyawan per WOK",
        xaxis_title="WOK",
        yaxis_title="Jumlah Karyawan",
        height=600,
        hovermode="closest",
        # xaxis=dict(showticklabels=False),
    )
    st.plotly_chart(fig, width="stretch")

    # persebaran UMR
    umr_df = df.sort_values(by="UMR", ascending=False)
    avg_val = umr_df["UMR"].mean()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=umr_df["WOK"],
            y=umr_df["UMR"],
            hovertemplate="<b>%{x}</b><br>" "UMR: Rp %{y:.2f}<br>" "<extra></extra>",
        )
    )
    fig.add_hline(
        y=avg_val,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Rata-rata: {avg_val:.2f}",
        annotation_position="top right",
    )
    fig.update_layout(
        title="UMR per WOK",
        xaxis_title="WOK",
        yaxis_title="UMR (Rupiah)",
        height=600,
        hovermode="closest",
        # xaxis=dict(showticklabels=False),
    )
    st.plotly_chart(fig, width="stretch")

    # persebaran luas wilayah
    luas_df = df.sort_values(by="Luas WOK", ascending=False)
    avg_val = luas_df["Luas WOK"].mean()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=luas_df["WOK"],
            y=luas_df["Luas WOK"],
            hovertemplate="<b>%{x}</b><br>"
            "Luas Wilayah: %{y:.2f} KM²<br>"
            "<extra></extra>",
        )
    )
    fig.add_hline(
        y=avg_val,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Rata-rata: {avg_val:.2f}",
        annotation_position="top right",
    )
    fig.update_layout(
        title="Luas Wilayah per WOK",
        xaxis_title="WOK",
        yaxis_title="Luas Wilayah (KM²)",
        height=600,
        hovermode="closest",
        # xaxis=dict(showticklabels=False),
    )
    st.plotly_chart(fig, width="stretch")

    # persebaran grapari
    gerai_df = df.sort_values(by="Jumlah Gerai", ascending=False)
    avg_val = gerai_df["Jumlah Gerai"].mean()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=gerai_df["WOK"],
            y=gerai_df["Jumlah Gerai"],
            hovertemplate="<b>%{x}</b><br>"
            "Jumlah GraPARI: %{y:.2f}<br>"
            "<extra></extra>",
        )
    )
    fig.add_hline(
        y=avg_val,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Rata-rata: {avg_val:.2f}",
        annotation_position="top right",
    )
    fig.update_layout(
        title="Jumlah GraPARI per WOK",
        xaxis_title="WOK",
        yaxis_title="Jumlah GraPARI",
        height=600,
        hovermode="closest",
        # xaxis=dict(showticklabels=False),
    )
    st.plotly_chart(fig, width="stretch")

    # persebaran revenue per karyawan
    rpe_df = df.copy()
    rpe_df["rpe"] = rpe_df["Revenue"] / rpe_df["Jumlah Karyawan"]
    rpe_df = rpe_df.sort_values(by="rpe", ascending=False)
    avg_val = rpe_df["rpe"].mean()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=rpe_df["WOK"],
            y=rpe_df["rpe"],
            hovertemplate="<b>%{x}</b><br>"
            "Revenue per Karyawan: %{y:.2f} Miliar<br>"
            "<extra></extra>",
        )
    )
    fig.add_hline(
        y=avg_val,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Rata-rata: {avg_val:.2f}",
        annotation_position="top right",
    )
    fig.update_layout(
        title="Revenue per Karyawan per WOK",
        xaxis_title="WOK",
        yaxis_title="Revenue per Karyawan (Miliar Rupiah)",
        height=600,
        hovermode="closest",
        # xaxis=dict(showticklabels=False),
    )
    st.plotly_chart(fig, width="stretch")

    # persebaran customer base per karyawan
    cb_karyawan_df = df.copy()
    cb_karyawan_df["cb_karyawan"] = (
        cb_karyawan_df["Jumlah Customer Base"] / cb_karyawan_df["Jumlah Karyawan"]
    )
    cb_karyawan_df = cb_karyawan_df.sort_values(by="cb_karyawan", ascending=False)
    avg_val = cb_karyawan_df["cb_karyawan"].mean()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=cb_karyawan_df["WOK"],
            y=cb_karyawan_df["cb_karyawan"],
            hovertemplate="<b>%{x}</b><br>"
            "Customer Base per Karyawan: %{y:.2f} Orang<br>"
            "<extra></extra>",
        )
    )
    fig.add_hline(
        y=avg_val,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Rata-rata: {avg_val:.2f}",
        annotation_position="top right",
    )
    fig.update_layout(
        title="Customer Base per Karyawan per WOK",
        xaxis_title="WOK",
        yaxis_title="Customer Base per Karyawan (Orang)",
        height=600,
        hovermode="closest",
        # xaxis=dict(showticklabels=False),
    )
    st.plotly_chart(fig, width="stretch")

    # persebaran luas wilayah per karyawan
    luas_karyawan_df = df.copy()
    luas_karyawan_df["luas_karyawan"] = (
        luas_karyawan_df["Luas WOK"] / luas_karyawan_df["Jumlah Karyawan"]
    )
    luas_karyawan_df = luas_karyawan_df.sort_values(by="luas_karyawan", ascending=False)
    avg_val = luas_karyawan_df["luas_karyawan"].mean()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=luas_karyawan_df["WOK"],
            y=luas_karyawan_df["luas_karyawan"],
            hovertemplate="<b>%{x}</b><br>"
            "Luas Wilayah per Karyawan: %{y:.2f} KM²<br>"
            "<extra></extra>",
        )
    )
    fig.add_hline(
        y=avg_val,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Rata-rata: {avg_val:.2f}",
        annotation_position="top right",
    )
    fig.update_layout(
        title="Luas Wilayah per Karyawan per WOK",
        xaxis_title="WOK",
        yaxis_title="Luas Wilayah per Karyawan (KM²)",
        height=600,
        hovermode="closest",
        # xaxis=dict(showticklabels=False),
    )
    st.plotly_chart(fig, width="stretch")

    # grafik dashboard di atas dipakai bersama oleh mode Household & Mobile

with tab2:
    # performa prediksi
    st.subheader("Performa Prediksi")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah WOK", len(df))
    c2.metric(
        "R²",
        f"{r2:.4f}",
        help="Koefisien determinasi (_higher better_)",
    )
    c3.metric("MAE", f"{mae:.2f} orang", help="Error rata-rata (_lower better_)")
    c4.metric(
        "RMSE", f"{rmse:.2f} orang", help="Error rata-rata terbobot (_lower better_)"
    )

    st.divider()

    # perbandingan scatter plot
    st.subheader("Perbandingan")
    st.markdown("Perbandingan jumlah karyawan saat ini dengan prediksi model matematis")

    fig = go.Figure()
    max_val = max(max(y_true), max(y_pred)) * 1.05
    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            line=dict(color="red", dash="dash", width=2),
            name="Prediksi Sempurna",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=y_true,
            y=y_pred,
            mode="markers",
            marker=dict(
                size=10,
                color="#1d8eff",
                opacity=0.75,
                line=dict(color="white", width=1),
            ),
            text=df["WOK"],
            hovertemplate="<b>%{text}</b><br>Aktual: %{x}<br>Prediksi: %{y:.1f}<extra></extra>",
            name="WOK",
        )
    )

    fig.update_layout(
        xaxis_title="Jumlah Karyawan Saat ini",
        yaxis_title="Jumlah Karyawan Prediksi",
        legend=dict(x=0.01, y=0.99),
        height=500,
        hovermode="closest",
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"Model berhasil memprediksi jumlah karyawan dengan selisih rata-rata prediksi dengan karyawan saat ini sebesar **±{mae:.2f} orang** per WOK."
    )

    st.divider()

    st.subheader("Tabel Prediksi")

    pred_table = pd.DataFrame(
        {
            "WOK": df["WOK"],
            "Jumlah Karyawan Saat Ini": df["Jumlah Karyawan"],
            "Jumlah Karyawan Prediksi": np.round(y_pred),
        }
    )
    pred_table["Selisih"] = np.abs(
        pred_table["Jumlah Karyawan Saat Ini"] - pred_table["Jumlah Karyawan Prediksi"]
    )

    s1, s2 = st.columns(2)
    s1.metric(
        "Total Karyawan Saat Ini", int(pred_table["Jumlah Karyawan Saat Ini"].sum())
    )
    net = int(pred_table["Jumlah Karyawan Prediksi"].sum()) - int(
        pred_table["Jumlah Karyawan Saat Ini"].sum()
    )
    s2.metric(
        "Total Prediksi Model",
        int(pred_table["Jumlah Karyawan Prediksi"].sum()),
        delta=net,
    )

    st.dataframe(
        pred_table.sort_values(by="Selisih", ascending=True),
        width="stretch",
        hide_index=True,
    )

    st.divider()

    st.subheader("Peta Prediksi")

    if mode == "📱 Mobile":
        wok_coor = _mobile_wok_coor(df, COOR_PATH)
        _geo = build_geo_df(df["WOK"], y_true, np.round(y_pred), wok_coor)
        st.plotly_chart(
            make_map(
                _geo,
                "Jumlah Karyawan Prediksi",
                "Oranges",
            ),
            width="stretch",
        )
    else:
        wok_coor = _wok_coor(HH_WOK_PATH, COOR_PATH)
        _geo = build_geo_df(df["WOK"], y_true, np.round(y_pred), wok_coor)
        st.plotly_chart(
            make_map(
                _geo,
                "Jumlah Karyawan Prediksi",
                "Oranges",
            ),
            width="stretch",
        )

with tab3:
    st.subheader("Prediksi Kebutuhan Karyawan — WOK Baru")
    st.markdown(
        "Masukkan data karakteristik WOK untuk mendapatkan estimasi jumlah karyawan yang direkomendasikan oleh model."
    )
    if mode == "📱 Mobile":
        mobile_features = [
            "Revenue",
            "Luas WOK",
            "Jumlah Penduduk (Jiwa)",
            "Jumlah Gerai",
            "Jumlah Mitra",
            "Jumlah Customer Base",
            "Total Kecamatan",
            "HDI",
            "PDRB",
            "Gerai Density",
        ]
        medians = preped_data[mobile_features].median()

        with st.form("form_mobile"):
            col_form1, col_form2, col_form3 = st.columns(3)

            with col_form1:
                revenue = st.number_input(
                    "Revenue Mobile (Miliar)",
                    min_value=0.0,
                    value=float(medians["Revenue"]),
                    step=1.0,
                )
                luas = st.number_input(
                    "Luas WOK (KM²)",
                    min_value=0.01,
                    value=float(medians["Luas WOK"]),
                    step=1.0,
                )
                penduduk = st.number_input(
                    "Jumlah Penduduk (Jiwa)",
                    min_value=0,
                    value=int(medians["Jumlah Penduduk (Jiwa)"]),
                    step=1000,
                )

            with col_form2:
                gerai = st.number_input(
                    "Jumlah Gerai (GraPARI)",
                    min_value=0,
                    value=int(medians["Jumlah Gerai"]),
                    step=1,
                )
                mitra = st.number_input(
                    "Jumlah Mitra",
                    min_value=0,
                    value=int(medians["Jumlah Mitra"]),
                    step=1,
                )
                cb = st.number_input(
                    "Jumlah _Customer Base_ (Pelanggan)",
                    min_value=0,
                    value=int(medians["Jumlah Customer Base"]),
                    step=1000,
                )

            with col_form3:
                kecamatan = st.number_input(
                    "Total Kecamatan",
                    min_value=0,
                    value=int(medians["Total Kecamatan"]),
                    step=1,
                )
                hdi = st.number_input(
                    "HDI (_Human Development Index_)",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(medians["HDI"]),
                    step=0.01,
                    format="%.2f",
                )
                pdrb = st.number_input(
                    "PDRB per Kapita",
                    min_value=0.0,
                    value=float(medians["PDRB"]),
                    step=0.1,
                )

            submitted = st.form_submit_button(
                "🎯 Prediksi", width="stretch", type="primary"
            )

        if submitted:
            if luas <= 0:
                st.error("Luas WOK tidak boleh bernilai 0")
            else:
                gerai_density = gerai / luas

                input_dict = {
                    "Revenue": revenue,
                    "Luas WOK": luas,
                    "Jumlah Penduduk (Jiwa)": penduduk,
                    "Jumlah Gerai": gerai,
                    "Jumlah Mitra": mitra,
                    "Jumlah Customer Base": cb,
                    "Total Kecamatan": kecamatan,
                    "HDI": hdi,
                    "PDRB": pdrb,
                    "Gerai Density": gerai_density,
                }

                input_df = pd.DataFrame(data=[input_dict])[mobile_features]
                pred_raw = float(model.predict(input_df)[0])
                pred = int(np.maximum(1, round(pred_raw)))

                st.success(f"Estimasi Kebutuhan Karyawan: **{pred} orang**")

                with st.expander("Detail input yang digunakan"):
                    st.json(input_dict)
    else:
        feature_cols = [
            "UMR",
            "Jumlah Gerai",
            "Jumlah Kepala Keluarga",
            "Luas WOK",
            "Revenue",
            "Jumlah Customer Base",
        ]
        medians = df[feature_cols].median()

        with st.form("form_household"):
            col_form1, col_form2, col_form3 = st.columns(3)

            with col_form1:
                umr = st.number_input(
                    "UMR (Rupiah)",
                    min_value=0.0,
                    value=float(medians["UMR"]),
                    step=float(1e4),
                )
                luas = st.number_input(
                    "Luas Wilayah (KM²)",
                    min_value=0.0,
                    value=float(medians["Luas WOK"]),
                    step=1.0,
                )

            with col_form2:
                kk = st.number_input(
                    "Jumlah Kepala Keluarga (Orang)",
                    min_value=0,
                    value=int(medians["Jumlah Kepala Keluarga"]),
                    step=1,
                )
                gerai = st.number_input(
                    "Jumlah GraPARI",
                    min_value=0,
                    value=int(medians["Jumlah Gerai"]),
                    step=1,
                )

            with col_form3:
                cb = st.number_input(
                    "Jumlah _Customer Base_ (Orang)",
                    min_value=0,
                    value=int(medians["Jumlah Customer Base"]),
                    step=100,
                )
                revenue = st.number_input(
                    "Revenue (Miliar)",
                    min_value=0.0,
                    value=float(medians["Revenue"]),
                    step=1.0,
                )

            submitted = st.form_submit_button(
                "🎯 Prediksi", width="stretch", type="primary"
            )

        if submitted:
            if luas <= 0:
                st.error("Luas Wilayah tidak boleh bernilai 0")
            else:
                gerai_density = np.log(gerai / luas)
                kk_density = np.log(kk / luas)
                revenue_cb = np.log(revenue) * np.log(cb)

                input_dict = {
                    "UMR": np.log(umr),
                    "Luas WOK": luas,
                    "Kepadatan Gerai": gerai_density,
                    "Kepadatan Kepala Keluarga": kk_density,
                    "revenue_cb": revenue_cb,
                }

                input_df = pd.DataFrame(data=[input_dict])
                pred_raw = float(model.predict(input_df)[0])
                pred = np.maximum(1, round(pred_raw))

                st.success(f"Estimasi Kebutuhan Karyawan: **{pred} orang**")

                with st.expander("Detail input yang digunakan"):
                    st.json(input_dict)
