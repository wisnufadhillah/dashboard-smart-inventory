import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

st.set_page_config(page_title="Smart Inventory UMKM", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_csv("dataset_inventory_umkm_bersih.csv")

    bobot = {
        "Sembako": 1.5,
        "Elektronik & Pulsa": 1.1,
        "Pakaian": 0.8,
        "Mainan Anak": 0.5,
        "Perabotan": 0.2,
    }

    df["Units Sold"] = df.apply(
        lambda row: int(row["Units Sold"] * bobot.get(row["Category"], 1)),
        axis=1,
    )
    df["Total_Pendapatan_Rp"] = df["Total_Pendapatan_Rp"] / 100
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def get_config_value(key, default=None):
    env_value = os.getenv(key)
    if env_value:
        return env_value

    try:
        return st.secrets.get(key, default)
    except st.errors.StreamlitSecretNotFoundError:
        return default


def get_prediction_from_api(api_url, sales_history):
    response = requests.post(
        f"{api_url.rstrip('/')}/predict",
        json={"history_30_hari": sales_history},
        timeout=20,
    )
    response.raise_for_status()
    result = response.json()
    if result.get("status") != "success":
        raise RuntimeError(result.get("detail") or result.get("error") or result)
    return result["prediksi_besok"]


df = load_data()
daily_sales = df.groupby("Date")["Units Sold"].sum().reset_index()

st.title("Dashboard Smart Inventory Forecasting UMKM")
st.markdown("Visualisasi data transaksi warung sebelum dicolok ke model AI prediksi.")

col1, col2, col3 = st.columns(3)
col1.metric("Total Transaksi", f"{len(df):,}")
col2.metric("Total Barang Terjual", f"{df['Units Sold'].sum():,}")

if "Total_Pendapatan_Rp" in df.columns:
    col3.metric("Total Pendapatan", f"Rp {df['Total_Pendapatan_Rp'].sum():,.0f}")
else:
    col3.metric("Total Pendapatan", "Data tidak tersedia")

st.divider()

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Tren Penjualan Harian")
    fig1 = px.line(
        daily_sales,
        x="Date",
        y="Units Sold",
        title="Total Barang Keluar per Hari",
    )
    st.plotly_chart(fig1, use_container_width=True)

with chart_col2:
    st.subheader("Top Kategori Paling Laris")
    cat_sales = (
        df.groupby("Category")["Units Sold"]
        .sum()
        .reset_index()
        .sort_values("Units Sold", ascending=False)
    )
    fig2 = px.bar(cat_sales, x="Category", y="Units Sold", color="Category")
    st.plotly_chart(fig2, use_container_width=True)

st.warning(
    "Early Warning System: Nanti di sini ditaruh indikator kalau "
    "'Inventory Level' sudah mau habis."
)

st.divider()
st.subheader("Asisten Bisnis AI (OpenRouter)")

OPENROUTER_API_KEY = get_config_value("OPENROUTER_API_KEY")
OPENROUTER_MODEL = get_config_value("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_SITE_URL = get_config_value("OPENROUTER_SITE_URL", "http://localhost:8501")
OPENROUTER_APP_NAME = get_config_value("OPENROUTER_APP_NAME", "Smart Inventory UMKM")
INVENTORY_API_URL = get_config_value("INVENTORY_API_URL", "http://127.0.0.1:8000")

if not OPENROUTER_API_KEY:
    st.error(
        "OpenRouter API Key belum ketemu. Isi OPENROUTER_API_KEY di file .env "
        "atau Streamlit Secrets."
    )
else:
    openrouter_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        default_headers={
            "HTTP-Referer": OPENROUTER_SITE_URL,
            "X-Title": OPENROUTER_APP_NAME,
        },
    )

    if st.button("Cek Koneksi OpenRouter"):
        with st.spinner("Lagi ngecek koneksi ke OpenRouter..."):
            try:
                response = openrouter_client.chat.completions.create(
                    model=OPENROUTER_MODEL,
                    messages=[
                        {
                            "role": "user",
                            "content": "Balas singkat: koneksi OpenRouter berhasil.",
                        }
                    ],
                    max_tokens=20,
                )
                st.success("Berhasil konek ke OpenRouter!")
                st.write(response.choices[0].message.content)
            except Exception as e:
                st.error(
                    "Gagal ngecek OpenRouter. Cek API key, nama model, "
                    f"atau saldo akun. Detail: {e}"
                )

    kategori_barang = "Sembako"
    history_30_hari = daily_sales["Units Sold"].tail(30).astype(float).tolist()

    try:
        angka_prediksi_lstm = get_prediction_from_api(
            INVENTORY_API_URL,
            history_30_hari,
        )
    except Exception as e:
        angka_prediksi_lstm = None
        st.error(f"Gagal mengambil prediksi dari API inventory. Detail: {e}")

    if angka_prediksi_lstm is not None:
        st.info(
            "Info Sistem: Model API memprediksi besok akan ada lonjakan penjualan "
            f"**{kategori_barang}** sebanyak **{angka_prediksi_lstm} unit**."
        )

    if st.button("Minta Saran Bisnis dari AI"):
        if angka_prediksi_lstm is None:
            st.error("Saran bisnis belum bisa dibuat karena prediksi API belum tersedia.")
            st.stop()

        with st.spinner("Si Asisten lagi mikir merangkai kata..."):
            prompt = f"""
            Kamu adalah asisten bisnis untuk UMKM warung kelontong di Indonesia.
            Sistem AI pemprediksi stok kami memperkirakan bahwa besok akan
            terjual {angka_prediksi_lstm} unit {kategori_barang}.
            Berikan 3 poin saran singkat, ramah, dan praktis kepada pemilik
            warung apa yang harus mereka persiapkan hari ini.
            Gunakan bahasa Indonesia yang santai tapi profesional.
            """

            try:
                response = openrouter_client.chat.completions.create(
                    model=OPENROUTER_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Kamu adalah asisten bisnis UMKM yang memberi "
                                "saran praktis dalam bahasa Indonesia."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.7,
                )
                st.success("Saran Bisnis untuk Juragan Warung:")
                st.write(response.choices[0].message.content)
            except Exception as e:
                st.error(f"Waduh, gagal manggil OpenRouter. Error detail: {e}")
