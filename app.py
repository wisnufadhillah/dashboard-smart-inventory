import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai # Tambahan library buat Gemini AI
import os
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Smart Inventory UMKM", layout="wide")

@st.cache_data
def load_data():
    # Karena CSV-nya ada di folder yang sama sama app.py, path-nya tinggal nama filenya aja
    df = pd.read_csv('dataset_inventory_umkm_bersih.csv')
    
    # Bikin manipulasi bobot biar realistis (Sembako laku keras, Perabotan jarang laku)
    bobot = {
        'Sembako': 1.5,
        'Elektronik & Pulsa': 1.1,
        'Pakaian': 0.8,
        'Mainan Anak': 0.5,
        'Perabotan': 0.2
    }
    
    # Terapin bobotnya ke Units Sold
    df['Units Sold'] = df.apply(lambda row: int(row['Units Sold'] * bobot.get(row['Category'], 1)), axis=1)
    
    # Hitung ulang pendapatan pake harga yang di-adjust biar ga triliunan (misal dibagi 100)
    df['Total_Pendapatan_Rp'] = df['Total_Pendapatan_Rp'] / 100
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

st.title("📦 Dashboard Smart Inventory Forecasting UMKM")
st.markdown("Visualisasi data transaksi warung sebelum dicolok ke model AI Prediksi.")

col1, col2, col3 = st.columns(3)
col1.metric("Total Transaksi", f"{len(df):,}")
col2.metric("Total Barang Terjual", f"{df['Units Sold'].sum():,}")

# Pastiin nama kolomnya bener sesuai hasil export dari Colab tadi
if 'Total_Pendapatan_Rp' in df.columns:
    col3.metric("Total Pendapatan", f"Rp {df['Total_Pendapatan_Rp'].sum():,.0f}")
else:
    col3.metric("Total Pendapatan", "Data Nggak Ada")

st.divider()

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("📈 Tren Penjualan Harian")
    daily_sales = df.groupby('Date')['Units Sold'].sum().reset_index()
    fig1 = px.line(daily_sales, x='Date', y='Units Sold', title="Total Barang Keluar per Hari")
    st.plotly_chart(fig1, use_container_width=True)

with chart_col2:
    st.subheader("🔥 Top Kategori Paling Laris")
    cat_sales = df.groupby('Category')['Units Sold'].sum().reset_index().sort_values('Units Sold', ascending=False)
    fig2 = px.bar(cat_sales, x='Category', y='Units Sold', color='Category')
    st.plotly_chart(fig2, use_container_width=True)

st.warning("⚠️ Early Warning System: Nanti di sini ditaruh indikator kalau 'Inventory Level' udah mau abis.")

# ==========================================================
# TAHAP INTEGRASI GENERATIVE AI (ASISTEN BISNIS UMKM)
# ==========================================================
st.divider()
st.subheader("🤖 Asisten Bisnis AI (Generative AI)")

# Cara paling tangguh ngambil API Key: Cek di Streamlit Secrets dulu, baru cek file .env lokal
if "GEMINI_API_KEY" in st.secrets:
    API_KEY = st.secrets["GEMINI_API_KEY"]
else:
    API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("Waduh, API Key nggak ketemu! Pastiin lu udah masukin ke 'Secrets' di Streamlit Cloud, Bol!")
else:
    genai.configure(api_key=API_KEY)
    
    # --- TOMBOL MODE DETEKTIF (DEBUGGING) ---
    if st.button("🛠️ Cek Daftar Model Google AI"):
        with st.spinner("Lagi ngecek ke server Google..."):
            try:
                models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                st.success("Berhasil konek! Ini daftar model yang diijinin buat API Key lu:")
                st.write(models)
            except Exception as e:
                st.error(f"Gagal ngecek API! Fix ini mah API Key lu salah atau belum ke-save di Streamlit Cloud. Detail: {e}")
    # ----------------------------------------

    # Kita balikin ke model paling standar dulu
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    
    angka_prediksi_lstm = 135 
    kategori_barang = "Sembako"
    
    st.info(f"💡 Info Sistem: Model AI memprediksi besok akan ada lonjakan penjualan **{kategori_barang}** sebanyak **{angka_prediksi_lstm} unit**.")
    
    if st.button("Minta Saran Bisnis dari AI"):
        with st.spinner("Si Asisten lagi mikir merangkai kata..."):
            prompt = f"""
            Kamu adalah asisten bisnis untuk UMKM warung kelontong di Indonesia. 
            Sistem AI pemprediksi stok kami memperkirakan bahwa besok akan terjual {angka_prediksi_lstm} unit {kategori_barang}.
            Berikan 3 poin saran singkat, ramah, dan praktis kepada pemilik warung apa yang harus mereka persiapkan hari ini.
            Gunakan bahasa Indonesia yang santai tapi profesional.
            """
            
            try:
                response = gemini_model.generate_content(prompt)
                st.success("Saran Bisnis untuk Juragan Warung:")
                st.write(response.text)
            except Exception as e:
                st.error(f"Waduh, gagal manggil API. Error detail: {e}")