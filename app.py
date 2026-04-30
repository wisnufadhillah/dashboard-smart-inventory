import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Smart Inventory UMKM", layout="wide")

@st.cache_data
def load_data():
    # Karena CSV-nya ada di folder yang sama sama app.py, path-nya tinggal nama filenya aja
    df = pd.read_csv('dataset_inventory_umkm_bersih.csv')
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