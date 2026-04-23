import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Konfigurasi Halaman Web
st.set_page_config(page_title="Smart Inventory UMKM", page_icon="📦", layout="wide")

# Bikin Judul Kece
st.title("📦 Dashboard Manajemen Inventaris Cerdas")
st.markdown("Pantau laju penjualan dan status stok barang UMKM lu di sini biar nggak kecolongan!")

# ==========================================
# 1. LOAD DATASET (Pake Caching biar webnya gak lemot)
# ==========================================
@st.cache_data
def load_data():
    # Pastiin file CSV ini ada di folder yang sama dengan app.py
    df = pd.read_csv('Inventory_Dataset_Ready.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()
df_sales = df[df['Transaction_Type'] == 'OUT (Penjualan)']

# ==========================================
# 2. BIKIN METRIK RINGKASAN (Nongkrong di paling atas)
# ==========================================
st.subheader("📊 Ringkasan Penjualan")
col1, col2, col3 = st.columns(3)

total_omset = df_sales['Sales'].sum()
total_barang_terjual = df_sales['Quantity'].sum()
total_transaksi = len(df_sales)

col1.metric("Total Omset (Rp)", f"Rp {total_omset:,.0f}")
col2.metric("Total Barang Keluar (Pcs)", f"{total_barang_terjual}")
col3.metric("Total Transaksi", f"{total_transaksi}")

st.divider() # Garis pembatas

# ==========================================
# 3. BAGIAN GRAFIK (Dibagi jadi 2 kolom biar rapi)
# ==========================================
kolom_kiri, kolom_kanan = st.columns(2)

# ---> GRAFIK KIRI: Top 10 Barang Laris
with kolom_kiri:
    st.subheader("🔥 Top-Selling Items")
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    top_items = df_sales.groupby('Product_Name')['Quantity'].sum().reset_index()
    top_items = top_items.sort_values(by='Quantity', ascending=False).head(10)
    
    sns.barplot(data=top_items, x='Quantity', y='Product_Name', hue='Product_Name', palette='viridis', legend=False, ax=ax1)
    ax1.set_xlabel('Total Terjual (Pcs)')
    ax1.set_ylabel('')
    st.pyplot(fig1)

# ---> GRAFIK KANAN: Status Stok (Early Warning)
with kolom_kanan:
    st.subheader("⚠️ Status Sisa Stok Terakhir")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    last_stock = df.sort_values('Date').groupby('Product_Name').tail(1)[['Product_Name', 'Current_Stock']]
    last_stock = last_stock.sort_values(by='Current_Stock')
    
    # Warna: Merah kalau di bawah 30, Hijau kalau aman
    colors = ['crimson' if x < 30 else 'mediumseagreen' for x in last_stock['Current_Stock']]
    
    sns.barplot(data=last_stock, x='Current_Stock', y='Product_Name', hue='Product_Name', palette=colors, legend=False, ax=ax2)
    ax2.axvline(x=30, color='red', linestyle='--', label='Batas Aman (30 pcs)')
    ax2.set_xlabel('Sisa Stok (Pcs)')
    ax2.set_ylabel('')
    ax2.legend()
    st.pyplot(fig2)

# ---> GRAFIK BAWAH: Tren Penjualan Harian
st.subheader("📈 Tren Penjualan Harian")
fig3, ax3 = plt.subplots(figsize=(15, 4))
tren_harian = df_sales.groupby('Date')['Quantity'].sum().reset_index()

sns.lineplot(data=tren_harian, x='Date', y='Quantity', color='coral', marker='o', ax=ax3)
ax3.set_xlabel('Tanggal')
ax3.set_ylabel('Total Barang Terjual')
plt.xticks(rotation=45)
st.pyplot(fig3)