import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

st.set_page_config(page_title='Retail Inventory Dashboard', layout='wide', page_icon='🏪')

@st.cache_data
def load_data():
    df = pd.read_csv('retail_store_inventory_data.csv', parse_dates=['Date'])
    df.columns = df.columns.str.replace(' ', '_').str.replace('/', '_')
    df['Year']    = df['Date'].dt.year
    df['Month']   = df['Date'].dt.month
    df['Holiday_Promotion'] = df['Holiday_Promotion'].astype(bool)
    df['Forecast_Error'] = df['Demand_Forecast'] - df['Units_Sold']
    df['MAPE'] = (df['Forecast_Error'] / df['Units_Sold'].replace(0, np.nan)).abs() * 100
    df['Effective_Price'] = df['Price'] * (1 - df['Discount'] / 100)
    df['Price_Competitive_Ratio'] = df['Price'] / df['Competitor_Pricing']
    return df

df = load_data()

# Sidebar
st.sidebar.title('Filter')
sel_region   = st.sidebar.multiselect('Region', df['Region'].unique(), default=list(df['Region'].unique()))
sel_category = st.sidebar.multiselect('Kategori', df['Category'].unique(), default=list(df['Category'].unique()))
sel_year     = st.sidebar.multiselect('Tahun', sorted(df['Year'].unique()), default=list(df['Year'].unique()))

dff = df[(df['Region'].isin(sel_region)) &
         (df['Category'].isin(sel_category)) &
         (df['Year'].isin(sel_year))]

st.title('Retail Inventory Analytics Dashboard')
st.markdown('**Dataset:** retail_store_inventory_data.csv  |  Dibuat dengan Streamlit + Plotly')

# KPI Cards
col1, col2, col3, col4 = st.columns(4)
col1.metric('Total Records', f"{len(dff):,}")
col2.metric('Avg Units Sold/hari', f"{dff['Units_Sold'].mean():.1f}")
col3.metric('Avg Inventory Level', f"{dff['Inventory_Level'].mean():.0f}")
col4.metric('Overall MAPE', f"{dff['MAPE'].mean():.1f}%")

st.divider()

# Row 1
col_a, col_b = st.columns(2)

with col_a:
    cat_sales = dff.groupby('Category')['Units_Sold'].mean().sort_values(ascending=False).reset_index()
    fig1 = px.bar(cat_sales, x='Category', y='Units_Sold',
                  title='Rata-rata Penjualan per Kategori',
                  color='Category', color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig1, use_container_width=True)

with col_b:
    season_sales = dff.groupby('Seasonality')['Units_Sold'].mean().reset_index()
    fig2 = px.bar(season_sales, x='Seasonality', y='Units_Sold',
                  title='Rata-rata Penjualan per Musim',
                  color='Seasonality', color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig2, use_container_width=True)

# Row 2
col_c, col_d = st.columns(2)

with col_c:
    monthly = dff.groupby(['Year','Month'])['Units_Sold'].mean().reset_index()
    monthly['Period'] = pd.to_datetime(monthly[['Year','Month']].assign(Day=1))
    fig3 = px.line(monthly, x='Period', y='Units_Sold', title='Trend Penjualan Bulanan')
    st.plotly_chart(fig3, use_container_width=True)

with col_d:
    fig4 = px.histogram(dff, x='Forecast_Error', nbins=80,
                        title='Distribusi Forecast Error',
                        color_discrete_sequence=['#3498db'])
    fig4.add_vline(x=0, line_dash='dash', line_color='red')
    st.plotly_chart(fig4, use_container_width=True)

# A/B Test Live
st.divider()
st.subheader('A/B Test: Promosi vs Non-Promosi')
grp_a = dff[dff['Holiday_Promotion']==False]['Units_Sold']
grp_b = dff[dff['Holiday_Promotion']==True]['Units_Sold']
if len(grp_a) > 1 and len(grp_b) > 1:
    t_stat, p_val = stats.ttest_ind(grp_a, grp_b, equal_var=False)
    col1, col2, col3 = st.columns(3)
    col1.metric('Avg Non-Promosi', f"{grp_a.mean():.2f}")
    col2.metric('Avg Promosi', f"{grp_b.mean():.2f}",
                delta=f"{grp_b.mean()-grp_a.mean():+.2f}")
    col3.metric('p-value', f"{p_val:.4f}",
                delta='Signifikan ✅' if p_val < 0.05 else 'Tidak Signifikan ❌')

st.caption('Dibuat oleh: [Nama Kamu] | Dataset: retail_store_inventory_data.csv')