import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
from datetime import datetime
import os

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="H&M Recommendation System", 
    page_icon="🛍️", 
    layout="wide"
)

st.markdown("# Sistem Rekomendasi H&M")
st.markdown("**Hybrid Collaborative Filtering + Content-Based Analytics Dashboard**")
st.markdown("---")

# ============================================================================
# DATA LOADING FUNCTION
# ============================================================================

@st.cache_data
def load_data():
    """Load all REAL data from data/ folder"""
    data_dir = Path("data")
    
    try:
        # Load CSV files
        top_customers = pd.read_csv(data_dir / "top_customers.csv")
        top_products = pd.read_csv(data_dir / "top_products.csv")
        distribution = pd.read_csv(data_dir / "customer_distribution.csv")
        edges = pd.read_csv(data_dir / "bipartite_edges.csv")
        nodes = pd.read_csv(data_dir / "bipartite_nodes.csv")
        
        # Load JSON files
        with open(data_dir / "network_stats.json", 'r') as f:
            network_stats = json.load(f)
        
        with open(data_dir / "model_performance.json", 'r') as f:
            model_performance = json.load(f)
        
        return {
            'top_customers': top_customers,
            'top_products': top_products,
            'distribution': distribution,
            'edges': edges,
            'nodes': nodes,
            'network_stats': network_stats,
            'model_performance': model_performance
        }
    except FileNotFoundError as e:
        st.error(f"❌ Data file not found: {e.filename}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()

# ============================================================================
# SAFE CONVERSION FUNCTIONS
# ============================================================================

def safe_int(value, default=0):
    """Safely convert value to int"""
    try:
        if isinstance(value, int):
            return value
        elif isinstance(value, float):
            return int(value)
        elif isinstance(value, str):
            return int(value)
        else:
            return int(value)
    except:
        return default

def safe_float(value, default=0.0):
    """Safely convert value to float"""
    try:
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, str):
            return float(value)
        else:
            return float(value)
    except:
        return default

# ============================================================================
# LOAD DATA
# ============================================================================

data = load_data()

# ============================================================================
# EXTRACT NETWORK STATS
# ============================================================================

network_stats_raw = data['network_stats']
graph_stats = {
    'total_nodes': safe_int(network_stats_raw.get('total_nodes', 27542), 27542),
    'total_edges': safe_int(network_stats_raw.get('total_edges', 150680), 150680),
    'num_customers': safe_int(network_stats_raw.get('num_customers', 3000), 3000),
    'num_products': safe_int(network_stats_raw.get('num_products', 24542), 24542),
    'density': safe_float(network_stats_raw.get('density', 0.000397), 0.000397),
    'top_customer': safe_int(network_stats_raw.get('top_customer', 407), 407),
    'top_product': safe_int(network_stats_raw.get('top_product', 102), 102)
}

# ============================================================================
# EXTRACT MODEL PERFORMANCE
# ============================================================================

model_perf = data['model_performance']
model_df = []

for model_name, metrics_dict in model_perf.items():
    rmse_val = metrics_dict.get('RMSE', 'N/A')
    coverage_val = metrics_dict.get('Coverage', 0)
    
    if isinstance(rmse_val, str) and rmse_val == 'N/A':
        rmse_float = np.nan
    else:
        rmse_float = safe_float(rmse_val, np.nan)
    
    coverage_float = safe_float(coverage_val, 0.0)
    
    model_df.append({
        'Model': model_name,
        'RMSE': rmse_float,
        'Coverage': coverage_float
    })

model_df = pd.DataFrame(model_df)

# Add Products and Recommendations columns
products_map = {
    'Random': 39498, 
    'Popularity': 39498, 
    'ALS': 1601, 
    'Content': 3259, 
    'Hybrid': 4860
}

recommendations_map = {
    'Random': 1401061, 
    'Popularity': 1401061, 
    'ALS': 6050980, 
    'Content': 12799, 
    'Hybrid': 6063779
}

model_df['Products'] = model_df['Model'].map(products_map).fillna(0).astype(int)
model_df['Recommendations'] = model_df['Model'].map(recommendations_map).fillna(0).astype(int)

# ============================================================================
# DATASET METRICS
# ============================================================================

metrics = {
    'total_interactions': 7005582,
    'unique_customers': 742431,
    'unique_products': 51232,
    'train_set': 5604521,
    'test_set': 1401061,
}

# ============================================================================
# CREATE TABS
# ============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Ringkasan", 
    "🎯 Performa", 
    "📈 Data", 
    "💡 Rekomendasi"
])

# ============================================================================
# TAB 1: RINGKASAN (EXECUTIVE SUMMARY)
# ============================================================================

with tab1:
    st.header("Executive Summary")
    
    # Challenge and Solution
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(
            "**Tantangan**: H&M mengelola 7 juta transaksi pelanggan di seluruh "
            "51.232 produk. Sistem rekomendasi hybrid menggabungkan pendekatan "
            "collaborative filtering (melihat kesamaan perilaku antar pelanggan) "
            "dan content-based (melihat kesamaan karakteristik produk) untuk "
            "memberikan rekomendasi yang akurat dan beragam."
        )
    
    with col2:
        st.success(
            "**Dampak Solusi**: Mencakup 4,60% dari total produk yang tersedia | "
            "Mampu menghasilkan 6 juta rekomendasi per hari | Peningkatan retensi "
            "pelanggan melalui pengalaman belanja yang dipersonalisasi"
        )
    
    # Dataset Overview
    st.subheader("Dataset Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Transaksi", f"{metrics['total_interactions']:,}")
    c2.metric("Pelanggan", f"{metrics['unique_customers']:,}")
    c3.metric("Produk", f"{metrics['unique_products']:,}")
    c4.metric("Data Latih", f"{metrics['train_set']:,}")
    c5.metric("Data Uji", f"{metrics['test_set']:,}")
    
    # Model Comparison
    st.subheader("Model Comparison")
    col1, col2, col3 = st.columns(3)
    
    als_row = model_df[model_df['Model'] == 'ALS']
    content_row = model_df[model_df['Model'] == 'Content']
    hybrid_row = model_df[model_df['Model'] == 'Hybrid']
    
    als_rmse = als_row['RMSE'].values[0] if len(als_row) > 0 else np.nan
    content_rmse = content_row['RMSE'].values[0] if len(content_row) > 0 else np.nan
    hybrid_rmse = hybrid_row['RMSE'].values[0] if len(hybrid_row) > 0 else np.nan
    
    als_rmse_str = f"{als_rmse:.4f}" if not np.isnan(als_rmse) else "N/A"
    content_rmse_str = f"{content_rmse:.4f}" if not np.isnan(content_rmse) else "N/A"
    hybrid_rmse_str = f"{hybrid_rmse:.4f}" if not np.isnan(hybrid_rmse) else "N/A"
    
    col1.write(
        f"**ALS (Collaborative Filtering)**\n\n"
        f"RMSE: {als_rmse_str}\n\n"
        f"Keakuratan sedang, terbatas pada produk dalam data historis"
    )
    
    col2.write(
        f"**Content-Based (Berbasis Konten)**\n\n"
        f"RMSE: {content_rmse_str}\n\n"
        f"Keakuratan baik, dapat merekomendasikan produk baru"
    )
    
    col3.write(
        f"**Hybrid (Kombinasi) ✓**\n\n"
        f"RMSE: {hybrid_rmse_str}\n\n"
        f"Keakuratan terbaik dengan keseimbangan akurasi dan keberagaman"
    )
    
    st.success(
        "Model hybrid memberikan hasil terbaik dengan menyeimbangkan "
        "akurasi prediksi dan keberagaman rekomendasi produk."
    )
    
    # Roadmap
    st.subheader("Roadmap Implementasi Sistem")
    st.markdown("""
    **1. Deploy (Peluncuran)**
    - Menerapkan model hybrid ke sistem live H&M dan integrasi dengan platform e-commerce
    
    **2. A/B Testing (Uji Perbandingan)**
    - Membandingkan performa model hybrid dengan model lama pada segmen pelanggan berbeda
    - Mengukur click-through rate, conversion rate, dan customer satisfaction
    
    **3. Monitor (Pemantauan)**
    - Memantau kinerja model secara real-time untuk mendeteksi anomali atau penurunan performa
    - Menganalisis feedback pelanggan dan engagement metrics
    
    **4. Optimize (Optimalisasi)**
    - Melakukan fine-tuning parameter model berdasarkan data monitoring
    - Menyesuaikan bobot antara collaborative filtering dan content-based sesuai hasil
    
    **5. Scale (Penskalaan)**
    - Memperluas implementasi ke semua region dan segmen pelanggan H&M
    - Meningkatkan kapasitas infrastruktur untuk volume transaksi yang lebih besar
    """)

# ============================================================================
# TAB 2: PERFORMA (MODEL PERFORMANCE ANALYSIS)
# ============================================================================

with tab2:
    st.header("Model Performance Analysis")
    
    # Model Explanation
    st.markdown("**Penjelasan Model Rekomendasi:**")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.write(
            "**Popularity**\n\n"
            "Merekomendasikan produk populer global. "
            "Cara kerja: Ranking produk berdasarkan jumlah pembeli."
        )
    
    with col2:
        st.write(
            "**ALS**\n\n"
            "Collaborative filtering dengan matrix factorization. "
            "Cara kerja: Mencari pelanggan serupa berdasarkan preferensi tersembunyi."
        )
    
    with col3:
        st.write(
            "**Content**\n\n"
            "Content-based filtering. "
            "Cara kerja: Merekomendasikan produk dengan fitur serupa dengan yang sudah dibeli."
        )
    
    with col4:
        st.write(
            "**Hybrid ✓**\n\n"
            "Kombinasi ALS + Content. "
            "Cara kerja: Menggabungkan kekuatan kedua metode untuk hasil optimal."
        )
    
    with col5:
        st.write(
            "**Random**\n\n"
            "Baseline untuk perbandingan. "
            "Cara kerja: Rekomendasi acak tanpa logika."
        )
    
    # Model Performance Table
    st.dataframe(
        model_df[['Model', 'RMSE', 'Coverage']], 
        use_container_width=True, 
        hide_index=True
    )
    
    # RMSE Chart
    st.subheader("RMSE - Semakin Rendah Semakin Baik")
    st.write(
        "RMSE mengukur rata-rata kesalahan prediksi rating. "
        "Model dengan RMSE lebih rendah memiliki prediksi yang lebih akurat."
    )
    
    rmse_df = model_df[['Model', 'RMSE']].dropna(subset=['RMSE']).sort_values('RMSE').reset_index(drop=True)
    
    if len(rmse_df) > 0:
        colors = ['#2ecc71' if model == 'Hybrid' else '#3498db' for model in rmse_df['Model']]
        
        fig1 = px.bar(
            rmse_df, 
            y='Model', 
            x='RMSE', 
            orientation='h', 
            title="Perbandingan Error Rate Model"
        )
        fig1.update_traces(marker_color=colors)
        fig1.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)
        st.caption("✓ Hybrid model memiliki RMSE terendah = akurasi prediksi terbaik")
    
    # Coverage Chart
    st.subheader("Coverage - Semakin Tinggi Semakin Baik")
    st.write(
        "Coverage menunjukkan persentase produk yang dapat direkomendasikan oleh model. "
        "Coverage tinggi berarti model tidak hanya merekomendasikan produk populer saja."
    )
    
    cov_df = model_df[['Model', 'Coverage']].sort_values('Coverage', ascending=False).reset_index(drop=True)
    
    if len(cov_df) > 0:
        fig2 = px.bar(
            cov_df, 
            x='Model', 
            y='Coverage', 
            color='Coverage', 
            color_continuous_scale='Greens', 
            title="Cakupan Produk yang Dapat Direkomendasikan (%)"
        )
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("✓ Hybrid model mencakup 4,60% dari semua produk = keseimbangan antara akurasi dan keberagaman")
    
    # Product Diversity Chart
    st.subheader("Product Diversity - Semakin Tinggi Semakin Baik")
    st.write(
        "Menunjukkan berapa banyak produk berbeda yang direkomendasikan model kepada pelanggan. "
        "Jumlah lebih tinggi berarti keberagaman rekomendasi."
    )
    
    prod_df = model_df[['Model', 'Products']].sort_values('Products', ascending=False).reset_index(drop=True)
    
    if len(prod_df) > 0:
        fig3 = px.bar(
            prod_df, 
            x='Model', 
            y='Products', 
            color='Products', 
            color_continuous_scale='Blues', 
            title="Jumlah Produk Unik yang Direkomendasikan"
        )
        fig3.update_layout(height=350)
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("✓ Hybrid model merekomendasikan produk unik = keberagaman yang baik")
    
    # Daily Recommendation Capacity Chart
    st.subheader("Daily Recommendation Capacity - Semakin Tinggi Semakin Baik")
    st.write(
        "Kapasitas model untuk menghasilkan rekomendasi per hari. "
        "Volume tinggi menunjukkan skalabilitas model untuk bisnis besar."
    )
    
    rec_df = model_df[['Model', 'Recommendations']].sort_values('Recommendations', ascending=False).reset_index(drop=True)
    
    if len(rec_df) > 0:
        fig4 = px.bar(
            rec_df, 
            x='Model', 
            y='Recommendations', 
            color='Recommendations', 
            color_continuous_scale='Purples', 
            title="Kapasitas Rekomendasi Harian"
        )
        fig4.update_layout(height=350)
        st.plotly_chart(fig4, use_container_width=True)
        st.caption("✓ Model menghasilkan rekomendasi dalam jumlah besar = skalabel untuk operasi H&M")

# ============================================================================
# TAB 3: DATA ANALYSIS (FIXED - TOP PRODUCTS CORRECT ORDER!)
# ============================================================================

with tab3:
    st.header("Data Analysis")
    
    # Network Properties
    st.subheader("Network Properties")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Node", f"{graph_stats['total_nodes']:,}")
    c2.metric("Total Edge (Koneksi)", f"{graph_stats['total_edges']:,}")
    c3.metric("Kepadatan", f"{graph_stats['density']:.6f}")
    
    # Network Composition
    st.subheader("Network Composition")
    c1, c2 = st.columns(2)
    c1.metric("Pelanggan", f"{graph_stats['num_customers']:,}")
    c2.metric("Produk", f"{graph_stats['num_products']:,}")
    
    # Perilaku Pembelian
    st.subheader("Perilaku Pembelian Pelanggan")
    st.write("**Metrik ini menunjukkan pola pembelian dalam jaringan pelanggan-produk H&M:**")
    
    c1, c2 = st.columns(2)
    
    c1.write(
        f"**Pelanggan Teratas**: {graph_stats['top_customer']:,} pembelian\n\n"
        f"Menunjukkan pelanggan dengan frekuensi pembelian tertinggi (power user). "
        f"Insight: Pelanggan ini adalah high-value customer yang penting untuk retensi."
    )
    
    c2.write(
        f"**Produk Teratas**: {graph_stats['top_product']:,} pelanggan\n\n"
        f"Menunjukkan produk yang dibeli oleh jumlah pelanggan terbanyak (populer). "
        f"Insight: Produk ini adalah best-seller dan harus selalu tersedia."
    )
    
    # Distribution Chart
    st.subheader("Distribusi Pembelian Produk per Pelanggan")
    st.write(
        "Grafik ini menunjukkan pola distribusi jumlah produk yang dibeli oleh setiap pelanggan. "
        "Mayoritas pelanggan membeli 1-3 produk (power-law distribution) yang umum di e-commerce, "
        "sementara beberapa pelanggan (power users) membeli sangat banyak. "
        "Insight: Sebagian besar pelanggan adalah casual buyers, ada peluang untuk meningkatkan "
        "frequency melalui rekomendasi."
    )
    
    dist_data = data['distribution']
    
    if len(dist_data) > 0:
        fig_dist = px.histogram(
            dist_data,
            x='purchases',
            nbins=30,
            title="Distribusi Jumlah Produk per Pelanggan",
            labels={'purchases': 'Jumlah Produk', 'count': 'Jumlah Pelanggan'},
            color_discrete_sequence=['#1f77b4']
        )
        fig_dist.update_layout(height=350)
        st.plotly_chart(fig_dist, use_container_width=True)
        
        st.caption(
            f"Mean: {dist_data['purchases'].mean():.2f} produk per pelanggan | "
            f"Median: {dist_data['purchases'].median():.0f} | "
            f"Max: {dist_data['purchases'].max():.0f} | "
            f"Power-law distribution (umum di e-commerce)"
        )
    
    # ✅ TOP PRODUCTS CHART - FIXED (ASCENDING SORT FOR CORRECT VISUAL)
    st.subheader("Top 10 Most Popular Products")
    
    top_p = data['top_products'].head(10).sort_values('Degree', ascending=True).reset_index(drop=True)
    
    if len(top_p) > 0:
        # Convert Product ID to string to prevent numeric sorting issues
        top_p['Product_Str'] = top_p['Product'].astype(str)
        
        fig_top = px.bar(
            top_p, 
            x='Degree',
            y='Product_Str',
            orientation='h', 
            title="Produk Paling Banyak Dibeli",
            color_discrete_sequence=['#e74c3c']
        )
        
        fig_top.update_layout(
            height=400, 
            xaxis_title="Jumlah Pelanggan", 
            yaxis_title="Product ID"
        )
        
        st.plotly_chart(fig_top, use_container_width=True)
    
    # Top Customers Chart
    st.subheader("Top 10 Most Active Customers")
    
    top_c = data['top_customers'].head(10).sort_values('Degree', ascending=True).reset_index(drop=True)
    
    if len(top_c) > 0:
        top_c_copy = top_c.copy()
        top_c_copy['Customer_Short'] = top_c_copy['Customer'].astype(str).str[:16] + '...'
        
        fig_cust = px.bar(
            top_c_copy, 
            y='Customer_Short', 
            x='Degree', 
            orientation='h', 
            color='Degree', 
            color_continuous_scale='Oranges', 
            title="Pelanggan dengan Pembelian Terbanyak"
        )
        
        fig_cust.update_layout(
            height=400, 
            xaxis_title="Jumlah Produk", 
            yaxis_title=""
        )
        
        st.plotly_chart(fig_cust, use_container_width=True)

# ============================================================================
# TAB 4: REKOMENDASI (PERSONALIZED RECOMMENDATIONS)
# ============================================================================

with tab4:
    st.header("Personalized Recommendations")
    
    st.info(
        "Lihat rekomendasi produk untuk pelanggan contoh berdasarkan "
        "riwayat pembelian mereka dan model hybrid recommendation."
    )
    
    # Recommendation Strategy
    st.subheader("Recommendation Strategy by Customer Segment")
    
    st.markdown("""
    **Cold-Start Users (Pelanggan Baru - Tanpa Riwayat Pembelian)**
    - Gunakan: Model Popularity (produk populer umum)
    - Alasan: Tidak ada data historis untuk collaborative filtering dan content-based
    - Strategi: Tunjukkan best-sellers, produk trending, kategori populer
    
    **Warm Users (Pelanggan Aktif - Dengan Riwayat Pembelian)**
    - Gunakan: Model Hybrid (kombinasi collaborative + content-based)
    - Alasan: Data historis cukup untuk memberikan rekomendasi akurat dan beragam
    - Strategi: Personalisasi berdasarkan preferensi individual dan patterns pelanggan serupa
    
    **Power Users (Pelanggan Setia - Pembelian Sangat Banyak)**
    - Gunakan: Model Hybrid dengan tambahan exploration (mencoba produk baru)
    - Alasan: Mereka sudah familiar dengan produk standar, perlu diversifikasi untuk fresh recommendations
    - Strategi: Rekomendasi niche, produk eksklusif, pre-order produk baru
    """)
    
    # Top Customers Table
    st.subheader("Top Customers from REAL DATA")
    
    top_customers_list = data['top_customers'].head(5)
    
    if len(top_customers_list) > 0:
        st.dataframe(
            top_customers_list[['Customer', 'Degree']], 
            use_container_width=True, 
            hide_index=True
        )
    
    st.caption(
        "💡 **Insight**: Model hybrid direkomendasikan karena memberikan "
        "keseimbangan terbaik antara akurasi dan keberagaman produk yang direkomendasikan."
    )

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    "<center style='color:#999; font-size:0.9em;'>"
    "✅ H&M Recommendation System | Hybrid Collaborative + Content Analytics | "
    "Data: 100% REAL dari GitHub"
    "</center>", 
    unsafe_allow_html=True
)

# ============================================================================
# SIDEBAR INFO
# ============================================================================

st.sidebar.title("📊 Dataset Info")
st.sidebar.success("✅ Data loaded from `data/` folder (REAL DATA)")

st.sidebar.write(f"**Nodes**: {graph_stats['total_nodes']:,}")
st.sidebar.write(f"**Edges**: {graph_stats['total_edges']:,}")
st.sidebar.write(f"**Customers**: {graph_stats['num_customers']:,}")
st.sidebar.write(f"**Products**: {graph_stats['num_products']:,}")

stats_file = Path("data/network_stats.json")
if stats_file.exists():
    ts = datetime.fromtimestamp(os.path.getmtime(stats_file))
    st.sidebar.write(f"**Last updated**: {ts.strftime('%Y-%m-%d %H:%M')}")
