import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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

st.markdown("# H&M Recommendation System")
st.markdown("**Hybrid Collaborative Filtering + Content-Based Analytics Dashboard**")
st.markdown("---")

DATA_DIR = Path("data")

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    return pd.read_csv(path)

@st.cache_data
def load_json(name: str):
    path = DATA_DIR / name
    with open(path, "r") as f:
        return json.load(f)

@st.cache_data
def load_all_data():
    top_customers = load_csv("top_customers.csv")
    top_products = load_csv("top_products.csv")
    distribution = load_csv("customer_distribution.csv")
    edges = load_csv("bipartite_edges.csv")
    nodes = load_csv("bipartite_nodes.csv")
    network_stats = load_json("network_stats.json")
    model_performance = load_json("model_performance.json")
    sample_recs = load_csv("sample_recommendations.csv")
    article_map = load_csv("article_mapping.csv")

    # join rekomendasi dengan nama produk
    # article_mapping.csv: article_id, prod_name, product_type_name, product_group_name
    if "article_id" in article_map.columns:
        sample_recs_named = sample_recs.merge(
            article_map,
            how="left",
            on="article_id"
        )
    else:
        sample_recs_named = sample_recs.copy()

    return {
        "top_customers": top_customers,
        "top_products": top_products,
        "distribution": distribution,
        "edges": edges,
        "nodes": nodes,
        "network_stats": network_stats,
        "model_performance": model_performance,
        "sample_recs": sample_recs_named
    }

# ============================================================================
# SAFE CONVERSIONS
# ============================================================================

def safe_int(v, default=0):
    try:
        return int(v)
    except Exception:
        return default

def safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default

# ============================================================================
# LOAD DATA
# ============================================================================

data = load_all_data()

# ============================================================================
# EXTRACT NETWORK STATS
# ============================================================================

ns = data["network_stats"]
graph_stats = {
    "total_nodes": safe_int(ns.get("total_nodes", 27542), 27542),
    "total_edges": safe_int(ns.get("total_edges", 150680), 150680),
    "num_customers": safe_int(ns.get("num_customers", 3000), 3000),
    "num_products": safe_int(ns.get("num_products", 24542), 24542),
    "density": safe_float(ns.get("density", 0.000397), 0.000397),
    "top_customer": safe_int(ns.get("top_customer", 407), 407),
    "top_product": safe_int(ns.get("top_product", 102), 102)
}

# ============================================================================
# EXTRACT MODEL PERFORMANCE
# ============================================================================

mp = data["model_performance"]
model_rows = []

for model_name, metrics in mp.items():
    rmse_val = metrics.get("RMSE", "N/A")
    cov_val = metrics.get("Coverage", 0)

    if isinstance(rmse_val, str) and rmse_val == "N/A":
        rmse_float = np.nan
    else:
        rmse_float = safe_float(rmse_val, np.nan)

    cov_float = safe_float(cov_val, 0.0)

    model_rows.append(
        {
            "Model": model_name,
            "RMSE": rmse_float,
            "Coverage": cov_float,
        }
    )

model_df = pd.DataFrame(model_rows)

# Tambah kolom Products & Recommendations (hardcoded dari analisis)
products_map = {
    "Random": 39498,
    "Popularity": 39498,
    "ALS": 1601,
    "Content": 3259,
    "Hybrid": 4860,
}
recs_map = {
    "Random": 1401061,
    "Popularity": 1401061,
    "ALS": 6050980,
    "Content": 12799,
    "Hybrid": 6063779,
}
model_df["Products"] = model_df["Model"].map(products_map).fillna(0).astype(int)
model_df["Recommendations"] = model_df["Model"].map(recs_map).fillna(0).astype(int)

# ============================================================================
# DATASET METRICS (STATIC SUMMARY)
# ============================================================================

dataset_metrics = {
    "total_interactions": 7005582,
    "unique_customers": 742431,
    "unique_products": 51232,
    "train_set": 5604521,
    "test_set": 1401061,
}

# ============================================================================
# TABS
# ============================================================================

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Ringkasan", "🎯 Performa", "📈 Data", "💡 Rekomendasi"]
)

# ============================================================================
# TAB 1: RINGKASAN
# ============================================================================

with tab1:
    st.header("Executive Summary")

    c1, c2 = st.columns(2)
    with c1:
        st.info(
            "H&M mengelola lebih dari 7 juta interaksi pelanggan dengan "
            "51.232 produk dalam periode 6 bulan. Sistem rekomendasi hybrid "
            "menggabungkan collaborative filtering dan content-based "
            "untuk memberikan rekomendasi yang relevan dan beragam."
        )
    with c2:
        st.success(
            "Model hybrid mampu menghasilkan jutaan rekomendasi per hari, "
            "mencakup ribuan produk unik, dan dirancang untuk meningkatkan "
            "retensi dan nilai seumur hidup pelanggan."
        )

    st.subheader("Dataset Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Transaksi", f"{dataset_metrics['total_interactions']:,}")
    c2.metric("Pelanggan", f"{dataset_metrics['unique_customers']:,}")
    c3.metric("Produk", f"{dataset_metrics['unique_products']:,}")
    c4.metric("Data Latih", f"{dataset_metrics['train_set']:,}")
    c5.metric("Data Uji", f"{dataset_metrics['test_set']:,}")

    st.subheader("Model Comparison")
    col1, col2, col3 = st.columns(3)

    als_row = model_df[model_df["Model"] == "ALS"]
    content_row = model_df[model_df["Model"] == "Content"]
    hybrid_row = model_df[model_df["Model"] == "Hybrid"]

    als_rmse = als_row["RMSE"].values[0] if len(als_row) else np.nan
    content_rmse = content_row["RMSE"].values[0] if len(content_row) else np.nan
    hybrid_rmse = hybrid_row["RMSE"].values[0] if len(hybrid_row) else np.nan

    als_rmse_str = f"{als_rmse:.4f}" if not np.isnan(als_rmse) else "N/A"
    content_rmse_str = f"{content_rmse:.4f}" if not np.isnan(content_rmse) else "N/A"
    hybrid_rmse_str = f"{hybrid_rmse:.4f}" if not np.isnan(hybrid_rmse) else "N/A"

    col1.write(
        f"**ALS (Collaborative Filtering)**\n\n"
        f"RMSE: {als_rmse_str}\n\n"
        f"Menggunakan pola kesamaan perilaku antar pelanggan."
    )
    col2.write(
        f"**Content-Based**\n\n"
        f"RMSE: {content_rmse_str}\n\n"
        f"Merekomendasikan produk dengan karakteristik serupa."
    )
    col3.write(
        f"**Hybrid (ALS + Content) ✓**\n\n"
        f"RMSE: {hybrid_rmse_str}\n\n"
        f"Menggabungkan akurasi collaborative dengan fleksibilitas content-based."
    )

    st.subheader("Roadmap Implementasi")
    st.markdown(
        """
        **1. Deploy** – Integrasi model hybrid ke platform e-commerce H&M.  
        **2. A/B Testing** – Bandingkan dengan sistem rekomendasi lama pada segmen pelanggan yang berbeda.  
        **3. Monitoring** – Pantau CTR, conversion rate, dan retensi pelanggan.  
        **4. Optimization** – Fine-tuning hyperparameter dan bobot hybrid berdasarkan hasil A/B test.  
        **5. Scale Up** – Roll-out ke seluruh region dan channel penjualan."""
    )

# ============================================================================
# TAB 2: PERFORMA MODEL
# ============================================================================

with tab2:
    st.header("Model Performance Analysis")

    st.markdown("**Ringkasan Model:**")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.write("**Random**\n\nBaseline acak untuk pembanding.")
    c2.write("**Popularity**\n\nMerekomendasikan produk paling populer global.")
    c3.write("**ALS**\n\nCollaborative filtering berbasis matrix factorization.")
    c4.write("**Content-Based**\n\nBerdasarkan kemiripan fitur produk.")
    c5.write("**Hybrid**\n\nKombinasi ALS + Content untuk hasil seimbang.")

    st.dataframe(
        model_df[["Model", "RMSE", "Coverage"]],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("RMSE – Semakin Rendah Semakin Baik")
    rmse_df = model_df[["Model", "RMSE"]].dropna().sort_values("RMSE").reset_index(
        drop=True
    )
    if len(rmse_df):
        colors = [
            "#2ecc71" if m == "Hybrid" else "#3498db" for m in rmse_df["Model"]
        ]
        fig_rmse = px.bar(
            rmse_df,
            y="Model",
            x="RMSE",
            orientation="h",
            title="Error Rate per Model",
        )
        fig_rmse.update_traces(marker_color=colors)
        fig_rmse.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_rmse, use_container_width=True)

    st.subheader("Coverage Produk")
    cov_df = model_df[["Model", "Coverage"]].sort_values(
        "Coverage", ascending=False
    ).reset_index(drop=True)
    if len(cov_df):
        fig_cov = px.bar(
            cov_df,
            x="Model",
            y="Coverage",
            color="Coverage",
            color_continuous_scale="Greens",
            title="Persentase Produk yang Tercakup Rekomendasi",
        )
        fig_cov.update_layout(height=350)
        st.plotly_chart(fig_cov, use_container_width=True)

    st.subheader("Keberagaman Produk")
    prod_df = model_df[["Model", "Products"]].sort_values(
        "Products", ascending=False
    ).reset_index(drop=True)
    if len(prod_df):
        fig_prod = px.bar(
            prod_df,
            x="Model",
            y="Products",
            color="Products",
            color_continuous_scale="Blues",
            title="Jumlah Produk Unik yang Direkomendasikan",
        )
        fig_prod.update_layout(height=350)
        st.plotly_chart(fig_prod, use_container_width=True)

    st.subheader("Kapasitas Rekomendasi Harian")
    rec_df = model_df[["Model", "Recommendations"]].sort_values(
        "Recommendations", ascending=False
    ).reset_index(drop=True)
    if len(rec_df):
        fig_rec = px.bar(
            rec_df,
            x="Model",
            y="Recommendations",
            color="Recommendations",
            color_continuous_scale="Purples",
            title="Perkiraan Volume Rekomendasi Harian",
        )
        fig_rec.update_layout(height=350)
        st.plotly_chart(fig_rec, use_container_width=True)

# ============================================================================
# TAB 3: DATA ANALYSIS
# ============================================================================

with tab3:
    st.header("Data Analysis")

    st.subheader("Network Properties")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Node", f"{graph_stats['total_nodes']:,}")
    c2.metric("Total Edge", f"{graph_stats['total_edges']:,}")
    c3.metric("Kepadatan", f"{graph_stats['density']:.6f}")

    st.subheader("Network Composition")
    c1, c2 = st.columns(2)
    c1.metric("Pelanggan", f"{graph_stats['num_customers']:,}")
    c2.metric("Produk", f"{graph_stats['num_products']:,}")

    st.subheader("Perilaku Pembelian Pelanggan")
    c1, c2 = st.columns(2)
    c1.write(
        f"**Pelanggan Teratas**: {graph_stats['top_customer']:,} pembelian\n\n"
        f"Menunjukkan pelanggan dengan frekuensi pembelian tertinggi."
    )
    c2.write(
        f"**Produk Teratas**: {graph_stats['top_product']:,} pelanggan\n\n"
        f"Menunjukkan produk yang dibeli oleh jumlah pelanggan terbanyak."
    )

    st.subheader("Distribusi Jumlah Produk per Pelanggan")
    dist = data["distribution"]
    if len(dist):
        fig_dist = px.histogram(
            dist,
            x="purchases",
            nbins=30,
            title="Distribusi Pembelian per Pelanggan",
            labels={"purchases": "Jumlah Produk"},
            color_discrete_sequence=["#1f77b4"],
        )
        fig_dist.update_layout(height=350)
        st.plotly_chart(fig_dist, use_container_width=True)
        st.caption(
            f"Rata-rata: {dist['purchases'].mean():.2f} | "
            f"Median: {dist['purchases'].median():.0f} | "
            f"Max: {dist['purchases'].max():.0f}"
        )

    st.subheader("Top 10 Most Popular Products")
    top_p = (
        data["top_products"]
        .head(10)
        .sort_values("Degree", ascending=True)
        .reset_index(drop=True)
    )
    if len(top_p):
        top_p["Product_Str"] = top_p["Product"].astype(str)
        fig_top = px.bar(
            top_p,
            x="Degree",
            y="Product_Str",
            orientation="h",
            title="Produk Paling Banyak Dibeli",
            color_discrete_sequence=["#e74c3c"],
        )
        fig_top.update_layout(
            height=400,
            xaxis_title="Jumlah Pelanggan",
            yaxis_title="Product ID",
        )
        st.plotly_chart(fig_top, use_container_width=True)

    st.subheader("Top 10 Most Active Customers")
    top_c = (
        data["top_customers"]
        .head(10)
        .sort_values("Degree", ascending=True)
        .reset_index(drop=True)
    )
    if len(top_c):
        top_c_copy = top_c.copy()
        top_c_copy["Customer_Short"] = (
            top_c_copy["Customer"].astype(str).str[:16] + "..."
        )
        fig_c = px.bar(
            top_c_copy,
            y="Customer_Short",
            x="Degree",
            orientation="h",
            color="Degree",
            color_continuous_scale="Oranges",
            title="Pelanggan dengan Pembelian Terbanyak",
        )
        fig_c.update_layout(
            height=400,
            xaxis_title="Jumlah Produk",
            yaxis_title="",
        )
        st.plotly_chart(fig_c, use_container_width=True)

# ============================================================================
# TAB 4: REKOMENDASI
# ============================================================================

with tab4:
    st.header("Personalized Recommendations (Sample)")

    st.info(
        "Bagian ini menampilkan contoh rekomendasi produk berbasis model ALS "
        "untuk subset pelanggan. Nama produk dan kategori diambil dari data artikel H&M."
    )

    # --- DATA YANG DIBUTUHKAN ---
    sample_recs = data["sample_recs"]  # sudah join dengan article_mapping di load_all_data()
    edges = load_csv("bipartite_edges.csv")  # history graph customer–product
    article_map = load_csv("article_mapping.csv")

    # --- PILIH CUSTOMER ---
    unique_customers = sample_recs["customer_id"].unique()
    selected_customer = st.selectbox(
        "Pilih contoh customer",
        sorted(unique_customers),
        index=0 if len(unique_customers) else None,
    )

    # --- PILIH TOP-N ---
    top_n = st.slider("Jumlah rekomendasi yang ditampilkan", 5, 30, 10)

    # --- REKOMENDASI UNTUK CUSTOMER TERPILIH ---
    cust_recs = (
        sample_recs[sample_recs["customer_id"] == selected_customer]
        .sort_values("rank")
        .head(top_n)
    )

    # Konversi score menjadi label confidence
    def score_to_band(s):
        try:
            s_val = float(s)
        except Exception:
            return "Unknown"
        if s_val >= 10:
            return "Very High"
        elif s_val >= 7:
            return "High"
        elif s_val >= 4:
            return "Medium"
        else:
            return "Exploratory"

    if len(cust_recs):
        cust_recs_display = cust_recs.copy()
        cust_recs_display["confidence"] = cust_recs_display["score"].apply(score_to_band)

        st.subheader("Top‑N Rekomendasi Produk untuk Customer Terpilih")

        display_cols = []
        if "rank" in cust_recs_display.columns:
            display_cols.append("rank")
        display_cols.append("article_id")
        for col in ["prod_name", "product_type_name", "product_group_name"]:
            if col in cust_recs_display.columns:
                display_cols.append(col)
        if "score" in cust_recs_display.columns:
            display_cols.append("score")
        if "confidence" in cust_recs_display.columns:
            display_cols.append("confidence")

        st.dataframe(
            cust_recs_display[display_cols],
            use_container_width=True,
        )
    else:
        st.warning("Tidak ada rekomendasi untuk customer ini di sample.")

    # --- HISTORY PEMBELIAN CUSTOMER (DARI GRAPH) ---
    st.subheader("Riwayat Produk yang Pernah Dibeli (sample)")

    # edges.csv di notebook kamu harus punya kolom customer_id & article_id
    cust_hist = edges[edges["customer_id"] == selected_customer].copy()
    if len(cust_hist):
        cust_hist = cust_hist.merge(article_map, how="left", on="article_id")

        hist_display = (
            cust_hist[
                [
                    "article_id",
                    "prod_name",
                    "product_type_name",
                    "product_group_name",
                ]
            ]
            .drop_duplicates()
            .head(20)
        )

        st.dataframe(
            hist_display,
            use_container_width=True,
        )

        # highlight category terbanyak di history dan di rekomendasi
        if "product_group_name" in cust_hist.columns:
            top_hist_group = (
                cust_hist["product_group_name"]
                .value_counts()
                .head(1)
                .index[0]
            )
            st.caption(
                f"Kategori yang paling sering dibeli (berdasarkan sample graph): "
                f"**{top_hist_group}**."
            )

        if len(cust_recs) and "product_group_name" in cust_recs.columns:
            top_rec_group = (
                cust_recs["product_group_name"]
                .value_counts()
                .head(1)
                .index[0]
            )
            st.caption(
                f"Kategori yang paling banyak direkomendasikan: "
                f"**{top_rec_group}**."
            )
    else:
        st.caption(
            "Belum ada history transaksi di sample graph untuk customer ini "
            "(atau tidak termasuk dalam subset network yang diekspor)."
        )

    st.subheader("Strategi Rekomendasi per Segmen Pelanggan")
    st.markdown(
        """
        **Pelanggan Baru (Cold-Start)**  
        - Gunakan model Popularity dan kampanye general.  

        **Pelanggan Aktif (Warm Users)**  
        - Gunakan model Hybrid untuk rekomendasi personal berdasarkan histori dan kemiripan produk.  

        **Pelanggan Loyal (Power Users)**  
        - Kombinasikan rekomendasi personal dengan eksplorasi produk baru dan eksklusif.
        """
    )


# ============================================================================
# SIDEBAR & FOOTER
# ============================================================================

st.sidebar.title("📊 Dataset Info")
st.sidebar.success("Data di-load dari folder `data/` (output Kaggle).")
st.sidebar.write(f"Nodes: {graph_stats['total_nodes']:,}")
st.sidebar.write(f"Edges: {graph_stats['total_edges']:,}")
st.sidebar.write(f"Customers: {graph_stats['num_customers']:,}")
st.sidebar.write(f"Products: {graph_stats['num_products']:,}")

stats_file = DATA_DIR / "network_stats.json"
if stats_file.exists():
    ts = datetime.fromtimestamp(stats_file.stat().st_mtime)
    st.sidebar.write(f"Last updated: {ts.strftime('%Y-%m-%d %H:%M')}")

st.markdown("---")
st.markdown(
    "<center style='color:#999; font-size:0.9em;'>"
    "H&M Recommendation System | Hybrid Collaborative + Content-Based Analytics"
    "</center>",
    unsafe_allow_html=True,
)

