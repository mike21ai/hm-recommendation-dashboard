import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime
import os

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="H&M Recommendation System Dashboard",
    page_icon="🛍️",
    layout="wide"
)

# -----------------------------------------------------------------------------
# DATA LOADING (FROM data/ FOLDER)
# -----------------------------------------------------------------------------

@st.cache_data
def load_data():
    data_dir = Path("data")

    top_customers = pd.read_csv(data_dir / "top_customers.csv")
    top_products = pd.read_csv(data_dir / "top_products.csv")
    distribution = pd.read_csv(data_dir / "customer_distribution.csv")
    edges = pd.read_csv(data_dir / "bipartite_edges.csv")
    nodes = pd.read_csv(data_dir / "bipartite_nodes.csv")

    with open(data_dir / "network_stats.json", "r") as f:
        network_stats = json.load(f)

    with open(data_dir / "model_performance.json", "r") as f:
        model_performance = json.load(f)

    return {
        "top_customers": top_customers,
        "top_products": top_products,
        "distribution": distribution,
        "edges": edges,
        "nodes": nodes,
        "network_stats": network_stats,
        "model_performance": model_performance,
    }


try:
    data = load_data()
except FileNotFoundError as e:
    st.error(
        f"""
    ❌ Data file not found: `{e.filename}`

    Pastikan semua file ini ada di folder `data/`:
    - top_customers.csv
    - top_products.csv
    - network_stats.json
    - customer_distribution.csv
    - bipartite_edges.csv
    - bipartite_nodes.csv
    - model_performance.json
    """
    )
    st.stop()

# Sidebar info
st.sidebar.title("Dataset Info")
st.sidebar.success("✅ Data loaded from `data/` folder")

net = data["network_stats"]
st.sidebar.write(f"**Nodes**: {net['total_nodes']:,}")
st.sidebar.write(f"**Edges**: {net['total_edges']:,}")
st.sidebar.write(f"**Customers (graph)**: {net['num_customers']:,}")
st.sidebar.write(f"**Products (graph)**: {net['num_products']:,}")

stats_file = Path("data/network_stats.json")
if stats_file.exists():
    ts = datetime.fromtimestamp(os.path.getmtime(stats_file))
    st.sidebar.write(f"**Last updated**: {ts.strftime('%Y-%m-%d %H:%M')}")

# -----------------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------------
st.title("🛍️ H&M E‑Commerce Recommendation System")
st.markdown(
    "Big Data Project · Collaborative Filtering · Content-Based · Graph Analytics"
)
st.divider()

# -----------------------------------------------------------------------------
# NETWORK OVERVIEW
# -----------------------------------------------------------------------------
st.header("📊 Network Overview")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Nodes", f"{net['total_nodes']:,}", "Customers + Products")
with col2:
    st.metric("Total Edges", f"{net['total_edges']:,}", "Purchase links")
with col3:
    st.metric("Density", f"{net['density']:.6f}")
with col4:
    st.metric("Clustering Coefficient", f"{net['clustering_coefficient']:.4f}")

st.divider()

# -----------------------------------------------------------------------------
# MODEL PERFORMANCE
# -----------------------------------------------------------------------------
st.header("🤖 Model Performance Comparison")

mp = data["model_performance"]

col1, col2 = st.columns(2)

with col1:
    # RMSE
    models_rmse = []
    rmse_vals = []
    for m, v in mp.items():
        if v["RMSE"] != "N/A":
            models_rmse.append(m)
            rmse_vals.append(v["RMSE"])

    fig_rmse = go.Figure(
        data=[
            go.Bar(
                x=models_rmse,
                y=rmse_vals,
                text=[f"{v:.4f}" for v in rmse_vals],
                textposition="auto",
                marker_color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"],
            )
        ]
    )
    fig_rmse.update_layout(
        title="RMSE Comparison (Lower is Better)",
        xaxis_title="Model",
        yaxis_title="RMSE",
        height=400,
        showlegend=False,
    )
    st.plotly_chart(fig_rmse, use_container_width=True)

with col2:
    # Coverage
    models_cov = list(mp.keys())
    cov_vals = [mp[m]["Coverage"] for m in models_cov]

    fig_cov = go.Figure(
        data=[
            go.Bar(
                x=models_cov,
                y=cov_vals,
                text=[f"{v:.2f}%" for v in cov_vals],
                textposition="auto",
                marker_color=["#F38181", "#FCEA7E", "#95E1D3", "#AA96DA", "#FCBAD3"],
            )
        ]
    )
    fig_cov.update_layout(
        title="Product Coverage (%)",
        xaxis_title="Model",
        yaxis_title="Coverage %",
        height=400,
        showlegend=False,
    )
    st.plotly_chart(fig_cov, use_container_width=True)

st.subheader("📋 Detailed Metrics")
perf_df = pd.DataFrame(mp).T.reset_index().rename(columns={"index": "Model"})
st.dataframe(perf_df, use_container_width=True, hide_index=True)

st.divider()

# -----------------------------------------------------------------------------
# TOP CUSTOMERS & PRODUCTS
# -----------------------------------------------------------------------------
st.header("🏆 Top Customers & Products")

tc = data["top_customers"].copy()
tp = data["top_products"].copy()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Top 10 Customers (by degree)")
    tc["Customer_short"] = tc["Customer"].str[:16] + "..."
    fig_tc = px.bar(
        tc,
        x="Degree",
        y="Customer_short",
        orientation="h",
        color="Degree",
        color_continuous_scale="Blues",
        labels={"Degree": "Products Purchased", "Customer_short": "Customer"},
        title="Most Connected Customers",
    )
    fig_tc.update_layout(
        yaxis={"categoryorder": "total ascending"}, height=450, showlegend=False
    )
    st.plotly_chart(fig_tc, use_container_width=True)
    st.expander("Show raw table").dataframe(tc, use_container_width=True)

with col2:
    st.subheader("Top 10 Products (by degree)")
    fig_tp = px.bar(
        tp,
        x="Degree",
        y="Product",
        orientation="h",
        color="Degree",
        color_continuous_scale="Oranges",
        labels={"Degree": "Customers", "Product": "Product"},
        title="Most Popular Products",
    )
    fig_tp.update_layout(
        yaxis={"categoryorder": "total ascending"}, height=450, showlegend=False
    )
    st.plotly_chart(fig_tp, use_container_width=True)
    st.expander("Show raw table").dataframe(tp, use_container_width=True)

st.divider()

# -----------------------------------------------------------------------------
# CUSTOMER PURCHASE DISTRIBUTION
# -----------------------------------------------------------------------------
st.header("📈 Customer Purchase Distribution")

dist = data["distribution"]

col1, col2 = st.columns([2, 1])

with col1:
    fig_dist = px.histogram(
        dist,
        x="purchases",
        nbins=50,
        color_discrete_sequence=["#667eea"],
        title="Histogram of Customer Purchases",
        labels={"purchases": "Number of Products", "count": "Number of Customers"},
    )
    fig_dist.update_layout(height=450, showlegend=False)
    st.plotly_chart(fig_dist, use_container_width=True)

with col2:
    st.subheader("Summary Stats")
    st.metric("Mean", f"{dist['purchases'].mean():.2f}")
    st.metric("Median", f"{dist['purchases'].median():.0f}")
    st.metric("Max", f"{dist['purchases'].max():.0f}")
    st.metric("Std Dev", f"{dist['purchases'].std():.2f}")
    st.metric("Customers (sample)", f"{len(dist):,}")

st.divider()

# -----------------------------------------------------------------------------
# BIPARTITE GRAPH
# -----------------------------------------------------------------------------
st.header("🕸️ Customer–Product Network (Sample)")

st.info(
    """
**Bipartite graph sample**

- 🔵 Blue nodes = Customers  
- 🟠 Orange nodes = Products  
- Node size ∝ degree (number of connections)  
- Data: 30 top customers & 6,000+ connected products (8,753 edges)  
"""
)

nodes_df = data["nodes"]
edges_df = data["edges"]


def create_bipartite_graph(nodes_df, edges_df):
    customers = nodes_df[nodes_df["type"] == "customer"].copy()
    products = nodes_df[nodes_df["type"] == "product"].copy()

    # Simple 2-layer layout: customers left (x=0), products right (x=2)
    pos = {}
    cust_spacing = 100 / max(len(customers), 1)
    for i, (_, row) in enumerate(customers.iterrows()):
        pos[row["id"]] = (0, i * cust_spacing)

    prod_spacing = 100 / max(len(products), 1)
    for i, (_, row) in enumerate(products.iterrows()):
        pos[row["id"]] = (2, i * prod_spacing)

    # Edges
    edge_x, edge_y = [], []
    for _, e in edges_df.iterrows():
        if e["source"] in pos and e["target"] in pos:
            x0, y0 = pos[e["source"]]
            x1, y1 = pos[e["target"]]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(color="rgba(120,120,120,0.3)", width=0.4),
        hoverinfo="none",
        showlegend=False,
    )

    # Customer nodes
    cust_x = [pos[id][0] for id in customers["id"]]
    cust_y = [pos[id][1] for id in customers["id"]]
    cust_sizes = [min(d / 3 + 8, 30) for d in customers["degree"]]

    cust_trace = go.Scatter(
        x=cust_x,
        y=cust_y,
        mode="markers",
        name="Customers",
        hoverinfo="text",
        text=[
            f"Customer: {row['label']}<br>Degree: {row['degree']}"
            for _, row in customers.iterrows()
        ],
        marker=dict(
            size=cust_sizes,
            color="#4299E1",
            line=dict(width=1, color="white"),
            opacity=0.9,
        ),
    )

    # Product nodes
    prod_x = [pos[id][0] for id in products["id"]]
    prod_y = [pos[id][1] for id in products["id"]]
    prod_sizes = [min(d / 1.5 + 5, 20) for d in products["degree"]]

    prod_trace = go.Scatter(
        x=prod_x,
        y=prod_y,
        mode="markers",
        name="Products",
        hoverinfo="text",
        text=[
            f"Product: {row['label']}<br>Degree: {row['degree']}"
            for _, row in products.iterrows()
        ],
        marker=dict(
            size=prod_sizes,
            color="#F56565",
            line=dict(width=1, color="white"),
            opacity=0.8,
        ),
    )

    fig = go.Figure(
        data=[edge_trace, cust_trace, prod_trace],
        layout=go.Layout(
            title=dict(
                text="Customer–Product Bipartite Network (Top Customers Sample)",
                x=0.5,
                xanchor="center",
            ),
            height=800,
            showlegend=True,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white",
            paper_bgcolor="white",
            legend=dict(orientation="h", x=0.5, y=1.02, xanchor="center"),
        ),
    )
    return fig


with st.spinner("Rendering network graph..."):
    fig_graph = create_bipartite_graph(nodes_df, edges_df)
    st.plotly_chart(fig_graph, use_container_width=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Nodes in Sample Graph", f"{len(nodes_df):,}")
with col2:
    st.metric("Edges in Sample Graph", f"{len(edges_df):,}")
with col3:
    n_cust = len(nodes_df[nodes_df["type"] == "customer"])
    n_prod = len(nodes_df[nodes_df["type"] == "product"])
    st.metric("Customers : Products", f"{n_cust} : {n_prod}")

st.divider()

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
<div style='text-align: center'>
    <p><strong>H&M Big Data Recommendation System Dashboard</strong></p>
    <p>Collaborative Filtering · Content-Based · Graph Analytics</p>
    <p>Data: Kaggle H&M Dataset · 7M+ interactions</p>
</div>
""",
    unsafe_allow_html=True,
)
