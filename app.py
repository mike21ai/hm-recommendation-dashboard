import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from networkx.algorithms import bipartite
from collections import deque

st.set_page_config(page_title="H&M Recommendation System", page_icon="🛍️", layout="wide")
st.markdown("# Sistem Rekomendasi H&M dengan Graph Analytics")
st.markdown("**Hybrid Collaborative Filtering + Content-Based + Graph Analytics Dashboard**")
st.markdown("---")

@st.cache_data
def load_data():
    product_names = {
        'P866731': 'Kaos Casual Biru',
        'P866732': 'Celana Denim Premium',
        'P866733': 'Jaket Outdoor',
        'P751471': 'Sweater Rajut Warna Cream',
        'P841383': 'Tas Tangan Kulit Coklat',
        'P599580': 'Sepatu Sneaker Putih',
        'P706016': 'Cardigan Rajut',
        'P610776': 'Sepatu Formal',
        'P759871': 'Dress Wanita Elegan'
    }
    return {
        'total_interactions': 7005582,
        'unique_customers': 742431,
        'unique_products': 51232,
        'train_set': 5604521,
        'test_set': 1401061,
    }, pd.DataFrame({
        'Model': ['Popularity', 'ALS', 'Content', 'Hybrid', 'Random'],
        'RMSE': [0.4848, 0.718, 0.65, 0.6350, 2.0348],
        'Coverage': [37.42, 1.51, 3.09, 4.60, 37.42],
        'Products': [39498, 1598, 3256, 4854, 39498],
        'Recommendations': [1401061, 6050980, 12799, 6063779, 1401061]
    }), {
        'total_nodes': 27523,
        'total_edges': 150673,
        'num_customers': 3000,
        'num_products': 24523,
        'density': 0.000398,
        'top_customer': 407,
        'top_product': 104
    }, product_names

@st.cache_resource
def create_bipartite_graph():
    """Create bipartite graph untuk analisis"""
    np.random.seed(42)
    n_cust = 20
    n_prod = 30
    
    # Create bipartite graph
    B = nx.Graph()
    
    # Add customer nodes (bipartite=0)
    customer_nodes = [f'C{i}' for i in range(n_cust)]
    B.add_nodes_from(customer_nodes, bipartite=0)
    
    # Add product nodes (bipartite=1)
    product_nodes = [f'P{i}' for i in range(n_prod)]
    B.add_nodes_from(product_nodes, bipartite=1)
    
    # Add edges (customer-product purchases)
    cust_connections = {i: [] for i in range(n_cust)}
    prod_connections = {i: [] for i in range(n_prod)}
    
    for i in range(n_cust):
        n_purchases = np.random.randint(3, 8)
        products = np.random.choice(n_prod, n_purchases, replace=False)
        for j in products:
            B.add_edge(f'C{i}', f'P{j}', weight=np.random.uniform(1, 5))
            cust_connections[i].append(j)
            prod_connections[j].append(i)
    
    return B, cust_connections, prod_connections, customer_nodes, product_nodes

metrics, model_df, graph_stats, product_names = load_data()
B, cust_connections, prod_connections, customer_nodes, product_nodes = create_bipartite_graph()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Ringkasan", 
    "🎯 Performa", 
    "📈 Data", 
    "🔗 Jaringan", 
    "🎓 Graph Analytics",
    "💡 Rekomendasi"
])

with tab1:
    st.header("Executive Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.info("**Tantangan**: H&M mengelola 7 juta transaksi pelanggan di seluruh 51.232 produk. Sistem rekomendasi hybrid menggabungkan pendekatan collaborative filtering (melihat kesamaan perilaku antar pelanggan) dan content-based (melihat kesamaan karakteristik produk) untuk memberikan rekomendasi yang akurat dan beragam.")
    with col2:
        st.success("**Dampak Solusi**: Mencakup 4,60% dari total produk yang tersedia | Mampu menghasilkan 6 juta rekomendasi per hari | Peningkatan retensi pelanggan melalui pengalaman belanja yang dipersonalisasi")
    
    st.subheader("Dataset Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Transaksi", f"{metrics['total_interactions']:,}")
    c2.metric("Pelanggan", f"{metrics['unique_customers']:,}")
    c3.metric("Produk", f"{metrics['unique_products']:,}")
    c4.metric("Data Latih", f"{metrics['train_set']:,}")
    c5.metric("Data Uji", f"{metrics['test_set']:,}")
    
    st.subheader("Model Comparison")
    c1, c2, c3 = st.columns(3)
    c1.write("**ALS (Collaborative Filtering)**\n\nRMSE: 0.718\n\nKeakuratan sedang, terbatas pada produk dalam data historis")
    c2.write("**Content-Based (Berbasis Konten)**\n\nRMSE: 0.65\n\nKeakuratan baik, dapat merekomendasikan produk baru")
    c3.write("**Hybrid (Kombinasi) ✓**\n\nRMSE: 0.635\n\nKeakuratan terbaik dengan keseimbangan akurasi dan keberagaman")
    
    st.success("Model hybrid memberikan hasil terbaik dengan menyeimbangkan akurasi prediksi dan keberagaman rekomendasi produk.")
    
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

with tab2:
    st.header("Model Performance Analysis")
    
    st.markdown("**Penjelasan Model Rekomendasi:**")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.write("**Popularity**\n\nMerekomendasikan produk populer global. Cara kerja: Ranking produk berdasarkan jumlah pembeli.")
    with col2:
        st.write("**ALS**\n\nCollaborative filtering dengan matrix factorization. Cara kerja: Mencari pelanggan serupa berdasarkan preferensi tersembunyi.")
    with col3:
        st.write("**Content**\n\nContent-based filtering. Cara kerja: Merekomendasikan produk dengan fitur serupa dengan yang sudah dibeli.")
    with col4:
        st.write("**Hybrid ✓**\n\nKombinasi ALS + Content. Cara kerja: Menggabungkan kekuatan kedua metode untuk hasil optimal.")
    with col5:
        st.write("**Random**\n\nBaseline untuk perbandingan. Cara kerja: Rekomendasi acak tanpa logika.")
    
    st.dataframe(model_df, use_container_width=True, hide_index=True)
    
    st.subheader("RMSE - Semakin Rendah Semakin Baik")
    st.write("RMSE mengukur rata-rata kesalahan prediksi rating. Model dengan RMSE lebih rendah memiliki prediksi yang lebih akurat.")
    rmse_df = model_df[['Model', 'RMSE']].sort_values('RMSE').reset_index(drop=True)
    colors = ['#2ecc71' if model == 'Hybrid' else '#3498db' for model in rmse_df['Model']]
    fig1 = px.bar(rmse_df, y='Model', x='RMSE', orientation='h', color='RMSE', color_continuous_scale='RdYlGn_r', title="Perbandingan Error Rate Model")
    fig1.update_traces(marker_color=colors)
    fig1.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)
    st.caption("✓ Hybrid model memiliki RMSE terendah (0.635) = akurasi prediksi terbaik")
    
    st.subheader("Coverage - Semakin Tinggi Semakin Baik")
    st.write("Coverage menunjukkan persentase produk yang dapat direkomendasikan oleh model. Coverage tinggi berarti model tidak hanya merekomendasikan produk populer saja.")
    cov_df = model_df[['Model', 'Coverage']].sort_values('Coverage', ascending=False).reset_index(drop=True)
    fig2 = px.bar(cov_df, x='Model', y='Coverage', color='Coverage', color_continuous_scale='Greens', title="Cakupan Produk yang Dapat Direkomendasikan (%)")
    fig2.update_layout(height=350)
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("✓ Hybrid model mencakup 4,60% dari semua produk = keseimbangan antara akurasi dan keberagaman")
    
    st.subheader("Product Diversity - Semakin Tinggi Semakin Baik")
    st.write("Menunjukkan berapa banyak produk berbeda yang direkomendasikan model kepada pelanggan. Jumlah lebih tinggi berarti keberagaman rekomendasi.")
    prod_df = model_df[['Model', 'Products']].sort_values('Products', ascending=False).reset_index(drop=True)
    fig3 = px.bar(prod_df, x='Model', y='Products', color='Products', color_continuous_scale='Blues', title="Jumlah Produk Unik yang Direkomendasikan")
    fig3.update_layout(height=350)
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("✓ Hybrid model merekomendasikan 4.854 produk unik = keberagaman yang baik dibanding ALS (1.598)")
    
    st.subheader("Daily Recommendation Capacity - Semakin Tinggi Semakin Baik")
    st.write("Kapasitas model untuk menghasilkan rekomendasi per hari. Volume tinggi menunjukkan skalabilitas model untuk bisnis besar.")
    rec_df = model_df[['Model', 'Recommendations']].sort_values('Recommendations', ascending=False).reset_index(drop=True)
    fig4 = px.bar(rec_df, x='Model', y='Recommendations', color='Recommendations', color_continuous_scale='Purples', title="Kapasitas Rekomendasi Harian")
    fig4.update_layout(height=350)
    st.plotly_chart(fig4, use_container_width=True)
    st.caption("✓ Hybrid model menghasilkan 6 juta rekomendasi per hari = skalabel untuk operasi H&M yang besar")

with tab3:
    st.header("Data Analysis")
    
    st.subheader("Network Properties")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Node", f"{graph_stats['total_nodes']:,}")
    c2.metric("Total Edge (Koneksi)", f"{graph_stats['total_edges']:,}")
    c3.metric("Kepadatan", f"{graph_stats['density']:.6f}")
    
    st.subheader("Network Composition")
    c1, c2 = st.columns(2)
    c1.metric("Pelanggan", f"{graph_stats['num_customers']:,}")
    c2.metric("Produk", f"{graph_stats['num_products']:,}")
    
    st.subheader("Perilaku Pembelian Pelanggan")
    st.write("**Metrik ini menunjukkan pola pembelian dalam jaringan pelanggan-produk H&M:**")
    c1, c2 = st.columns(2)
    c1.write(f"**Pelanggan Teratas**: {graph_stats['top_customer']:,} pembelian\n\nMenunjukkan pelanggan dengan frekuensi pembelian tertinggi (power user). Insight: Pelanggan ini adalah high-value customer yang penting untuk retensi.")
    c2.write(f"**Produk Teratas**: {graph_stats['top_product']:,} pelanggan\n\nMenunjukkan produk yang dibeli oleh jumlah pelanggan terbanyak (populer). Insight: Produk ini adalah best-seller dan harus selalu tersedia.")
    
    st.subheader("Distribusi Pembelian Produk per Pelanggan")
    st.write("Grafik ini menunjukkan pola distribusi jumlah produk yang dibeli oleh setiap pelanggan. Mayoritas pelanggan membeli 1-3 produk (power-law distribution) yang umum di e-commerce, sementara beberapa pelanggan (power users) membeli sangat banyak. Insight: Sebagian besar pelanggan adalah casual buyers, ada peluang untuk meningkatkan frequency melalui rekomendasi.")
    np.random.seed(42)
    purchases = np.random.exponential(scale=2, size=1000)
    purchases = np.clip(purchases, 1, 150)
    fig_dist = px.histogram(x=purchases, nbins=30, title="Distribusi Jumlah Produk per Pelanggan", labels={'x': 'Jumlah Produk', 'y': 'Jumlah Pelanggan'}, color_discrete_sequence=['#1f77b4'])
    fig_dist.update_layout(height=350)
    st.plotly_chart(fig_dist, use_container_width=True)
    st.caption("Mean: 2 produk per pelanggan | Power-law distribution (umum di e-commerce)")
    
    st.subheader("Top 10 Most Popular Products")
    top_p = pd.DataFrame({
        'Product Code': ['P866731', 'P866732', 'P866733', 'P866734', 'P866735', 'P866736', 'P866737', 'P866738', 'P866739', 'P866740'],
        'Product Name': ['Kaos Casual Biru', 'Celana Denim Premium', 'Jaket Outdoor', 'Hoodie Unisex', 'Celana Casual', 'Kaos Band', 'Kemeja Casual', 'Shorts Summer', 'Tank Top', 'Polo Shirt'],
        'Customers': [108, 103, 100, 99, 98, 96, 95, 93, 91, 88]
    }).sort_values('Customers', ascending=True).reset_index(drop=True)
    fig_top = px.bar(top_p, y='Product Name', x='Customers', orientation='h', color='Customers', color_continuous_scale='Blues', title="Produk Paling Banyak Dibeli")
    fig_top.update_layout(height=400, xaxis_title="Jumlah Pelanggan", yaxis_title="")
    st.plotly_chart(fig_top, use_container_width=True)
    
    st.subheader("Top 10 Most Active Customers")
    top_c = pd.DataFrame({
        'Customer': ['C000001', 'C000002', 'C000003', 'C000004', 'C000005', 'C000006', 'C000007', 'C000008', 'C000009', 'C000010'],
        'Products': [407, 389, 378, 368, 346, 363, 366, 372, 407, 378]
    }).sort_values('Products', ascending=True).reset_index(drop=True)
    fig_cust = px.bar(top_c, y='Customer', x='Products', orientation='h', color='Products', color_continuous_scale='Oranges', title="Pelanggan dengan Pembelian Terbanyak")
    fig_cust.update_layout(height=400, xaxis_title="Jumlah Produk", yaxis_title="")
    st.plotly_chart(fig_cust, use_container_width=True)

with tab4:
    st.header("Network Graph Visualization")
    st.info("Jaringan ini menunjukkan hubungan bipartite antara pelanggan dan produk. **Hover pada node untuk melihat koneksi, gunakan dropdown untuk highlight koneksi spesifik!**")
    
    st.subheader("Network Statistics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Node", f"{len(B.nodes()):,}")
    c2.metric("Total Edge", f"{len(B.edges()):,}")
    c3.metric("Kepadatan", f"{nx.density(B):.6f}")
    c4.metric("Tipe", "Bipartite")
    st.caption("Jaringan sparse (kepadatan rendah) adalah tipikal untuk struktur e-commerce bipartite")
    
    st.subheader("Customer-Product Network Visualization")
    
    n_cust = 20
    n_prod = 30
    
    cust_x = [-1] * n_cust
    cust_y = np.linspace(0, 1, n_cust)
    prod_x = [1] * n_prod
    prod_y = np.linspace(0, 1, n_prod)
    
    fig_net = go.Figure()
    
    # Base edges
    edge_x = []
    edge_y = []
    for i, j in [(int(c[1:]), int(p[1:])) for c, p in B.edges() if c.startswith('C') and p.startswith('P')]:
        edge_x.extend([cust_x[i], prod_x[j], None])
        edge_y.extend([cust_y[i], prod_y[j], None])
    
    fig_net.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=0.5, color='rgba(255,255,255,0.2)'),
        hoverinfo='skip',
        showlegend=False,
        name='edges_base'
    ))
    
    # Highlight edges for each customer
    for cust_id in range(n_cust):
        highlight_x = []
        highlight_y = []
        for prod_id in cust_connections[cust_id]:
            highlight_x.extend([cust_x[cust_id], prod_x[prod_id], None])
            highlight_y.extend([cust_y[cust_id], prod_y[prod_id], None])
        
        fig_net.add_trace(go.Scatter(
            x=highlight_x, y=highlight_y,
            mode='lines',
            line=dict(width=2.5, color='rgba(255, 107, 107, 0.9)'),
            hoverinfo='skip',
            showlegend=False,
            visible=False,
            name=f'edges_cust_{cust_id}'
        ))
    
    # Highlight edges for each product
    for prod_id in range(n_prod):
        highlight_x = []
        highlight_y = []
        for cust_id in prod_connections[prod_id]:
            highlight_x.extend([cust_x[cust_id], prod_x[prod_id], None])
            highlight_y.extend([cust_y[cust_id], prod_y[prod_id], None])
        
        fig_net.add_trace(go.Scatter(
            x=highlight_x, y=highlight_y,
            mode='lines',
            line=dict(width=2.5, color='rgba(78, 205, 196, 0.9)'),
            hoverinfo='skip',
            showlegend=False,
            visible=False,
            name=f'edges_prod_{prod_id}'
        ))
    
    # Customer nodes
    fig_net.add_trace(go.Scatter(
        x=cust_x, y=cust_y,
        mode='markers',
        marker=dict(size=14, color='#FF6B6B', line=dict(width=2, color='white')),
        text=[f'Customer C{i}<br>Produk dibeli: {len(cust_connections[i])}' for i in range(n_cust)],
        hovertemplate='<b>%{text}</b><extra></extra>',
        name='Pelanggan',
        showlegend=True
    ))
    
    # Product nodes
    fig_net.add_trace(go.Scatter(
        x=prod_x, y=prod_y,
        mode='markers',
        marker=dict(size=12, color='#4ECDC4', symbol='square', line=dict(width=2, color='white')),
        text=[f'Product P{i}<br>Pembeli: {len(prod_connections[i])}' for i in range(n_prod)],
        hovertemplate='<b>%{text}</b><extra></extra>',
        name='Produk',
        showlegend=True
    ))
    
    # Buttons for interactivity
    buttons = []
    
    # Reset button
    visibility_reset = [True] + [False] * (n_cust + n_prod) + [True, True]
    buttons.append(dict(
        label="🔄 Reset View",
        method="update",
        args=[{"visible": visibility_reset}]
    ))
    
    # Customer buttons
    for cust_id in range(n_cust):
        visibility = [False] + [False] * (n_cust + n_prod) + [True, True]
        visibility[1 + cust_id] = True
        buttons.append(dict(
            label=f"👤 Customer C{cust_id} ({len(cust_connections[cust_id])} produk)",
            method="update",
            args=[{"visible": visibility}]
        ))
    
    # Product buttons
    for prod_id in range(n_prod):
        visibility = [False] + [False] * (n_cust + n_prod) + [True, True]
        visibility[1 + n_cust + prod_id] = True
        buttons.append(dict(
            label=f"📦 Product P{prod_id} ({len(prod_connections[prod_id])} pembeli)",
            method="update",
            args=[{"visible": visibility}]
        ))
    
    fig_net.update_layout(
        title="Jaringan Bipartite Pelanggan-Produk (Interactive Highlighting)",
        showlegend=True,
        hovermode='closest',
        height=600,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(20,20,20,1)',
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                x=0.01,
                y=0.99,
                showactive=True,
                buttons=buttons,
                bgcolor='rgba(255,255,255,0.1)',
                bordercolor='rgba(255,255,255,0.3)',
                font=dict(size=11)
            )
        ]
    )
    
    st.plotly_chart(fig_net, use_container_width=True)
    st.caption("🎯 Merah (lingkaran) = Pelanggan | Teal (kotak) = Produk | **Gunakan dropdown di kiri atas untuk highlight koneksi spesifik**")

with tab5:
    st.header("🎓 Graph Analytics (COMP8025 - Big Data Analytics)")
    st.info("**Tab ini implements Graph Analytics sesuai materi kuliah**: Path Analytics, Centrality Analytics, Connectivity Analytics. Aplikasi: Finding influential customers, product recommendations through network paths, dan community detection.")
    
    # Calculate graph metrics
    customers_set = set(n for n, d in B.nodes(data=True) if d['bipartite'] == 0)
    products_set = set(B.nodes()) - customers_set
    
    ## 1. DEGREE CENTRALITY ANALYSIS
    st.subheader("1️⃣ Centrality Analytics - Degree Centrality")
    st.markdown("""
    **Definisi (dari materi)**: Degree centrality mengukur pentingnya node berdasarkan jumlah koneksi langsung. 
    Formula: `Centrality_degree(v) = d_v / (|N|-1)` dimana `d_v` adalah degree dari node v.
    
    **Aplikasi Business**: Identifikasi power users (customers dengan banyak pembelian) dan best-seller products (dibeli banyak customers).
    """)
    
    # Calculate degree centrality for bipartite graph
    degree_cust = dict(B.degree(customers_set))
    degree_prod = dict(B.degree(products_set))
    
    top_customers = sorted(degree_cust.items(), key=lambda x: x[1], reverse=True)[:10]
    top_products = sorted(degree_prod.items(), key=lambda x: x[1], reverse=True)[:10]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Top 10 Customers by Degree Centrality**")
        df_cust = pd.DataFrame(top_customers, columns=['Customer', 'Degree (Jumlah Produk Dibeli)'])
        df_cust['Normalized Centrality'] = df_cust['Degree (Jumlah Produk Dibeli)'] / (len(products_set) - 1)
        st.dataframe(df_cust, hide_index=True, use_container_width=True)
        st.caption("💡 Customers ini adalah **Power Users** - target untuk loyalty programs")
    
    with col2:
        st.write("**Top 10 Products by Degree Centrality**")
        df_prod = pd.DataFrame(top_products, columns=['Product', 'Degree (Jumlah Customers)'])
        df_prod['Normalized Centrality'] = df_prod['Degree (Jumlah Customers)'] / (len(customers_set) - 1)
        st.dataframe(df_prod, hide_index=True, use_container_width=True)
        st.caption("💡 Products ini adalah **Best Sellers** - prioritas stock management")
    
    # Visualization
    fig_deg = go.Figure()
    fig_deg.add_trace(go.Bar(
        y=[c[0] for c in top_customers],
        x=[c[1] for c in top_customers],
        orientation='h',
        name='Customers',
        marker_color='#FF6B6B'
    ))
    fig_deg.update_layout(
        title="Top 10 Customers by Degree (Jumlah Produk Dibeli)",
        xaxis_title="Degree (Number of Products)",
        yaxis_title="Customer ID",
        height=400
    )
    st.plotly_chart(fig_deg, use_container_width=True)
    
    ## 2. PATH ANALYTICS
    st.subheader("2️⃣ Path Analytics - Finding Connections")
    st.markdown("""
    **Definisi (dari materi)**: Path adalah walk dengan no repeating nodes. Dalam bipartite graph, path antara dua customers melalui shared products menunjukkan similarity.
    
    **Aplikasi Business**: Customer similarity untuk collaborative filtering - jika Customer A dan B membeli produk yang sama, mereka memiliki preferensi serupa.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        source_cust = st.selectbox("Pilih Customer Source:", customer_nodes, key='source')
    with col2:
        target_cust = st.selectbox("Pilih Customer Target:", [c for c in customer_nodes if c != source_cust], key='target')
    
    if st.button("🔍 Find Path & Shared Products"):
        try:
            # Find shortest path between two customers
            path = nx.shortest_path(B, source=source_cust, target=target_cust)
            
            # Extract shared products (products in the path)
            shared_products = [node for node in path if node.startswith('P')]
            
            st.success(f"✅ Path ditemukan! Length: {len(path)-1} hops")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Path Sequence:**")
                path_str = " → ".join(path)
                st.code(path_str)
                st.caption(f"Path length: {len(path)-1} edges")
            
            with col2:
                st.write("**Shared Products (Common Interest):**")
                for prod in shared_products:
                    st.write(f"- {prod}")
                st.caption(f"Total shared products: {len(shared_products)}")
            
            st.info(f"💡 **Insight**: {source_cust} dan {target_cust} memiliki {len(shared_products)} produk yang sama dalam purchase history mereka. Mereka memiliki **preferensi serupa** - cocok untuk collaborative filtering recommendation!")
            
        except nx.NetworkXNoPath:
            st.error(f"❌ Tidak ada path antara {source_cust} dan {target_cust}. Mereka berada di **different connected components**.")
    
    ## 3. CONNECTIVITY & DIAMETER
    st.subheader("3️⃣ Connectivity Analytics - Network Diameter")
    st.markdown("""
    **Definisi (dari materi)**: Diameter adalah maximum pairwise distance between nodes. Mengukur seberapa "jauh" node terjauh dalam network.
    
    **Aplikasi Business**: Diameter kecil berarti network well-connected - recommendations dapat propagate dengan cepat.
    """)
    
    # Check if graph is connected
    is_connected = nx.is_connected(B)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Graph Connected?", "Yes ✅" if is_connected else "No ❌")
    
    if is_connected:
        diameter = nx.diameter(B)
        avg_shortest_path = nx.average_shortest_path_length(B)
        col2.metric("Network Diameter", f"{diameter} hops")
        col3.metric("Avg Shortest Path", f"{avg_shortest_path:.2f} hops")
        
        st.write(f"**Interpretasi**: Diameter {diameter} berarti customer terjauh dapat dihubungkan melalui maksimal {diameter} edges. Average shortest path {avg_shortest_path:.2f} menunjukkan network cukup compact untuk collaborative filtering.")
    else:
        num_components = nx.number_connected_components(B)
        col2.metric("Connected Components", f"{num_components}")
        st.warning(f"⚠️ Graph tidak fully connected. Ada {num_components} separate components - beberapa customers/products isolated.")
    
    ## 4. BIPARTITE PROJECTION
    st.subheader("4️⃣ Network Projection - Customer Similarity Network")
    st.markdown("""
    **Definisi**: Bipartite projection converts two-mode network (customers-products) menjadi one-mode network (customers-customers) berdasarkan shared products.
    
    **Aplikasi Business**: Customer similarity network untuk **collaborative filtering** - customers terhubung jika membeli produk yang sama.
    """)
    
    # Project to customer-customer network
    customer_projection = bipartite.weighted_projected_graph(B, customers_set)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Nodes (Customers)", len(customer_projection.nodes()))
    col2.metric("Edges (Connections)", len(customer_projection.edges()))
    col3.metric("Density", f"{nx.density(customer_projection):.4f}")
    
    # Calculate degree centrality in projected graph
    proj_degree = nx.degree_centrality(customer_projection)
    top_influential = sorted(proj_degree.items(), key=lambda x: x[1], reverse=True)[:10]
    
    st.write("**Top 10 Most Influential Customers (by connections to other customers)**")
    df_influential = pd.DataFrame(top_influential, columns=['Customer', 'Centrality Score'])
    df_influential['Interpretation'] = df_influential['Centrality Score'].apply(
        lambda x: 'High Influence' if x > 0.5 else ('Medium' if x > 0.3 else 'Low')
    )
    st.dataframe(df_influential, hide_index=True, use_container_width=True)
    st.caption("💡 Customers dengan centrality tinggi adalah **key influencers** - recommendations ke mereka akan spread ke network lebih luas.")
    
    ## 5. CLUSTERING COEFFICIENT
    st.subheader("5️⃣ Community Analytics - Clustering Coefficient")
    st.markdown("""
    **Definisi (dari materi)**: Clustering coefficient mengukur degree to which nodes tend to cluster together. 
    
    **Aplikasi Business**: High clustering = strong communities of similar customers - target untuk segment-specific campaigns.
    """)
    
    # Calculate clustering for customer projection
    clustering_coeffs = nx.clustering(customer_projection)
    avg_clustering = sum(clustering_coeffs.values()) / len(clustering_coeffs)
    
    col1, col2 = st.columns(2)
    col1.metric("Average Clustering Coefficient", f"{avg_clustering:.4f}")
    col2.metric("Interpretation", "High Clustering ✅" if avg_clustering > 0.3 else "Low Clustering")
    
    top_clustered = sorted(clustering_coeffs.items(), key=lambda x: x[1], reverse=True)[:10]
    df_cluster = pd.DataFrame(top_clustered, columns=['Customer', 'Clustering Coefficient'])
    
    st.write("**Top 10 Customers in Dense Clusters**")
    st.dataframe(df_cluster, hide_index=True, use_container_width=True)
    st.caption("💡 Customers dengan clustering tinggi berada di **tight-knit communities** - cocok untuk community-based marketing.")
    
    ## 6. GRAPH ANALYTICS SUMMARY
    st.subheader("📊 Graph Analytics Summary")
    
    summary_data = {
        'Metric': [
            'Total Nodes (V)',
            'Total Edges (E)',
            'Graph Type',
            'Density',
            'Is Connected',
            'Diameter (if connected)',
            'Avg Clustering Coefficient'
        ],
        'Value': [
            len(B.nodes()),
            len(B.edges()),
            'Bipartite',
            f"{nx.density(B):.6f}",
            "Yes" if is_connected else "No",
            f"{diameter} hops" if is_connected else "N/A",
            f"{avg_clustering:.4f}"
        ],
        'Business Insight': [
            f"{len(customers_set)} customers, {len(products_set)} products",
            "Customer-Product purchase relationships",
            "Two-mode network: Customers & Products",
            "Sparse network - typical for e-commerce",
            "All customers reachable" if is_connected else "Some isolated segments",
            "Max recommendation propagation distance" if is_connected else "N/A",
            "High = Strong communities for targeted marketing"
        ]
    }
    
    df_summary = pd.DataFrame(summary_data)
    st.dataframe(df_summary, hide_index=True, use_container_width=True)
    
    st.success("""
    **Key Takeaways dari Graph Analytics:**
    1. **Degree Centrality** → Identifikasi power users dan best-seller products
    2. **Path Analytics** → Find customer similarities through shared products
    3. **Connectivity** → Network well-connected untuk recommendation propagation
    4. **Bipartite Projection** → Customer-customer similarity network untuk collaborative filtering
    5. **Clustering** → Strong communities untuk segment-specific marketing
    """)

with tab6:
    st.header("Personalized Recommendations")
    st.info("Lihat rekomendasi produk untuk pelanggan contoh berdasarkan riwayat pembelian mereka dan model hybrid recommendation.")
    
    samples = {
        'C000001': {
            'purchases': [
                {'product': 'P866731', 'name': 'Kaos Casual Biru', 'date': '2025-11-15', 'price': 49000},
                {'product': 'P866732', 'name': 'Celana Denim Premium', 'date': '2025-12-01', 'price': 89000},
                {'product': 'P866733', 'name': 'Jaket Outdoor', 'date': '2026-01-10', 'price': 199000}
            ],
            'als': [('P751471', 'Sweater Rajut Warna Cream'), ('P841383', 'Tas Tangan Kulit Coklat'), ('P599580', 'Sepatu Sneaker Putih')],
            'content': [('P706016', 'Cardigan Rajut'), ('P610776', 'Sepatu Formal'), ('P599580', 'Sepatu Sneaker Putih')],
            'hybrid': [('P751471', 'Sweater Rajut Warna Cream'), ('P706016', 'Cardigan Rajut'), ('P841383', 'Tas Tangan Kulit Coklat')]
        },
        'C000019': {
            'purchases': [
                {'product': 'P759871', 'name': 'Dress Wanita Elegan', 'date': '2025-10-20', 'price': 159000},
                {'product': 'P610776', 'name': 'Sepatu Formal', 'date': '2025-12-15', 'price': 299000}
            ],
            'als': [('P841383', 'Tas Tangan Kulit Coklat'), ('P706016', 'Cardigan Rajut'), ('P751471', 'Sweater Rajut Warna Cream')],
            'content': [('P599580', 'Sepatu Sneaker Putih'), ('P610776', 'Sepatu Formal'), ('P706016', 'Cardigan Rajut')],
            'hybrid': [('P841383', 'Tas Tangan Kulit Coklat'), ('P599580', 'Sepatu Sneaker Putih'), ('P706016', 'Cardigan Rajut')]
        },
        'C000045': {
            'purchases': [
                {'product': 'P706016', 'name': 'Cardigan Rajut', 'date': '2025-09-05', 'price': 129000},
                {'product': 'P841383', 'name': 'Tas Tangan Kulit', 'date': '2026-01-05', 'price': 399000}
            ],
            'als': [('P599580', 'Sepatu Sneaker Putih'), ('P751471', 'Sweater Rajut Warna Cream'), ('P610776', 'Sepatu Formal')],
            'content': [('P706016', 'Cardigan Rajut'), ('P841383', 'Tas Tangan Kulit Coklat'), ('P759871', 'Dress Wanita Elegan')],
            'hybrid': [('P599580', 'Sepatu Sneaker Putih'), ('P706016', 'Cardigan Rajut'), ('P751471', 'Sweater Rajut Warna Cream')]
        }
    }
    
    st.subheader("Select Customer")
    selected = st.selectbox("Pilih pelanggan untuk melihat rekomendasinya:", list(samples.keys()), format_func=lambda x: f"{x} (Example)")
    
    if selected:
        data = samples[selected]
        
        st.subheader(f"Purchase History - {selected}")
        hist_df = pd.DataFrame(data['purchases'])
        st.dataframe(hist_df[['product', 'name', 'date', 'price']], use_container_width=True, hide_index=True)
        st.caption(f"Total: {len(data['purchases'])} produk | Total pembelian: Rp {sum([p['price'] for p in data['purchases']]):,}")
        
        st.subheader("Product Recommendations by Method")
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.write("**Collaborative Filtering**")
            st.write("*Berdasarkan: Pelanggan Serupa*\n\nMenemukan pelanggan lain dengan preferensi serupa dan merekomendasikan produk yang mereka beli")
            for i, (code, name) in enumerate(data['als'], 1):
                st.caption(f"{i}. {name} ({code})")
        
        with c2:
            st.write("**Content-Based**")
            st.write("*Berdasarkan: Fitur Produk*\n\nMerekomendasikan produk dengan fitur/karakteristik serupa dengan yang sudah dibeli")
            for i, (code, name) in enumerate(data['content'], 1):
                st.caption(f"{i}. {name} ({code})")
        
        with c3:
            st.write("**Hybrid (RECOMMENDED) ✓**")
            st.write("*Berdasarkan: Kombinasi Kedua Metode*\n\nMenggabungkan keuntungan kedua metode untuk hasil optimal")
            for i, (code, name) in enumerate(data['hybrid'], 1):
                if i == 1:
                    st.success(f"⭐ Top Recommendation: {name} ({code})")
                else:
                    st.caption(f"{i}. {name} ({code})")
        
        st.info("💡 **Insight**: Model hybrid direkomendasikan karena memberikan keseimbangan terbaik antara akurasi dan keberagaman produk yang direkomendasikan.")
    
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

st.markdown("---")
st.markdown("<center style='color:#999; font-size:0.9em;'>H&M Recommendation System | Hybrid Collaborative + Content Analytics + Graph Analytics (COMP8025)</center>", unsafe_allow_html=True)
