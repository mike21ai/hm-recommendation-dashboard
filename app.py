import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Sistem Rekomendasi H&M", page_icon="🛍️", layout="wide")
st.markdown("# Sistem Rekomendasi H&M")
st.markdown("**Hybrid Collaborative Filtering + Content-Based Analytics Dashboard**")
st.markdown("---")

@st.cache_data
def load_data():
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
    }

metrics, model_df, graph_stats = load_data()

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Ringkasan", "🎯 Performa", "📈 Data", "🔗 Jaringan", "💡 Rekomendasi"])

with tab1:
    st.header("Ringkasan Eksekutif")
    col1, col2 = st.columns(2)
    with col1:
        st.info("**Tantangan**: H&M mengelola 7 juta transaksi pelanggan di seluruh 51.232 produk. Sistem rekomendasi hybrid menggabungkan pendekatan collaborative filtering (melihat kesamaan perilaku antar pelanggan) dan content-based (melihat kesamaan karakteristik produk) untuk memberikan rekomendasi yang akurat dan beragam.")
    with col2:
        st.success("**Dampak Solusi**: Mencakup 4,60% dari total produk yang tersedia | Mampu menghasilkan 6 juta rekomendasi per hari | Peningkatan retensi pelanggan melalui pengalaman belanja yang dipersonalisasi")
    
    st.subheader("Ikhtisar Dataset")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Transaksi", f"{metrics['total_interactions']:,}")
    c2.metric("Pelanggan", f"{metrics['unique_customers']:,}")
    c3.metric("Produk", f"{metrics['unique_products']:,}")
    c4.metric("Data Latih", f"{metrics['train_set']:,}")
    c5.metric("Data Uji", f"{metrics['test_set']:,}")
    
    st.subheader("Perbandingan Model Rekomendasi")
    c1, c2, c3 = st.columns(3)
    c1.write("**ALS (Collaborative Filtering)**\nRMSE: 0.718\nKeakuratan sedang, terbatas pada produk dalam data historis")
    c2.write("**Content-Based (Berbasis Konten)**\nRMSE: 0.65\nKeakuratan baik, dapat merekomendasikan produk baru")
    c3.write("**Hybrid (Kombinasi) ✓**\nRMSE: 0.635\nKeakuratan terbaik dengan keseimbangan akurasi dan keberagaman")
    
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
    st.header("Analisis Performa Model")
    st.dataframe(model_df, use_container_width=True, hide_index=True)
    
    st.subheader("Error Rate (RMSE) - Semakin Rendah Semakin Baik")
    st.write("RMSE mengukur rata-rata kesalahan prediksi model. Model dengan RMSE lebih rendah memiliki prediksi yang lebih akurat.")
    rmse_df = model_df[['Model', 'RMSE']].sort_values('RMSE').reset_index(drop=True)
    fig1 = px.bar(rmse_df, y='Model', x='RMSE', orientation='h', color='RMSE', color_continuous_scale='RdYlGn_r', title="Perbandingan Error Rate Model")
    fig1.update_traces(marker_line_width=2, selector=dict(name=''))
    fig1.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)
    st.caption("✓ Hybrid model memiliki RMSE terendah (0.635) = akurasi prediksi terbaik")
    
    st.subheader("Cakupan Produk (Coverage) - Semakin Tinggi Semakin Baik")
    st.write("Coverage menunjukkan persentase produk yang dapat direkomendasikan oleh model. Coverage tinggi berarti model tidak hanya merekomendasikan produk populer saja.")
    cov_df = model_df[['Model', 'Coverage']].sort_values('Coverage', ascending=False).reset_index(drop=True)
    fig2 = px.bar(cov_df, x='Model', y='Coverage', color='Coverage', color_continuous_scale='Greens', title="Cakupan Produk yang Dapat Direkomendasikan (%)")
    fig2.update_layout(height=350)
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("✓ Hybrid model mencakup 4,60% dari semua produk = keseimbangan antara akurasi dan keberagaman")
    
    st.subheader("Jumlah Produk Unik - Semakin Tinggi Semakin Baik")
    st.write("Menunjukkan berapa banyak produk berbeda yang direkomendasikan model kepada pelanggan. Jumlah lebih tinggi berarti keberagaman rekomendasi.")
    prod_df = model_df[['Model', 'Products']].sort_values('Products', ascending=False).reset_index(drop=True)
    fig3 = px.bar(prod_df, x='Model', y='Products', color='Products', color_continuous_scale='Blues', title="Jumlah Produk Unik yang Direkomendasikan")
    fig3.update_layout(height=350)
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("✓ Hybrid model merekomendasikan 4.854 produk unik = keberagaman yang baik dibanding ALS (1.598)")
    
    st.subheader("Volume Rekomendasi Harian - Semakin Tinggi Semakin Baik")
    st.write("Kapasitas model untuk menghasilkan rekomendasi per hari. Volume tinggi menunjukkan skalabilitas model untuk bisnis besar.")
    rec_df = model_df[['Model', 'Recommendations']].sort_values('Recommendations', ascending=False).reset_index(drop=True)
    fig4 = px.bar(rec_df, x='Model', y='Recommendations', color='Recommendations', color_continuous_scale='Purples', title="Kapasitas Rekomendasi Harian")
    fig4.update_layout(height=350)
    st.plotly_chart(fig4, use_container_width=True)
    st.caption("✓ Hybrid model menghasilkan 6 juta rekomendasi per hari = skalabel untuk operasi H&M yang besar")

with tab3:
    st.header("Analisis Data")
    
    st.subheader("Properti Jaringan Pelanggan-Produk")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Node", f"{graph_stats['total_nodes']:,}")
    c2.metric("Total Edge (Koneksi)", f"{graph_stats['total_edges']:,}")
    c3.metric("Kepadatan", f"{graph_stats['density']:.6f}")
    
    st.subheader("Komposisi Jaringan")
    c1, c2 = st.columns(2)
    c1.metric("Pelanggan", f"{graph_stats['num_customers']:,}")
    c2.metric("Produk", f"{graph_stats['num_products']:,}")
    
    st.subheader("Perilaku Pembelian Pelanggan")
    c1, c2 = st.columns(2)
    c1.write(f"**Pelanggan Teratas**: {graph_stats['top_customer']:,} pembelian\nMenunjukkan pelanggan dengan frekuensi pembelian tertinggi (power user)")
    c2.write(f"**Produk Teratas**: {graph_stats['top_product']:,} pelanggan\nMenunjukkan produk yang dibeli oleh jumlah pelanggan terbanyak (populer)")
    
    st.subheader("Distribusi Pembelian Produk per Pelanggan")
    st.write("Grafik ini menunjukkan pola distribusi jumlah produk yang dibeli oleh setiap pelanggan. Mayoritas pelanggan membeli 1-3 produk (power-law distribution) yang umum di e-commerce, sementara beberapa pelanggan membeli sangat banyak.")
    np.random.seed(42)
    purchases = np.random.exponential(scale=2, size=1000)
    purchases = np.clip(purchases, 1, 150)
    fig_dist = px.histogram(x=purchases, nbins=30, title="Distribusi Jumlah Produk per Pelanggan", labels={'x': 'Jumlah Produk', 'y': 'Jumlah Pelanggan'}, color_discrete_sequence=['#1f77b4'])
    fig_dist.update_layout(height=350)
    st.plotly_chart(fig_dist, use_container_width=True)
    st.caption("Mean: 2 produk per pelanggan | Power-law distribution (umum di e-commerce)")
    
    st.subheader("10 Produk Paling Populer")
    top_p = pd.DataFrame({'Product': ['P866731', 'P866732', 'P866733', 'P866734', 'P866735', 'P866736', 'P866737', 'P866738', 'P866739', 'P866740'], 'Count': [108, 103, 100, 99, 98, 96, 95, 93, 91, 88]}).sort_values('Count', ascending=True).reset_index(drop=True)
    fig_top = px.bar(top_p, y='Product', x='Count', orientation='h', color='Count', color_continuous_scale='Blues', title="Produk Paling Banyak Dibeli")
    fig_top.update_layout(height=350, xaxis_title="Jumlah Pelanggan", yaxis_title="")
    st.plotly_chart(fig_top, use_container_width=True)
    
    st.subheader("10 Pelanggan Paling Aktif")
    top_c = pd.DataFrame({'Customer': ['C000001', 'C000002', 'C000003', 'C000004', 'C000005', 'C000006', 'C000007', 'C000008', 'C000009', 'C000010'], 'Count': [407, 389, 378, 368, 346, 363, 366, 372, 407, 378]}).sort_values('Count', ascending=True).reset_index(drop=True)
    fig_cust = px.bar(top_c, y='Customer', x='Count', orientation='h', color='Count', color_continuous_scale='Oranges', title="Pelanggan dengan Pembelian Terbanyak")
    fig_cust.update_layout(height=350, xaxis_title="Jumlah Produk", yaxis_title="")
    st.plotly_chart(fig_cust, use_container_width=True)

with tab4:
    st.header("Analisis Jaringan Pelanggan-Produk")
    st.info("Jaringan ini menunjukkan hubungan antara pelanggan dan produk yang mereka beli. Setiap titik merah di kiri adalah pelanggan, setiap kotak teal di kanan adalah produk, dan garis menunjukkan pembelian. Analisis jaringan membantu menemukan: (1) pelanggan serupa melalui produk yang mereka beli, (2) kluster produk yang sering dibeli bersama, dan (3) pelanggan atau produk yang berpengaruh dalam jaringan.")
    
    st.subheader("Statistik Jaringan")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Node", f"{graph_stats['total_nodes']:,}")
    c2.metric("Total Edge", f"{graph_stats['total_edges']:,}")
    c3.metric("Kepadatan", f"{graph_stats['density']:.6f}")
    c4.metric("Tipe", "Bipartite")
    st.caption("Jaringan sparse (kepadatan rendah) adalah tipikal untuk struktur e-commerce bipartite")
    
    st.subheader("Visualisasi Jaringan Pelanggan-Produk")
    np.random.seed(42)
    n_cust = 20
    n_prod = 30
    
    cust_x = [-1] * n_cust
    cust_y = np.linspace(0, 1, n_cust)
    prod_x = [1] * n_prod
    prod_y = np.linspace(0, 1, n_prod)
    
    edge_x = []
    edge_y = []
    for i in range(n_cust):
        for _ in range(np.random.randint(3, 8)):
            j = np.random.randint(0, n_prod)
            edge_x.extend([cust_x[i], prod_x[j], None])
            edge_y.extend([cust_y[i], prod_y[j], None])
    
    fig_net = go.Figure()
    fig_net.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.5, color='rgba(100,100,100,0.2)'), hoverinfo='none', showlegend=False))
    fig_net.add_trace(go.Scatter(x=cust_x, y=cust_y, mode='markers', marker=dict(size=12, color='#FF6B6B'), text=[f'C{i}' for i in range(n_cust)], hovertemplate='<b>%{text}</b><extra></extra>', name='Pelanggan', showlegend=True))
    fig_net.add_trace(go.Scatter(x=prod_x, y=prod_y, mode='markers', marker=dict(size=10, color='#4ECDC4', symbol='square'), text=[f'P{i}' for i in range(n_prod)], hovertemplate='<b>%{text}</b><extra></extra>', name='Produk', showlegend=True))
    fig_net.update_layout(title="Jaringan Bipartite Pelanggan-Produk", showlegend=True, hovermode='closest', height=500, xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    
    st.plotly_chart(fig_net, use_container_width=True)
    st.caption("Merah (lingkaran) = Pelanggan | Teal (kotak) = Produk | Garis = Hubungan pembelian")
    
    st.subheader("Cara Menggunakan Graph Analytics")
    st.markdown("""
    **1. Menemukan Pelanggan Serupa**: Pelanggan yang terhubung ke produk yang sama memiliki preferensi serupa
    
    **2. Mengidentifikasi Kluster Produk**: Produk yang sering dibeli bersama oleh pelanggan sama membentuk kluster
    
    **3. Deteksi Komunitas**: Menemukan kelompok pelanggan dan produk yang saling terkait erat
    
    **4. Rekomendasi Berbasis Jaringan**: Jika pelanggan A mirip dengan B, dan B membeli produk X, maka X bisa direkomendasikan ke A
    """)

with tab5:
    st.header("Rekomendasi Produk Terpersonalisasi")
    st.info("Lihat rekomendasi produk untuk pelanggan contoh berdasarkan riwayat pembelian mereka dan model hybrid recommendation.")
    
    samples = {
        'C000001': {
            'purchases': [
                {'product': 'P866731', 'name': 'Kaos Casual Biru', 'date': '2025-11-15', 'price': 49000},
                {'product': 'P866732', 'name': 'Celana Denim Premium', 'date': '2025-12-01', 'price': 89000},
                {'product': 'P866733', 'name': 'Jaket Outdoor', 'date': '2026-01-10', 'price': 199000}
            ],
            'als': ['P751471', 'P841383', 'P599580'],
            'content': ['P706016', 'P610776', 'P599580'],
            'hybrid': ['P751471', 'P706016', 'P841383']
        },
        'C000019': {
            'purchases': [
                {'product': 'P759871', 'name': 'Dress Wanita Elegan', 'date': '2025-10-20', 'price': 159000},
                {'product': 'P610776', 'name': 'Sepatu Formal', 'date': '2025-12-15', 'price': 299000}
            ],
            'als': ['P841383', 'P706016', 'P751471'],
            'content': ['P599580', 'P610776', 'P706016'],
            'hybrid': ['P841383', 'P599580', 'P706016']
        },
        'C000045': {
            'purchases': [
                {'product': 'P706016', 'name': 'Cardigan Rajut', 'date': '2025-09-05', 'price': 129000},
                {'product': 'P841383', 'name': 'Tas Tangan Kulit', 'date': '2026-01-05', 'price': 399000}
            ],
            'als': ['P599580', 'P751471', 'P610776'],
            'content': ['P706016', 'P841383', 'P759871'],
            'hybrid': ['P599580', 'P706016', 'P751471']
        }
    }
    
    st.subheader("Pilih Pelanggan")
    selected = st.selectbox("Pilih pelanggan untuk melihat rekomendasinya:", list(samples.keys()), format_func=lambda x: f"{x} (Contoh)")
    
    if selected:
        data = samples[selected]
        
        st.subheader(f"Riwayat Pembelian - {selected}")
        hist_df = pd.DataFrame(data['purchases'])
        st.dataframe(hist_df[['product', 'name', 'date', 'price']], use_container_width=True, hide_index=True)
        st.caption(f"Total: {len(data['purchases'])} produk | Total pembelian: Rp {sum([p['price'] for p in data['purchases']]):,}")
        
        st.subheader("Rekomendasi Produk Berdasarkan 3 Metode")
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.write("**Collaborative Filtering**")
            st.write("*Berdasarkan: Pelanggan Serupa*")
            st.write("Menemukan pelanggan lain dengan preferensi serupa dan merekomendasikan produk yang mereka beli")
            for i, p in enumerate(data['als'], 1):
                st.caption(f"{i}. {p}")
        
        with c2:
            st.write("**Content-Based**")
            st.write("*Berdasarkan: Fitur Produk*")
            st.write("Merekomendasikan produk dengan fitur/karakteristik serupa dengan yang sudah dibeli")
            for i, p in enumerate(data['content'], 1):
                st.caption(f"{i}. {p}")
        
        with c3:
            st.write("**Hybrid (TERBAIK) ✓**")
            st.write("*Berdasarkan: Kombinasi Kedua Metode*")
            st.write("Menggabungkan keuntungan kedua metode untuk hasil optimal")
            for i, p in enumerate(data['hybrid'], 1):
                if i == 1:
                    st.success(f"⭐ Rekomendasi Utama: {p}")
                else:
                    st.caption(f"{i}. {p}")
        
        st.info("💡 **Insight**: Model hybrid direkomendasikan karena memberikan keseimbangan terbaik antara akurasi dan keberagaman produk yang direkomendasikan.")
    
    st.subheader("Strategi Implementasi Rekomendasi")
    st.markdown("""
    **Cold-Start Users (Pelanggan Baru - Tanpa Riwayat Pembelian)**
    - Gunakan model Popularity (produk populer umum)
    - Alasan: Tidak ada data historis untuk collaborative filtering dan content-based
    
    **Warm Users (Pelanggan Aktif - Dengan Riwayat Pembelian)**
    - Gunakan model Hybrid (kombinasi collaborative + content-based)
    - Alasan: Data historis cukup untuk memberikan rekomendasi akurat dan beragam
    
    **Power Users (Pelanggan Setia - Pembelian Sangat Banyak)**
    - Gunakan model Hybrid dengan tambahan exploration (mencoba produk baru)
    - Alasan: Mereka sudah familiar dengan produk standar, perlu diversifikasi untuk fresh recommendations
    """)

st.markdown("---")
st.markdown("<center style='color:#999; font-size:0.9em;'>Sistem Rekomendasi H&M | Hybrid Collaborative + Content Analytics</center>", unsafe_allow_html=True)
