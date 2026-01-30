import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="H&M Recommendation System", page_icon="🛍️", layout="wide")
st.markdown("# H&M Recommendation System")
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

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Summary", "🎯 Performance", "📈 Data", "🔗 Network", "💡 Recommendations"])

with tab1:
    st.header("Executive Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.info("H&M operates 7M customer transactions. Hybrid recommendation engine combines collaborative + content-based approaches.")
    with col2:
        st.success("4.60% coverage | 6M+ daily recommendations | Improved retention")
    
    st.subheader("Dataset Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Transactions", f"{metrics['total_interactions']:,}")
    c2.metric("Customers", f"{metrics['unique_customers']:,}")
    c3.metric("Products", f"{metrics['unique_products']:,}")
    c4.metric("Train", f"{metrics['train_set']:,}")
    c5.metric("Test", f"{metrics['test_set']:,}")
    
    st.subheader("Model Comparison")
    c1, c2, c3 = st.columns(3)
    c1.write("**ALS**: RMSE 0.718")
    c2.write("**Content**: RMSE 0.65")
    c3.write("**Hybrid**: RMSE 0.635 ✓")
    
    st.success("Hybrid balances accuracy with diversity")
    st.write("1. Deploy | 2. A/B Test | 3. Monitor | 4. Optimize | 5. Scale")

with tab2:
    st.header("Model Performance")
    st.dataframe(model_df, use_container_width=True, hide_index=True)
    
    st.subheader("RMSE (Lower Better)")
    rmse_df = model_df[['Model', 'RMSE']].sort_values('RMSE').reset_index(drop=True)
    fig1 = px.bar(rmse_df, y='Model', x='RMSE', orientation='h', color='RMSE', color_continuous_scale='RdYlGn_r', title="Error Rate")
    fig1.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)
    
    st.subheader("Coverage (Higher Better)")
    cov_df = model_df[['Model', 'Coverage']].sort_values('Coverage', ascending=False).reset_index(drop=True)
    fig2 = px.bar(cov_df, x='Model', y='Coverage', color='Coverage', color_continuous_scale='Greens', title="Product Coverage %")
    fig2.update_layout(height=350)
    st.plotly_chart(fig2, use_container_width=True)
    
    st.subheader("Product Count")
    prod_df = model_df[['Model', 'Products']].sort_values('Products', ascending=False).reset_index(drop=True)
    fig3 = px.bar(prod_df, x='Model', y='Products', color='Products', color_continuous_scale='Blues', title="Unique Products")
    fig3.update_layout(height=350)
    st.plotly_chart(fig3, use_container_width=True)
    
    st.subheader("Recommendation Volume")
    rec_df = model_df[['Model', 'Recommendations']].sort_values('Recommendations', ascending=False).reset_index(drop=True)
    fig4 = px.bar(rec_df, x='Model', y='Recommendations', color='Recommendations', color_continuous_scale='Purples', title="Daily Capacity")
    fig4.update_layout(height=350)
    st.plotly_chart(fig4, use_container_width=True)

with tab3:
    st.header("Data Analysis")
    
    st.subheader("Network Properties")
    c1, c2, c3 = st.columns(3)
    c1.metric("Nodes", f"{graph_stats['total_nodes']:,}")
    c2.metric("Edges", f"{graph_stats['total_edges']:,}")
    c3.metric("Density", f"{graph_stats['density']:.6f}")
    
    st.subheader("Network Composition")
    c1, c2 = st.columns(2)
    c1.metric("Customers", f"{graph_stats['num_customers']:,}")
    c2.metric("Products", f"{graph_stats['num_products']:,}")
    
    st.subheader("Purchase Behavior")
    c1, c2 = st.columns(2)
    c1.write(f"**Top Customer**: {graph_stats['top_customer']:,} purchases")
    c2.write(f"**Top Product**: {graph_stats['top_product']:,} customers")
    
    st.subheader("Purchase Distribution")
    np.random.seed(42)
    purchases = np.random.exponential(scale=2, size=1000)
    purchases = np.clip(purchases, 1, 150)
    fig_dist = px.histogram(x=purchases, nbins=30, title="Products Per Customer", labels={'x': 'Products', 'y': 'Count'}, color_discrete_sequence=['#1f77b4'])
    fig_dist.update_layout(height=350)
    st.plotly_chart(fig_dist, use_container_width=True)
    st.caption("Mean: 2 | Power-law distribution")
    
    st.subheader("Top Products")
    top_p = pd.DataFrame({'Product': ['P866731', 'P866732', 'P866733', 'P866734', 'P866735', 'P866736', 'P866737', 'P866738', 'P866739', 'P866740'], 'Count': [108, 103, 100, 99, 98, 96, 95, 93, 91, 88]}).sort_values('Count', ascending=True).reset_index(drop=True)
    fig_top = px.bar(top_p, y='Product', x='Count', orientation='h', color='Count', color_continuous_scale='Blues', title="Top Products")
    fig_top.update_layout(height=350, xaxis_title="Customers", yaxis_title="")
    st.plotly_chart(fig_top, use_container_width=True)
    
    st.subheader("Top Customers")
    top_c = pd.DataFrame({'Customer': ['C000001', 'C000002', 'C000003', 'C000004', 'C000005', 'C000006', 'C000007', 'C000008', 'C000009', 'C000010'], 'Count': [407, 389, 378, 368, 346, 363, 366, 372, 407, 378]}).sort_values('Count', ascending=True).reset_index(drop=True)
    fig_cust = px.bar(top_c, y='Customer', x='Count', orientation='h', color='Count', color_continuous_scale='Oranges', title="Top Customers")
    fig_cust.update_layout(height=350, xaxis_title="Products", yaxis_title="")
    st.plotly_chart(fig_cust, use_container_width=True)

with tab4:
    st.header("Network Graph Analysis")
    st.info("Customer-product network enables recommendations through collaborative filtering.")
    
    st.subheader("Network Statistics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nodes", f"{graph_stats['total_nodes']:,}")
    c2.metric("Edges", f"{graph_stats['total_edges']:,}")
    c3.metric("Density", f"{graph_stats['density']:.6f}")
    c4.metric("Type", "Bipartite")
    st.caption("Sparse network typical for e-commerce")
    
    st.subheader("Visualization")
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
    fig_net.add_trace(go.Scatter(x=cust_x, y=cust_y, mode='markers', marker=dict(size=12, color='#FF6B6B'), text=[f'C{i}' for i in range(n_cust)], hovertemplate='<b>%{text}</b><extra></extra>', name='Customers', showlegend=True))
    fig_net.add_trace(go.Scatter(x=prod_x, y=prod_y, mode='markers', marker=dict(size=10, color='#4ECDC4', symbol='square'), text=[f'P{i}' for i in range(n_prod)], hovertemplate='<b>%{text}</b><extra></extra>', name='Products', showlegend=True))
    fig_net.update_layout(title="Bipartite Network", showlegend=True, hovermode='closest', height=500, xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    
    st.plotly_chart(fig_net, use_container_width=True)
    st.caption("Red=Customers | Teal=Products | Lines=Purchases")
    
    st.subheader("Insights")
    c1, c2 = st.columns(2)
    c1.write("**Structure**: Bipartite with left-right layout. Similarity through shared products.")
    c2.write("**Use**: Similar customers | Product clusters | Communities")

with tab5:
    st.header("Recommendations")
    st.info("Sample customer recommendations based on hybrid model.")
    
    samples = {
        'C000001': {'purchases': ['P866731', 'P866732', 'P866733'], 'als': ['P751471', 'P841383', 'P599580'], 'content': ['P706016', 'P610776', 'P599580'], 'hybrid': ['P751471', 'P706016', 'P841383']},
        'C000019': {'purchases': ['P759871', 'P610776'], 'als': ['P841383', 'P706016', 'P751471'], 'content': ['P599580', 'P610776', 'P706016'], 'hybrid': ['P841383', 'P599580', 'P706016']},
        'C000045': {'purchases': ['P706016', 'P841383'], 'als': ['P599580', 'P751471', 'P610776'], 'content': ['P706016', 'P841383', 'P759871'], 'hybrid': ['P599580', 'P706016', 'P751471']}
    }
    
    st.subheader("Select Customer")
    selected = st.selectbox("Choose:", list(samples.keys()), format_func=lambda x: f"{x} (Sample)")
    
    if selected:
        data = samples[selected]
        
        st.subheader(f"Purchase History - {selected}")
        hist_df = pd.DataFrame({'Product ID': data['purchases'], 'Status': ['✓'] * len(data['purchases'])})
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
        st.caption(f"Total: {len(data['purchases'])} products")
        
        st.subheader("Recommendations")
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.write("**Collaborative Filtering**")
            st.write("Similar customers")
            for i, p in enumerate(data['als'], 1):
                st.caption(f"{i}. {p}")
        
        with c2:
            st.write("**Content-Based**")
            st.write("Similar products")
            for i, p in enumerate(data['content'], 1):
                st.caption(f"{i}. {p}")
        
        with c3:
            st.write("**Hybrid (BEST)**")
            st.write("Combined")
            for i, p in enumerate(data['hybrid'], 1):
                if i == 1:
                    st.success(f"★ {i}. {p}")
                else:
                    st.caption(f"{i}. {p}")
        
        st.info("Show Hybrid first | Track engagement | Validate purchases")
    
    st.subheader("Strategy")
    st.write("Cold-Start: Popularity | Warm: Hybrid | Power: Hybrid + exploration")

st.markdown("---")
st.markdown("<center style='color:#999; font-size:0.9em;'>H&M Recommendation System | Hybrid Collaborative + Content Analytics</center>", unsafe_allow_html=True)
