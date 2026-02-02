import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from pathlib import Path

# ====================================
# PAGE CONFIG
# ====================================
st.set_page_config(
    page_title="H&M Recommendation System Dashboard",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================================
# CUSTOM CSS
# ====================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stAlert {
        background-color: #e3f2fd;
    }
</style>
""", unsafe_allow_html=True)

# ====================================
# DATA LOADING WITH CACHING
# ====================================
@st.cache_data
def load_model_performance():
    """Load model performance metrics"""
    with open('data/model_performance.json', 'r') as f:
        return json.load(f)

@st.cache_data
def load_network_stats():
    """Load network statistics"""
    with open('data/network_stats.json', 'r') as f:
        return json.load(f)

@st.cache_data
def load_top_customers():
    """Load top customers data"""
    return pd.read_csv('data/top_customers.csv')

@st.cache_data
def load_top_products():
    """Load top products data"""
    return pd.read_csv('data/top_products.csv')

@st.cache_data
def load_customer_distribution():
    """Load customer purchase distribution"""
    return pd.read_csv('data/customer_distribution.csv')

@st.cache_data
def load_bipartite_nodes():
    """Load bipartite graph nodes"""
    return pd.read_csv('data/bipartite_nodes.csv')

@st.cache_data
def load_bipartite_edges():
    """Load bipartite graph edges"""
    return pd.read_csv('data/bipartite_edges.csv')

# ====================================
# SIDEBAR NAVIGATION
# ====================================
st.sidebar.markdown("# 🛍️ H&M Big Data Analytics")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["📊 Model Performance", "📈 Graph Analytics", "🕸️ Network Visualization"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### About
Dashboard untuk visualisasi sistem rekomendasi H&M menggunakan:
- **Collaborative Filtering** (ALS)
- **Content-Based Filtering**
- **Graph Analytics** (NetworkX)

**Dataset:** H&M Personalized Fashion Recommendations

**Author:** Michael Sanjaya  
**Course:** Big Data Analytics - BINUS Graduate Program
""")

# ====================================
# PAGE 1: MODEL PERFORMANCE
# ====================================
if page == "📊 Model Performance":
    st.markdown('<h1 class="main-header">📊 Model Performance Overview</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data
    model_perf = load_model_performance()
    network_stats = load_network_stats()
    
    # ========== DATASET STATISTICS ==========
    st.subheader("📂 Dataset Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Nodes",
            value=f"{network_stats['total_nodes']:,}",
            help="Total customer + product nodes in graph"
        )
    
    with col2:
        st.metric(
            label="Total Edges",
            value=f"{network_stats['total_edges']:,}",
            help="Total purchase interactions"
        )
    
    with col3:
        st.metric(
            label="Unique Customers",
            value=f"{network_stats['num_customers']:,}",
            help="Number of unique customers analyzed"
        )
    
    with col4:
        st.metric(
            label="Unique Products",
            value=f"{network_stats['num_products']:,}",
            help="Number of unique products in catalog"
        )
    
    st.markdown("---")
    
    # ========== MODEL COMPARISON ==========
    st.subheader("🎯 Model Performance Comparison")
    
    # Prepare data for visualization
    models = list(model_perf.keys())
    rmse_values = [model_perf[m]['RMSE'] if model_perf[m]['RMSE'] != 'N/A' else None for m in models]
    coverage_values = [model_perf[m]['Coverage'] for m in models]
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('RMSE Comparison (Lower is Better)', 'Coverage Comparison (Higher is Better)'),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    # RMSE Chart
    rmse_colors = ['red' if v and v > 1 else 'orange' if v and v > 0.7 else 'green' for v in rmse_values]
    fig.add_trace(
        go.Bar(
            x=[m for m, v in zip(models, rmse_values) if v is not None],
            y=[v for v in rmse_values if v is not None],
            name='RMSE',
            marker_color=rmse_colors,
            text=[f"{v:.4f}" if v else "" for v in rmse_values if v is not None],
            textposition='outside'
        ),
        row=1, col=1
    )
    
    # Coverage Chart
    coverage_colors = ['green' if v > 30 else 'orange' if v > 5 else 'red' for v in coverage_values]
    fig.add_trace(
        go.Bar(
            x=models,
            y=coverage_values,
            name='Coverage (%)',
            marker_color=coverage_colors,
            text=[f"{v:.2f}%" for v in coverage_values],
            textposition='outside'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=500,
        showlegend=False,
        title_text="Model Performance Metrics",
        title_x=0.5,
        title_font_size=20
    )
    
    fig.update_xaxes(title_text="Model", row=1, col=1)
    fig.update_xaxes(title_text="Model", row=1, col=2)
    fig.update_yaxes(title_text="RMSE", row=1, col=1)
    fig.update_yaxes(title_text="Coverage (%)", row=1, col=2)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ========== MODEL DETAILS TABLE ==========
    st.subheader("📋 Detailed Model Metrics")
    
    model_df = pd.DataFrame([
        {
            'Model': model,
            'RMSE': model_perf[model]['RMSE'],
            'Coverage (%)': model_perf[model]['Coverage'],
            'Status': '✅ Best RMSE' if model == 'ALS' else '✅ Best Coverage' if model in ['Random', 'Popularity'] else '⚠️ Rule-based' if model == 'Content' else '🔄 Combined'
        }
        for model in models
    ])
    
    st.dataframe(
        model_df,
        use_container_width=True,
        hide_index=True
    )
    
    # ========== KEY INSIGHTS ==========
    st.subheader("💡 Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **🎯 Best Accuracy: ALS Model**
        - RMSE: 0.718 (lowest error)
        - Coverage: 1.52% (personalized but limited)
        - Best for: Returning customers dengan history cukup
        """)
    
    with col2:
        st.info("""
        **🌟 Best Coverage: Baseline Models**
        - Random/Popularity: 37.42% coverage
        - Hybrid: 4.60% coverage (balanced)
        - Best for: New users & product discovery
        """)
    
    st.success("""
    **📊 Recommendation:**  
    Use **Hybrid approach** combining ALS (accuracy) + Content-Based (coverage) + Graph-based (network effects)
    untuk mendapatkan balance antara personalization accuracy dan product discovery.
    """)

# ====================================
# PAGE 2: GRAPH ANALYTICS
# ====================================
elif page == "📈 Graph Analytics":
    st.markdown('<h1 class="main-header">📈 Graph Analytics & Network Insights</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data
    network_stats = load_network_stats()
    top_customers = load_top_customers()
    top_products = load_top_products()
    customer_dist = load_customer_distribution()
    
    # ========== NETWORK METRICS ==========
    st.subheader("🔍 Network Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="Network Density",
            value=f"{network_stats['density']:.6f}",
            help="Ratio of actual edges to possible edges"
        )
    
    with col2:
        st.metric(
            label="Clustering Coef.",
            value=f"{network_stats['clustering_coefficient']:.4f}",
            help="Average clustering coefficient (0.0 for bipartite)"
        )
    
    with col3:
        st.metric(
            label="Top Customer",
            value=f"{network_stats['top_customer_degree']} products",
            help="Most diverse shopper"
        )
    
    with col4:
        st.metric(
            label="Top Product",
            value=f"{network_stats['top_product_degree']} customers",
            help="Most popular product"
        )
    
    with col5:
        avg_degree = network_stats['total_edges'] / network_stats['total_nodes']
        st.metric(
            label="Avg Degree",
            value=f"{avg_degree:.2f}",
            help="Average connections per node"
        )
    
    st.markdown("---")
    
    # ========== TOP CUSTOMERS & PRODUCTS ==========
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👥 Top 10 Customers by Purchase Diversity")
        
        fig_customers = px.bar(
            top_customers,
            x='Degree',
            y=top_customers.index,
            orientation='h',
            text='Degree',
            labels={'Degree': 'Number of Unique Products Purchased', 'y': 'Customer Rank'},
            color='Degree',
            color_continuous_scale='Blues'
        )
        fig_customers.update_layout(
            height=400,
            yaxis={'categoryorder': 'total ascending'},
            showlegend=False
        )
        fig_customers.update_traces(textposition='outside')
        st.plotly_chart(fig_customers, use_container_width=True)
        
        st.caption(f"🏆 Most diverse shopper: **{network_stats['top_customer_degree']} unique products**")
    
    with col2:
        st.subheader("🛍️ Top 10 Most Popular Products")
        
        fig_products = px.bar(
            top_products,
            x='Degree',
            y=top_products.index,
            orientation='h',
            text='Degree',
            labels={'Degree': 'Number of Customers', 'y': 'Product Rank'},
            color='Degree',
            color_continuous_scale='Reds'
        )
        fig_products.update_layout(
            height=400,
            yaxis={'categoryorder': 'total ascending'},
            showlegend=False
        )
        fig_products.update_traces(textposition='outside')
        st.plotly_chart(fig_products, use_container_width=True)
        
        st.caption(f"🌟 Most popular product (ID: {network_stats['top_product']}): **{network_stats['top_product_degree']} customers**")
    
    st.markdown("---")
    
    # ========== CUSTOMER PURCHASE DISTRIBUTION ==========
    st.subheader("📊 Customer Purchase Distribution")
    
    fig_dist = px.histogram(
        customer_dist,
        x='purchases',
        nbins=30,
        labels={'purchases': 'Number of Products Purchased', 'count': 'Number of Customers'},
        title='Distribution of Purchase Counts per Customer',
        color_discrete_sequence=['#1f77b4']
    )
    
    # Add statistics
    mean_purchases = customer_dist['purchases'].mean()
    median_purchases = customer_dist['purchases'].median()
    
    fig_dist.add_vline(x=mean_purchases, line_dash="dash", line_color="red", 
                       annotation_text=f"Mean: {mean_purchases:.1f}", annotation_position="top left")
    fig_dist.add_vline(x=median_purchases, line_dash="dash", line_color="green", 
                       annotation_text=f"Median: {median_purchases:.1f}", annotation_position="top right")
    
    fig_dist.update_layout(height=400)
    st.plotly_chart(fig_dist, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean Purchases", f"{mean_purchases:.2f}")
    with col2:
        st.metric("Median Purchases", f"{median_purchases:.0f}")
    with col3:
        st.metric("Max Purchases", f"{customer_dist['purchases'].max()}")
    
    # ========== BUSINESS INSIGHTS ==========
    st.markdown("---")
    st.subheader("💼 Business Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **🎯 Customer Segmentation:**
        - **VIP Customers** (top 10): Average 350+ unique products
        - **Regular Customers**: Median ~50 products
        - **Opportunity**: Cross-sell campaigns for low-diversity shoppers
        """)
    
    with col2:
        st.info("""
        **📦 Product Strategy:**
        - **Hero Products** (top 10): Average 85+ customers each
        - **Long-tail**: 24,542 products, many with low reach
        - **Opportunity**: Bundle recommendations, outfit suggestions
        """)
    
    st.success("""
    **📊 Network Insight:**  
    Low network density (0.0004) menunjukkan masih banyak **unexplored connections**.  
    Graph-based recommendations dapat membantu **product discovery** dan **cross-category selling**.
    """)

# ====================================
# PAGE 3: NETWORK VISUALIZATION
# ====================================
elif page == "🕸️ Network Visualization":
    st.markdown('<h1 class="main-header">🕸️ Customer-Product Network Visualization</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data
    nodes_df = load_bipartite_nodes()
    edges_df = load_bipartite_edges()
    
    st.info("""
    **ℹ️ About This Network:**  
    Bipartite graph showing connections between **customers** (blue) and **products** (red).  
    Edge = Purchase relationship. Node size = Degree centrality (number of connections).
    """)
    
    # ========== FILTERS ==========
    st.sidebar.markdown("### 🎛️ Visualization Controls")
    
    min_degree = st.sidebar.slider(
        "Minimum Node Degree",
        min_value=1,
        max_value=int(nodes_df['degree'].max()),
        value=50,
        help="Filter nodes with at least this many connections"
    )
    
    max_nodes = st.sidebar.slider(
        "Maximum Nodes to Display",
        min_value=100,
        max_value=1000,
        value=500,
        step=50,
        help="Limit total nodes for performance"
    )
    
    # Filter nodes by degree
    filtered_nodes = nodes_df[nodes_df['degree'] >= min_degree].head(max_nodes)
    
    # Filter edges to only include filtered nodes
    filtered_edges = edges_df[
        edges_df['source'].isin(filtered_nodes['id']) & 
        edges_df['target'].isin(filtered_nodes['id'])
    ]
    
    st.markdown(f"**Displaying:** {len(filtered_nodes)} nodes, {len(filtered_edges)} edges")
    
    # ========== BUILD NETWORKX GRAPH ==========
    G = nx.Graph()
    
    # Add nodes
    for _, row in filtered_nodes.iterrows():
        G.add_node(row['id'], 
                  node_type=row['type'], 
                  degree=row['degree'],
                  label=row['label'])
    
    # Add edges
    for _, row in filtered_edges.iterrows():
        G.add_edge(row['source'], row['target'])
    
    # ========== COMPUTE LAYOUT ==========
    with st.spinner("Computing network layout..."):
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    
    # ========== CREATE PLOTLY FIGURE ==========
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])
    
    # Separate customer and product nodes
    customer_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'customer']
    product_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'product']
    
    # Customer trace
    customer_trace = go.Scatter(
        x=[pos[node][0] for node in customer_nodes],
        y=[pos[node][1] for node in customer_nodes],
        mode='markers',
        name='Customers',
        marker=dict(
            size=[G.nodes[node]['degree']/10 for node in customer_nodes],
            color='#1f77b4',
            line=dict(width=1, color='white')
        ),
        text=[f"Customer {G.nodes[node].get('label', '')}<br>Degree: {G.nodes[node]['degree']}" for node in customer_nodes],
        hoverinfo='text'
    )
    
    # Product trace
    product_trace = go.Scatter(
        x=[pos[node][0] for node in product_nodes],
        y=[pos[node][1] for node in product_nodes],
        mode='markers',
        name='Products',
        marker=dict(
            size=[G.nodes[node]['degree']/5 for node in product_nodes],
            color='#ff7f0e',
            line=dict(width=1, color='white')
        ),
        text=[f"Product {node}<br>Degree: {G.nodes[node]['degree']}" for node in product_nodes],
        hoverinfo='text'
    )
    
    # Create figure
    fig = go.Figure(
        data=[edge_trace, customer_trace, product_trace],
        layout=go.Layout(
            title='Customer-Product Bipartite Network',
            titlefont_size=20,
            showlegend=True,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=700,
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='black',
                borderwidth=1
            )
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ========== NETWORK STATISTICS ==========
    st.markdown("---")
    st.subheader("📊 Filtered Network Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Customers", len(customer_nodes))
    with col2:
        st.metric("Products", len(product_nodes))
    with col3:
        st.metric("Edges", len(filtered_edges))
    with col4:
        density = nx.density(G) if len(G) > 0 else 0
        st.metric("Density", f"{density:.6f}")
    
    st.caption("""
    **💡 Interpretation:**  
    - **Blue nodes** = Customers (size = number of products purchased)  
    - **Orange nodes** = Products (size = number of customers)  
    - **Lines** = Purchase relationships  
    - **Clusters** = Customers with similar product preferences
    """)

# ====================================
# FOOTER
# ====================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p><strong>H&M Big Data Recommendation System Dashboard</strong></p>
    <p>Built with Streamlit | Data Source: H&M Personalized Fashion Recommendations (Kaggle)</p>
    <p>© 2026 Michael Sanjaya - BINUS Graduate Program</p>
</div>
""", unsafe_allow_html=True)
