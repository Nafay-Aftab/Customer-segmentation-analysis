import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import datetime as dt
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. PAGE CONFIGURATION & ULTRA-PROFESSIONAL STYLING
# =============================================================================
st.set_page_config(
    page_title="Retail Intelligence Dashboard Pro",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ENHANCED PROFESSIONAL CSS (FIXED SIDEBAR TOGGLE)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hiding MainMenu and Footer is fine, but we MUST keep header visible for the sidebar toggle */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* header {visibility: hidden;}  <-- DELETED THIS LINE */
    
    .main {
        padding: 2rem 3rem;
    }
    
    :root {
        --primary-color: #4F8BF9;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --danger-color: #ef4444;
        --purple-color: #8b5cf6;
        --pink-color: #ec4899;
        --card-bg: rgba(255, 255, 255, 0.05);
        --card-border: rgba(255, 255, 255, 0.1);
        --text-primary: inherit;
        --text-secondary: rgba(156, 163, 175, 1);
        --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        --hover-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-weight: 700 !important;
        letter-spacing: -0.025em;
    }
    
    /* --- FIXED KPI CARD STYLING --- */
    .kpi-card {
        background: var(--card-bg);
        backdrop-filter: blur(10px);
        border: 1px solid var(--card-border);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: var(--shadow);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        min-height: 160px; 
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--primary-color), var(--success-color));
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--hover-shadow);
        border-color: var(--primary-color);
    }
    
    .kpi-card:hover::before {
        opacity: 1;
    }
    
    /* FIX: HIDE ICONS COMPLETELY */
    .kpi-icon {
        display: none !important;
    }
    
    .kpi-content-wrapper {
        position: relative;
        z-index: 1;
    }

    .kpi-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-secondary);
        margin-bottom: 0.5rem;
    }
    
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        line-height: 1;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, var(--primary-color), var(--success-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .kpi-subtitle {
        font-size: 0.875rem;
        color: var(--text-secondary);
        font-weight: 500;
    }

    .trend-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 0.5rem;
        width: fit-content;
    }
    
    .trend-up {
        background: rgba(16, 185, 129, 0.1);
        color: var(--success-color);
    }
    
    .trend-down {
        background: rgba(239, 68, 68, 0.1);
        color: var(--danger-color);
    }
    
    [data-testid="stSidebar"] {
        background: var(--card-bg);
        backdrop-filter: blur(20px);
        border-right: 1px solid var(--card-border);
    }
    
    [data-testid="stSidebar"] > div:first-child {
        padding: 2rem 1.5rem;
    }
    
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, var(--primary-color), #3b82f6);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        box-shadow: var(--shadow);
        letter-spacing: 0.025em;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: var(--hover-shadow);
        background: linear-gradient(135deg, #3b82f6, var(--primary-color));
    }
    
    .stDownloadButton > button {
        background: linear-gradient(135deg, var(--success-color), #059669) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: var(--hover-shadow);
    }
    
    [data-testid="stFileUploader"] {
        border: 2px dashed var(--card-border);
        border-radius: 16px;
        padding: 2rem;
        background: var(--card-bg);
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: var(--primary-color);
        background: rgba(79, 139, 249, 0.05);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: var(--card-bg);
        padding: 0.5rem;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary-color), #3b82f6);
        color: white !important;
    }
    
    .dataframe {
        border-radius: 12px !important;
        overflow: hidden;
        border: 1px solid var(--card-border) !important;
    }
    
    [data-testid="metric-container"] {
        background: var(--card-bg);
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid var(--card-border);
    }
    
    .stAlert {
        border-radius: 12px;
        border: 1px solid var(--card-border);
        background: var(--card-bg);
        backdrop-filter: blur(10px);
    }
    
    .empty-state {
        text-align: center;
        padding: 4rem 2rem;
        background: var(--card-bg);
        border-radius: 20px;
        border: 2px dashed var(--card-border);
        margin: 2rem 0;
    }
    
    .empty-state-icon {
        font-size: 4rem;
        opacity: 0.3;
        margin-bottom: 1rem;
    }
    
    .empty-state-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .empty-state-subtitle {
        color: var(--text-secondary);
        font-size: 1rem;
    }
    
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--card-border), transparent);
        margin: 2rem 0;
    }
    
    .insight-box {
        background: linear-gradient(135deg, rgba(79, 139, 249, 0.1), rgba(16, 185, 129, 0.1));
        border-left: 4px solid var(--primary-color);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0;
    }
    
    .insight-box h4 {
        margin-top: 0;
        color: var(--primary-color);
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .stat-item {
        background: var(--card-bg);
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid var(--card-border);
        text-align: center;
    }
    
    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--primary-color);
    }
    
    .stat-label {
        font-size: 0.75rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.25rem;
    }
    
    .js-plotly-plot {
        border-radius: 12px;
        overflow: hidden;
    }
    
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--card-bg);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--card-border);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary-color);
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 2. HELPER FUNCTIONS (CACHED FOR PERFORMANCE)
# =============================================================================
@st.cache_data(show_spinner=False)
def load_data(uploaded_file):
    """Load data from uploaded Excel or CSV file"""
    try:
        if uploaded_file.name.endswith('.xlsx'):
            all_sheets = pd.read_excel(uploaded_file, sheet_name=None)
            df = pd.concat(all_sheets.values(), ignore_index=True)
        else:
            df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        return None

@st.cache_data(show_spinner=False)
def preprocess_data(df):
    """Clean and preprocess transaction data"""
    df_clean = df.copy()
    
    df_clean["Invoice"] = df_clean["Invoice"].astype(str)
    df_clean = df_clean[~df_clean["Invoice"].str.contains("C", na=False)]
    
    df_clean["StockCode"] = df_clean["StockCode"].astype(str)
    mask = (df_clean["StockCode"].str.match(r"^\d{5}$") == True) | \
           (df_clean["StockCode"].str.match(r"^\d{5}[a-zA-Z]+$") == True)
    df_clean = df_clean[mask]
    
    df_clean.dropna(subset=["Customer ID"], inplace=True)
    df_clean = df_clean[df_clean["Price"] > 0]
    
    df_clean["SalesLineTotal"] = df_clean["Quantity"] * df_clean["Price"]
    df_clean["InvoiceDate"] = pd.to_datetime(df_clean["InvoiceDate"])
    
    return df_clean

@st.cache_data(show_spinner=False)
def calculate_rfm(df):
    """Calculate RFM metrics for each customer"""
    max_date = df["InvoiceDate"].max() + dt.timedelta(days=1)
    
    rfm = df.groupby("Customer ID").agg({
        "InvoiceDate": lambda x: (max_date - x.max()).days,
        "Invoice": "nunique",
        "SalesLineTotal": "sum"
    }).reset_index()
    
    rfm.rename(columns={
        "InvoiceDate": "Recency",
        "Invoice": "Frequency",
        "SalesLineTotal": "MonetaryValue"
    }, inplace=True)
    
    return rfm

@st.cache_data(show_spinner=False)
def remove_outliers(df, columns):
    """Remove statistical outliers using IQR method"""
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.05)
        Q3 = df_clean[col].quantile(0.95)
        IQR = Q3 - Q1
        df_clean = df_clean[
            (df_clean[col] >= Q1 - 1.5 * IQR) & 
            (df_clean[col] <= Q3 + 1.5 * IQR)
        ]
    return df_clean

def perform_clustering(rfm_data, n_clusters):
    """Perform K-Means clustering on RFM data with quality metrics"""
    rfm_log = np.log1p(rfm_data[["Recency", "Frequency", "MonetaryValue"]])
    
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(rfm_scaled)
    
    rfm_final = rfm_data.copy()
    rfm_final['Cluster'] = clusters
    
    # Calculate clustering quality metrics
    silhouette_avg = silhouette_score(rfm_scaled, clusters)
    davies_bouldin = davies_bouldin_score(rfm_scaled, clusters)
    
    return rfm_final, kmeans, silhouette_avg, davies_bouldin, rfm_scaled

# --- NEW FUNCTION REPLACING COHORT ANALYSIS ---
@st.cache_data(show_spinner=False)
def calculate_sales_heatmap(df):
    """Calculate sales intensity by Day of Week and Hour"""
    df_heat = df.copy()
    df_heat['DayOfWeek'] = df_heat['InvoiceDate'].dt.day_name()
    df_heat['Hour'] = df_heat['InvoiceDate'].dt.hour
    
    # Ensure correct day order
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df_heat['DayOfWeek'] = pd.Categorical(df_heat['DayOfWeek'], categories=days_order, ordered=True)
    
    # Aggregate transaction counts
    heatmap_data = df_heat.groupby(['DayOfWeek', 'Hour'])['Invoice'].nunique().reset_index()
    heatmap_pivot = heatmap_data.pivot(index='DayOfWeek', columns='Hour', values='Invoice').fillna(0)
    
    return heatmap_pivot

@st.cache_data(show_spinner=False)
def calculate_customer_lifetime_metrics(df):
    """Calculate comprehensive customer lifetime metrics"""
    customer_metrics = df.groupby('Customer ID').agg({
        'InvoiceDate': ['min', 'max', 'count'],
        'SalesLineTotal': ['sum', 'mean', 'std'],
        'Invoice': 'nunique',
        'StockCode': 'nunique'
    }).reset_index()
    
    customer_metrics.columns = ['Customer ID', 'FirstPurchase', 'LastPurchase', 'TotalOrders',
                                 'TotalRevenue', 'AvgOrderValue', 'StdOrderValue', 
                                 'UniqueInvoices', 'UniqueProducts']
    
    customer_metrics['CustomerLifespanDays'] = (customer_metrics['LastPurchase'] - 
                                                  customer_metrics['FirstPurchase']).dt.days
    customer_metrics['AvgDaysBetweenOrders'] = customer_metrics['CustomerLifespanDays'] / customer_metrics['UniqueInvoices']
    
    return customer_metrics

@st.cache_data(show_spinner=False)
def calculate_product_affinity(df, top_n=20):
    """Calculate product purchase patterns and top products"""
    product_revenue = df.groupby('Description').agg({
        'SalesLineTotal': 'sum',
        'Quantity': 'sum',
        'Customer ID': 'nunique',
        'Invoice': 'nunique'
    }).reset_index()
    
    product_revenue.columns = ['Product', 'TotalRevenue', 'QuantitySold', 'UniqueCustomers', 'Orders']
    product_revenue = product_revenue.sort_values('TotalRevenue', ascending=False).head(top_n)
    product_revenue['AvgOrderValue'] = product_revenue['TotalRevenue'] / product_revenue['Orders']
    
    return product_revenue

# =============================================================================
# 3. MAIN APPLICATION
# =============================================================================
def main():
    # -------------------------------------------------------------------------
    # SIDEBAR CONTROLS
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### 🎯 Control Panel")
        st.markdown("---")
        
        uploaded_file = st.file_uploader(
            "Upload Transaction Data",
            type=['xlsx', 'csv'],
            help="Upload your Online Retail dataset (Excel or CSV format)"
        )
        
        st.markdown("---")
        st.markdown("### ⚙️ Model Configuration")
        
        remove_outliers_option = st.checkbox(
            "Remove Statistical Outliers",
            value=True,
            help="Uses IQR method to remove extreme values for cleaner clusters"
        )
        
        n_clusters = st.slider(
            "Number of Customer Segments",
            min_value=2,
            max_value=8,
            value=4,
            help="More segments = finer granularity, fewer = broader groups"
        )
        
        st.markdown("---")
        st.markdown("### 🎨 Visualization Options")
        
        show_advanced_charts = st.checkbox(
            "Show Advanced Analytics",
            value=True,
            help="Display trading times heatmap, product insights, and trend forecasting"
        )
        
        chart_template = st.selectbox(
            "Chart Theme",
            ["plotly_white", "plotly_dark", "seaborn", "simple_white"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("### 📊 About This Dashboard")
        st.info(
            "**Advanced RFM Analysis Suite**\n\n"
            "Features include:\n"
            "- K-Means clustering with quality metrics\n"
            "- Peak Trading Times Heatmap\n"
            "- Customer lifetime value tracking\n"
            "- Product affinity mining\n"
            "- Predictive trend analysis"
        )
        
        st.markdown("---")
        st.markdown(
            "**Retail Intelligence Suite Pro**\n\n"
            "© 2024 · Version 2.1"
        )
    
    # -------------------------------------------------------------------------
    # HEADER SECTION WITH STATUS
    # -------------------------------------------------------------------------
    col_header1, col_header2, col_header3 = st.columns([3, 1, 1])
    
    with col_header1:
        st.markdown("# 🛍️ Retail Intelligence Dashboard Pro")
        st.markdown(
            "Advanced customer analytics with AI-powered insights and predictive modeling"
        )
    
    with col_header2:
        if uploaded_file:
            st.success("✅ Data Loaded")
        else:
            st.warning("⏳ Awaiting Data")
    
    with col_header3:
        current_time = dt.datetime.now().strftime("%H:%M")
        st.info(f"🕐 {current_time}")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # -------------------------------------------------------------------------
    # MAIN CONTENT
    # -------------------------------------------------------------------------
    if uploaded_file is not None:
        with st.spinner('🔄 Processing your data with advanced analytics...'):
            raw_df = load_data(uploaded_file)
            
            if raw_df is not None:
                clean_df = preprocess_data(raw_df)
                
                # Calculate comprehensive KPIs
                total_rev = clean_df['SalesLineTotal'].sum()
                total_trans = len(clean_df)
                total_cust = clean_df['Customer ID'].nunique()
                avg_order = total_rev / total_trans
                date_range = (clean_df['InvoiceDate'].max() - clean_df['InvoiceDate'].min()).days
                unique_products = clean_df['StockCode'].nunique()
                
                # Calculate growth metrics (comparing first half vs second half)
                mid_date = clean_df['InvoiceDate'].min() + pd.Timedelta(days=date_range//2)
                first_half_rev = clean_df[clean_df['InvoiceDate'] <= mid_date]['SalesLineTotal'].sum()
                second_half_rev = clean_df[clean_df['InvoiceDate'] > mid_date]['SalesLineTotal'].sum()
                revenue_growth = ((second_half_rev - first_half_rev) / first_half_rev * 100) if first_half_rev > 0 else 0
                
                # -------------------------
                # ENHANCED KPI CARDS ROW (FIXED ALIGNMENT)
                # -------------------------
                st.markdown("### 📈 Executive Dashboard Overview")
                
                kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
                
                with kpi1:
                    trend_class = "trend-up" if revenue_growth > 0 else "trend-down"
                    trend_icon = "↗" if revenue_growth > 0 else "↘"
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-icon">💰</div>
                        <div class="kpi-content-wrapper">
                            <div class="kpi-label">Total Revenue</div>
                            <div class="kpi-value">${total_rev:,.0f}</div>
                            <div class="kpi-subtitle">Lifetime Value</div>
                            <div class="trend-badge {trend_class}">{trend_icon} {abs(revenue_growth):.1f}%</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with kpi2:
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-icon">🛒</div>
                        <div class="kpi-content-wrapper">
                            <div class="kpi-label">Transactions</div>
                            <div class="kpi-value">{total_trans:,}</div>
                            <div class="kpi-subtitle">{total_trans/date_range:.1f} per day</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with kpi3:
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-icon">👥</div>
                        <div class="kpi-content-wrapper">
                            <div class="kpi-label">Active Customers</div>
                            <div class="kpi-value">{total_cust:,}</div>
                            <div class="kpi-subtitle">{total_trans/total_cust:.1f} orders/customer</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with kpi4:
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-icon">📊</div>
                        <div class="kpi-content-wrapper">
                            <div class="kpi-label">Avg Order Value</div>
                            <div class="kpi-value">${avg_order:.2f}</div>
                            <div class="kpi-subtitle">Per Transaction</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with kpi5:
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-icon">📦</div>
                        <div class="kpi-content-wrapper">
                            <div class="kpi-label">Product Catalog</div>
                            <div class="kpi-value">{unique_products:,}</div>
                            <div class="kpi-subtitle">Unique SKUs</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                
                # -------------------------
                # ENHANCED TABBED INTERFACE
                # -------------------------
                tab_overview, tab_eda, tab_model, tab_insights, tab_advanced, tab_export = st.tabs([
                    "📊 Overview",
                    "🔍 RFM Analysis",
                    "🧠 Segmentation",
                    "💡 Strategy",
                    "📈 Advanced Analytics",
                    "📥 Export"
                ])
                
                # =================================================================
                # TAB 0: OVERVIEW DASHBOARD
                # =================================================================
                with tab_overview:
                    st.markdown("### 🎯 Business Performance Snapshot")
                    
                    col_left, col_right = st.columns([2, 1])
                    
                    with col_left:
                        # Revenue trend over time
                        revenue_trend = clean_df.groupby(clean_df['InvoiceDate'].dt.to_period('M'))['SalesLineTotal'].sum().reset_index()
                        revenue_trend['InvoiceDate'] = revenue_trend['InvoiceDate'].dt.to_timestamp()
                        
                        fig_revenue_trend = go.Figure()
                        fig_revenue_trend.add_trace(go.Scatter(
                            x=revenue_trend['InvoiceDate'],
                            y=revenue_trend['SalesLineTotal'],
                            mode='lines+markers',
                            name='Revenue',
                            line=dict(color='#4F8BF9', width=3),
                            fill='tozeroy',
                            fillcolor='rgba(79, 139, 249, 0.1)'
                        ))
                        
                        fig_revenue_trend.update_layout(
                            title="📈 Monthly Revenue Trend",
                            xaxis_title="Month",
                            yaxis_title="Revenue ($)",
                            template=chart_template,
                            height=400,
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig_revenue_trend, use_container_width=True)
                    
                    with col_right:
                        st.markdown("#### 🎯 Quick Stats")
                        
                        # Time period
                        st.metric(
                            "Data Time Span",
                            f"{date_range} days",
                            f"{date_range//30} months"
                        )
                        
                        # Customer acquisition
                        customers_per_day = total_cust / date_range
                        st.metric(
                            "Daily New Customers",
                            f"{customers_per_day:.1f}",
                            "Average"
                        )
                        
                        # Revenue per customer
                        rev_per_customer = total_rev / total_cust
                        st.metric(
                            "Revenue per Customer",
                            f"${rev_per_customer:.2f}",
                            "Lifetime"
                        )
                        
                        # Product diversity
                        products_per_transaction = clean_df.groupby('Invoice')['StockCode'].nunique().mean()
                        st.metric(
                            "Items per Order",
                            f"{products_per_transaction:.1f}",
                            "Average"
                        )
                    
                    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                    
                    # Top performers section
                    col_top1, col_top2 = st.columns(2)
                    
                    with col_top1:
                        st.markdown("#### 🏆 Top 10 Customers by Revenue")
                        top_customers = clean_df.groupby('Customer ID')['SalesLineTotal'].sum().sort_values(ascending=False).head(10)
                        
                        fig_top_cust = go.Figure(go.Bar(
                            x=top_customers.values,
                            y=[f"Customer {int(cid)}" for cid in top_customers.index],
                            orientation='h',
                            marker=dict(
                                color=top_customers.values,
                                colorscale='Blues',
                                showscale=False
                            )
                        ))
                        
                        fig_top_cust.update_layout(
                            template=chart_template,
                            height=400,
                            xaxis_title="Total Revenue ($)",
                            yaxis_title="",
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig_top_cust, use_container_width=True)
                    
                    with col_top2:
                        st.markdown("#### 📦 Top 10 Products by Revenue")
                        top_products = clean_df.groupby('Description')['SalesLineTotal'].sum().sort_values(ascending=False).head(10)
                        
                        fig_top_prod = go.Figure(go.Bar(
                            x=top_products.values,
                            y=top_products.index,
                            orientation='h',
                            marker=dict(
                                color=top_products.values,
                                colorscale='Greens',
                                showscale=False
                            )
                        ))
                        
                        fig_top_prod.update_layout(
                            template=chart_template,
                            height=400,
                            xaxis_title="Total Revenue ($)",
                            yaxis_title="",
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig_top_prod, use_container_width=True)
                
                # =================================================================
                # TAB 1: EXPLORATORY DATA ANALYSIS
                # =================================================================
                with tab_eda:
                    st.markdown("### 🔍 RFM Distribution Analysis")
                    
                    rfm_df = calculate_rfm(clean_df)
                    
                    # Enhanced distribution charts with box plots
                    col_chart1, col_chart2 = st.columns(2)
                    
                    with col_chart1:
                        # Recency Distribution with stats
                        fig_r = make_subplots(
                            rows=2, cols=1,
                            row_heights=[0.7, 0.3],
                            vertical_spacing=0.05,
                            subplot_titles=("Distribution", "Box Plot")
                        )
                        
                        fig_r.add_trace(
                            go.Histogram(x=rfm_df["Recency"], nbinsx=50, 
                                         marker_color='#4F8BF9', name='Histogram'),
                            row=1, col=1
                        )
                        
                        fig_r.add_trace(
                            go.Box(x=rfm_df["Recency"], marker_color='#4F8BF9', 
                                   name='Box Plot', showlegend=False),
                            row=2, col=1
                        )
                        
                        fig_r.update_layout(
                            title="📅 Recency Distribution (Days Since Last Purchase)",
                            template=chart_template,
                            height=500,
                            showlegend=False
                        )
                        fig_r.update_xaxes(title_text="Days", row=2, col=1)
                        fig_r.update_yaxes(title_text="Count", row=1, col=1)
                        
                        st.plotly_chart(fig_r, use_container_width=True)
                        
                        # Recency statistics
                        st.markdown("**Key Statistics:**")
                        col_stat1, col_stat2, col_stat3 = st.columns(3)
                        col_stat1.metric("Median", f"{rfm_df['Recency'].median():.0f} days")
                        col_stat2.metric("Mean", f"{rfm_df['Recency'].mean():.0f} days")
                        col_stat3.metric("Std Dev", f"{rfm_df['Recency'].std():.0f} days")
                    
                    with col_chart2:
                        # Frequency Distribution
                        freq_filtered = rfm_df[rfm_df['Frequency'] < 100]
                        
                        fig_f = make_subplots(
                            rows=2, cols=1,
                            row_heights=[0.7, 0.3],
                            vertical_spacing=0.05,
                            subplot_titles=("Distribution", "Box Plot")
                        )
                        
                        fig_f.add_trace(
                            go.Histogram(x=freq_filtered["Frequency"], nbinsx=50,
                                         marker_color='#9b59b6', name='Histogram'),
                            row=1, col=1
                        )
                        
                        fig_f.add_trace(
                            go.Box(x=freq_filtered["Frequency"], marker_color='#9b59b6',
                                   name='Box Plot', showlegend=False),
                            row=2, col=1
                        )
                        
                        fig_f.update_layout(
                            title="🔄 Frequency Distribution (< 100 Orders)",
                            template=chart_template,
                            height=500,
                            showlegend=False
                        )
                        fig_f.update_xaxes(title_text="Orders", row=2, col=1)
                        fig_f.update_yaxes(title_text="Count", row=1, col=1)
                        
                        st.plotly_chart(fig_f, use_container_width=True)
                        
                        # Frequency statistics
                        st.markdown("**Key Statistics:**")
                        col_stat1, col_stat2, col_stat3 = st.columns(3)
                        col_stat1.metric("Median", f"{rfm_df['Frequency'].median():.0f}")
                        col_stat2.metric("Mean", f"{rfm_df['Frequency'].mean():.0f}")
                        col_stat3.metric("Std Dev", f"{rfm_df['Frequency'].std():.0f}")
                    
                    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                    
                    # Monetary value analysis
                    col_chart3, col_chart4 = st.columns(2)
                    
                    with col_chart3:
                        monetary_filtered = rfm_df[rfm_df['MonetaryValue'] < 5000]
                        
                        fig_m = make_subplots(
                            rows=2, cols=1,
                            row_heights=[0.7, 0.3],
                            vertical_spacing=0.05,
                            subplot_titles=("Distribution", "Box Plot")
                        )
                        
                        fig_m.add_trace(
                            go.Histogram(x=monetary_filtered["MonetaryValue"], nbinsx=50,
                                         marker_color='#10b981', name='Histogram'),
                            row=1, col=1
                        )
                        
                        fig_m.add_trace(
                            go.Box(x=monetary_filtered["MonetaryValue"], marker_color='#10b981',
                                   name='Box Plot', showlegend=False),
                            row=2, col=1
                        )
                        
                        fig_m.update_layout(
                            title="💵 Monetary Value Distribution (< $5,000)",
                            template=chart_template,
                            height=500,
                            showlegend=False
                        )
                        fig_m.update_xaxes(title_text="Revenue ($)", row=2, col=1)
                        fig_m.update_yaxes(title_text="Count", row=1, col=1)
                        
                        st.plotly_chart(fig_m, use_container_width=True)
                        
                        # Monetary statistics
                        st.markdown("**Key Statistics:**")
                        col_stat1, col_stat2, col_stat3 = st.columns(3)
                        col_stat1.metric("Median", f"${rfm_df['MonetaryValue'].median():.2f}")
                        col_stat2.metric("Mean", f"${rfm_df['MonetaryValue'].mean():.2f}")
                        col_stat3.metric("Std Dev", f"${rfm_df['MonetaryValue'].std():.2f}")
                    
                    with col_chart4:
                        # RFM Correlation Heatmap
                        corr_matrix = rfm_df[['Recency', 'Frequency', 'MonetaryValue']].corr()
                        
                        fig_corr = go.Figure(data=go.Heatmap(
                            z=corr_matrix.values,
                            x=['Recency', 'Frequency', 'Monetary'],
                            y=['Recency', 'Frequency', 'Monetary'],
                            colorscale='RdBu_r',
                            zmid=0,
                            text=corr_matrix.values,
                            texttemplate='%{text:.2f}',
                            textfont={"size": 14},
                            colorbar=dict(title="Correlation")
                        ))
                        
                        fig_corr.update_layout(
                            title="🔗 RFM Correlation Matrix",
                            template=chart_template,
                            height=500
                        )
                        
                        st.plotly_chart(fig_corr, use_container_width=True)
                        
                        # Interpretation
                        st.markdown("""
                        **Correlation Insights:**
                        - Strong positive correlation between Frequency and Monetary indicates repeat customers spend more
                        - Negative correlation with Recency shows recent buyers tend to be more active
                        """)
                    
                    # Statistical summary table
                    st.markdown("#### 📋 Comprehensive Statistical Summary")
                    summary_stats = rfm_df[['Recency', 'Frequency', 'MonetaryValue']].describe()
                    
                    # Add percentiles
                    for pct in [10, 25, 50, 75, 90, 95, 99]:
                        summary_stats.loc[f'{pct}%'] = rfm_df[['Recency', 'Frequency', 'MonetaryValue']].quantile(pct/100)
                    
                    st.dataframe(
                        summary_stats.style.background_gradient(cmap='RdYlGn_r', axis=1).format("{:.2f}"),
                        use_container_width=True
                    )
                    
                    # Interpretation box
                    st.markdown("""
                    <div class="insight-box">
                        <h4>🎓 Distribution Analysis Insights</h4>
                        <ul>
                            <li><strong>Recency:</strong> Right-skewed distribution indicates healthy recent activity. Monitor the long tail for at-risk customers.</li>
                            <li><strong>Frequency:</strong> Most customers are occasional buyers. Focus strategies on converting them to repeat purchasers.</li>
                            <li><strong>Monetary:</strong> Power law distribution—small percentage of high-value customers drive majority of revenue (Pareto principle).</li>
                            <li><strong>Correlations:</strong> Strong frequency-monetary correlation validates that repeat purchase programs can significantly boost revenue.</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                # =================================================================
                # TAB 2: CLUSTER MODELING
                # =================================================================
                with tab_model:
                    st.markdown("### 🧠 Advanced Customer Segmentation Model")
                    
                    with st.spinner('🔬 Running K-Means clustering with quality assessment...'):
                        processed_rfm = rfm_df.copy()
                        
                        if remove_outliers_option:
                            processed_rfm = remove_outliers(
                                processed_rfm,
                                ["Recency", "Frequency", "MonetaryValue"]
                            )
                            st.info(f"ℹ️ Statistical outliers removed: {len(rfm_df) - len(processed_rfm)} customers ({(len(rfm_df) - len(processed_rfm))/len(rfm_df)*100:.1f}%)")
                        
                        final_df, model, silhouette, davies_bouldin, scaled_data = perform_clustering(processed_rfm, n_clusters)
                        
                        # Display clustering quality metrics
                        col_metric1, col_metric2, col_metric3 = st.columns(3)
                        
                        with col_metric1:
                            quality_color = "normal" if silhouette > 0.3 else "inverse"
                            st.metric(
                                "Silhouette Score",
                                f"{silhouette:.3f}",
                                "Higher is better (0-1)",
                                delta_color=quality_color
                            )
                        
                        with col_metric2:
                            st.metric(
                                "Davies-Bouldin Index",
                                f"{davies_bouldin:.3f}",
                                "Lower is better"
                            )
                        
                        with col_metric3:
                            quality_assessment = "Excellent" if silhouette > 0.5 else "Good" if silhouette > 0.3 else "Fair"
                            st.metric(
                                "Clustering Quality",
                                quality_assessment,
                                f"{n_clusters} segments"
                            )
                        
                        # Cluster naming
                        if n_clusters == 4:
                            cluster_labels = {0: 'REWARD', 1: 'RE-ENGAGE', 2: 'RETAIN', 3: 'NURTURE'}
                        elif n_clusters == 7:
                            cluster_labels = {0: 'REWARD', 1: 'RE-ENGAGE', 2: 'RETAIN', 3: 'NURTURE',
                                              4: 'PAMPER', 5: 'UPSELL', 6: 'DELIGHT'}
                        else:
                            cluster_labels = {i: f'SEGMENT_{i}' for i in range(n_clusters)}
                        
                        final_df['Segment Name'] = final_df['Cluster'].map(cluster_labels)
                    
                    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                    
                    # Enhanced 3D visualization
                    st.markdown("#### 🎯 Interactive 3D Cluster Visualization")
                    
                    fig_3d = px.scatter_3d(
                        final_df.sample(min(1000, len(final_df))),  # Sample for performance
                        x='Recency',
                        y='Frequency',
                        z='MonetaryValue',
                        color='Segment Name',
                        opacity=0.7,
                        height=700,
                        title="RFM Cluster Space (Sample of customers for performance)",
                        template=chart_template,
                        hover_data={'Customer ID': True}
                    )
                    
                    fig_3d.update_layout(
                        scene=dict(
                            xaxis_title='Recency (Days)',
                            yaxis_title='Frequency (Orders)',
                            zaxis_title='Monetary Value ($)',
                            bgcolor="rgba(0,0,0,0)"
                        ),
                        margin=dict(l=0, r=0, b=0, t=40)
                    )
                    
                    st.plotly_chart(fig_3d, use_container_width=True)
                    
                    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                    
                    # Comprehensive segment profiles
                    st.markdown("#### 📊 Detailed Segment Performance Profiles")
                    
                    summary_display = final_df.groupby('Segment Name').agg({
                        'Recency': ['mean', 'median', 'std'],
                        'Frequency': ['mean', 'median', 'std'],
                        'MonetaryValue': ['mean', 'median', 'sum', 'std'],
                        'Customer ID': 'count'
                    }).reset_index()
                    
                    summary_display.columns = ['Segment', 'Avg Recency', 'Med Recency', 'Std Recency',
                                               'Avg Frequency', 'Med Frequency', 'Std Frequency',
                                               'Avg Monetary', 'Med Monetary', 'Total Revenue', 'Std Monetary',
                                               'Customer Count']
                    
                    summary_display = summary_display.sort_values('Total Revenue', ascending=False)
                    summary_display['Revenue %'] = (summary_display['Total Revenue'] / summary_display['Total Revenue'].sum() * 100)
                    summary_display['Customer %'] = (summary_display['Customer Count'] / summary_display['Customer Count'].sum() * 100)
                    summary_display['CLV'] = summary_display['Total Revenue'] / summary_display['Customer Count']
                    
                    st.dataframe(
                        summary_display.style.background_gradient(
                            subset=['Total Revenue', 'Customer Count', 'CLV'],
                            cmap='Blues'
                        ).format({
                            'Avg Recency': '{:.0f}', 'Med Recency': '{:.0f}', 'Std Recency': '{:.0f}',
                            'Avg Frequency': '{:.1f}', 'Med Frequency': '{:.0f}', 'Std Frequency': '{:.1f}',
                            'Avg Monetary': '${:,.0f}', 'Med Monetary': '${:,.0f}',
                            'Total Revenue': '${:,.0f}', 'Std Monetary': '${:,.0f}',
                            'Customer Count': '{:,.0f}',
                            'Revenue %': '{:.1f}%', 'Customer %': '{:.1f}%',
                            'CLV': '${:,.0f}'
                        }),
                        use_container_width=True,
                        height=400
                    )
                    
                    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                    
                    # Enhanced visualizations row
                    col_viz1, col_viz2 = st.columns(2)
                    
                    with col_viz1:
                        # Snake plot with enhanced styling
                        st.markdown("#### 🐍 Relative Attribute Strength (Snake Plot)")
                        
                        scaler_viz = StandardScaler()
                        scaled_viz = scaler_viz.fit_transform(final_df[['Recency', 'Frequency', 'MonetaryValue']])
                        scaled_df = pd.DataFrame(scaled_viz, columns=['Recency', 'Frequency', 'MonetaryValue'])
                        scaled_df['Segment Name'] = final_df['Segment Name'].values
                        
                        melted_df = pd.melt(
                            scaled_df.groupby('Segment Name').mean().reset_index(),
                            id_vars=['Segment Name'],
                            var_name='Metric',
                            value_name='Standardized Value'
                        )
                        
                        fig_snake = px.line(
                            melted_df,
                            x='Metric',
                            y='Standardized Value',
                            color='Segment Name',
                            markers=True,
                            title="Comparative Segment Performance",
                            template=chart_template
                        )
                        
                        fig_snake.update_traces(line=dict(width=3), marker=dict(size=10))
                        fig_snake.update_layout(hovermode='x unified', height=400)
                        
                        st.plotly_chart(fig_snake, use_container_width=True)
                    
                    with col_viz2:
                        # Radar chart for segment comparison
                        st.markdown("#### 📡 Segment Radar Comparison")
                        
                        # Normalize for radar chart
                        radar_data = final_df.groupby('Segment Name')[['Recency', 'Frequency', 'MonetaryValue']].mean()
                        radar_norm = (radar_data - radar_data.min()) / (radar_data.max() - radar_data.min())
                        
                        fig_radar = go.Figure()
                        
                        for segment in radar_norm.index:
                            fig_radar.add_trace(go.Scatterpolar(
                                r=radar_norm.loc[segment].values.tolist() + [radar_norm.loc[segment].values[0]],
                                theta=['Recency', 'Frequency', 'Monetary', 'Recency'],
                                fill='toself',
                                name=segment
                            ))
                        
                        fig_radar.update_layout(
                            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                            showlegend=True,
                            template=chart_template,
                            height=400
                        )
                        
                        st.plotly_chart(fig_radar, use_container_width=True)
                    
                    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                    
                    # Distribution visualizations
                    col_dist1, col_dist2 = st.columns(2)
                    
                    with col_dist1:
                        # Customer distribution pie chart
                        fig_pie_count = px.pie(
                            summary_display,
                            values='Customer Count',
                            names='Segment',
                            title="🎯 Customer Distribution by Segment",
                            template=chart_template,
                            hole=0.4
                        )
                        fig_pie_count.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig_pie_count, use_container_width=True)
                    
                    with col_dist2:
                        # Revenue distribution pie chart
                        fig_pie_rev = px.pie(
                            summary_display,
                            values='Total Revenue',
                            names='Segment',
                            title="💰 Revenue Distribution by Segment",
                            template=chart_template,
                            hole=0.4
                        )
                        fig_pie_rev.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig_pie_rev, use_container_width=True)
                    
                    # CLV comparison
                    st.markdown("#### 💎 Customer Lifetime Value by Segment")
                    
                    fig_clv = go.Figure(go.Bar(
                        x=summary_display['Segment'],
                        y=summary_display['CLV'],
                        marker=dict(
                            color=summary_display['CLV'],
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="CLV ($)")
                        ),
                        text=summary_display['CLV'].apply(lambda x: f'${x:,.0f}'),
                        textposition='outside'
                    ))
                    
                    fig_clv.update_layout(
                        title="Average Customer Lifetime Value per Segment",
                        xaxis_title="Segment",
                        yaxis_title="CLV ($)",
                        template=chart_template,
                        height=400
                    )
                    
                    st.plotly_chart(fig_clv, use_container_width=True)
                
                # =================================================================
                # TAB 3: BUSINESS INSIGHTS
                # =================================================================
                with tab_insights:
                    st.markdown("### 💡 Strategic Business Recommendations")
                    
                    # Store final_df and summary_display in session state for Strategy tab
                    if 'final_df' not in st.session_state:
                        st.session_state.final_df = final_df
                        st.session_state.summary_display = summary_display
                    
                    for idx, row in summary_display.iterrows():
                        segment = row['Segment']
                        cust_count = int(row['Customer Count'])
                        avg_monetary = row['Avg Monetary']
                        total_rev = row['Total Revenue']
                        clv = row['CLV']
                        
                        # Segment-specific strategies
                        if segment == 'REWARD':
                            strategy = {
                                'title': f"🏆 {segment}",
                                'emoji': '🎁',
                                'action': 'Reward & Retain Top Performers',
                                'tactics': [
                                    'Launch exclusive VIP program with tiered benefits',
                                    'Personal concierge service for high-value orders',
                                    'Early access to new collections and limited editions',
                                    'Annual appreciation events and milestone rewards',
                                    'Referral incentives with premium rewards',
                                    'Custom product recommendations based on purchase history'
                                ],
                                'color': '#10b981',
                                'risk': 'Low',
                                'priority': '⭐⭐⭐⭐⭐'
                            }
                        elif segment == 'RE-ENGAGE':
                            strategy = {
                                'title': f"🔔 {segment}",
                                'emoji': '💌',
                                'action': 'Win-Back Campaign',
                                'tactics': [
                                    'Personalized "We miss you" email campaigns',
                                    '20% win-back discount with urgency messaging',
                                    'Survey to understand churn reasons',
                                    'Showcase new products and features launched since last visit',
                                    'Limited-time free shipping offers',
                                    'Re-engagement series with increasing incentives'
                                ],
                                'color': '#ef4444',
                                'risk': 'High',
                                'priority': '⭐⭐⭐⭐'
                            }
                        elif segment == 'RETAIN':
                            strategy = {
                                'title': f"💎 {segment}",
                                'emoji': '🔒',
                                'action': 'Loyalty Reinforcement',
                                'tactics': [
                                    'Points-based loyalty program with tier progression',
                                    'Monthly exclusive member-only offers',
                                    'Product bundles with complementary items',
                                    'Birthday and anniversary special discounts',
                                    'Early bird access to seasonal sales',
                                    'Community engagement through social media'
                                ],
                                'color': '#4F8BF9',
                                'risk': 'Medium',
                                'priority': '⭐⭐⭐⭐'
                            }
                        elif segment == 'NURTURE':
                            strategy = {
                                'title': f"🌱 {segment}",
                                'emoji': '📈',
                                'action': 'Growth & Development',
                                'tactics': [
                                    'Educational email series about product value',
                                    'New customer onboarding with progressive offers',
                                    'Gamified engagement (progress trackers, achievements)',
                                    'Social proof through reviews and testimonials',
                                    'Time-limited upgrade offers to boost AOV',
                                    'Cross-sell campaigns for complementary products'
                                ],
                                'color': '#f59e0b',
                                'risk': 'Medium',
                                'priority': '⭐⭐⭐'
                            }
                        else:
                            strategy = {
                                'title': f"📊 {segment}",
                                'emoji': '🎯',
                                'action': 'Custom Strategy Required',
                                'tactics': [
                                    'Deep-dive analysis of segment characteristics',
                                    'A/B test different engagement approaches',
                                    'Monitor behavioral patterns over time',
                                    'Develop tailored messaging',
                                    'Test multiple channel strategies',
                                    'Iterate based on performance data'
                                ],
                                'color': '#64748b',
                                'risk': 'Unknown',
                                'priority': '⭐⭐'
                            }
                        
                        with st.expander(f"**{strategy['title']}** - {cust_count:,} customers (${total_rev:,.0f} revenue)", expanded=True):
                            col1, col2 = st.columns([1, 3])
                            
                            with col1:
                                st.markdown(f"<div style='font-size: 4rem; text-align: center;'>{strategy['emoji']}</div>", unsafe_allow_html=True)
                                st.metric("Avg CLV", f"${clv:,.0f}")
                                st.caption(f"Risk Level: {strategy['risk']}")
                                st.markdown(f"**Priority:** {strategy['priority']}")
                            
                            with col2:
                                st.markdown(f"#### 🎯 Strategy: **{strategy['action']}**")
                                st.markdown("**Recommended Tactics:**")
                                for tactic in strategy['tactics']:
                                    st.markdown(f"- {tactic}")

                    # Global Priority Actions
                    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                    st.markdown("### 🚀 Immediate Action Plan")
                    
                    col_act1, col_act2, col_act3 = st.columns(3)
                    
                    with col_act1:
                        st.markdown("""
                        <div class="insight-box" style="border-left-color: #10b981;">
                            <h4>🥇 Priority 1: Protect Revenue</h4>
                            <p><strong>Focus:</strong> REWARD Segment</p>
                            <p>These customers drive the majority of your cash flow. Ensure they feel valued immediately.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_act2:
                        st.markdown("""
                        <div class="insight-box" style="border-left-color: #ef4444;">
                            <h4>🥈 Priority 2: Stop the Bleeding</h4>
                            <p><strong>Focus:</strong> RE-ENGAGE Segment</p>
                            <p>High churn risk. Activate win-back campaigns this week before they are lost.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_act3:
                        st.markdown("""
                        <div class="insight-box" style="border-left-color: #4F8BF9;">
                            <h4>🥉 Priority 3: Drive Frequency</h4>
                            <p><strong>Focus:</strong> RETAIN Segment</p>
                            <p>They buy, but not often enough. Implement loyalty triggers to shorten the purchase cycle.</p>
                        </div>
                        """, unsafe_allow_html=True)

                # =================================================================
                # TAB 4: ADVANCED ANALYTICS (REPLACED COHORT WITH HEATMAP)
                # =================================================================
                with tab_advanced:
                    if show_advanced_charts:
                        st.markdown("### 🔥 Peak Trading Times Heatmap")
                        st.markdown("Analyze transaction density by **Day of Week** and **Hour of Day** to optimize staffing and ad scheduling.")
                        
                        with st.spinner("Calculating sales patterns..."):
                            # CALLING THE NEW FUNCTION
                            sales_heatmap = calculate_sales_heatmap(clean_df)
                            
                            fig_heat = go.Figure(data=go.Heatmap(
                                z=sales_heatmap.values,
                                x=sales_heatmap.columns,
                                y=sales_heatmap.index,
                                colorscale='Magma',
                                hoverongaps=False
                            ))
                            
                            fig_heat.update_layout(
                                title="Sales Intensity: Day vs Hour",
                                xaxis_title="Hour of Day",
                                yaxis_title="Day of Week",
                                height=500,
                                template=chart_template
                            )
                            
                            st.plotly_chart(fig_heat, use_container_width=True)
                            
                            st.info("💡 **Insight:** Darker areas indicate peak trading hours. Use this data to schedule flash sales or customer support shifts.")

                        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                        
                        st.markdown("### 🛍️ Product Affinity & Performance")
                        col_prod1, col_prod2 = st.columns([2, 1])
                        
                        with col_prod1:
                            product_df = calculate_product_affinity(clean_df, top_n=15)
                            
                            fig_prod = go.Figure()
                            fig_prod.add_trace(go.Bar(
                                x=product_df['TotalRevenue'],
                                y=product_df['Product'],
                                orientation='h',
                                name='Revenue',
                                marker_color='#4F8BF9'
                            ))
                            
                            fig_prod.update_layout(
                                title="Top 15 Products by Revenue",
                                xaxis_title="Total Revenue ($)",
                                yaxis={'categoryorder':'total ascending'},
                                height=500,
                                template=chart_template
                            )
                            st.plotly_chart(fig_prod, use_container_width=True)
                            
                        with col_prod2:
                            st.markdown("#### 💡 Product Insights")
                            top_prod = product_df.iloc[0]
                            st.info(
                                f"**Best Seller:**\n\n"
                                f"{top_prod['Product']}\n\n"
                                f"Generates **${top_prod['TotalRevenue']:,.0f}** from **{top_prod['Orders']:,}** orders."
                            )
                            
                            avg_val = product_df['AvgOrderValue'].mean()
                            st.metric("Avg Item Value (Top 15)", f"${avg_val:.2f}")

                        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                        
                        st.markdown("### ⏳ Customer Lifespan Metrics")
                        cust_metrics = calculate_customer_lifetime_metrics(clean_df)
                        
                        col_life1, col_life2 = st.columns(2)
                        
                        with col_life1:
                            fig_life = px.histogram(
                                cust_metrics[cust_metrics['CustomerLifespanDays'] > 0], 
                                x='CustomerLifespanDays',
                                nbins=50,
                                title="Distribution of Customer Lifespan (Days)",
                                color_discrete_sequence=['#8b5cf6'],
                                template=chart_template
                            )
                            st.plotly_chart(fig_life, use_container_width=True)
                            
                        with col_life2:
                            fig_inter = px.histogram(
                                cust_metrics[cust_metrics['AvgDaysBetweenOrders'] > 0], 
                                x='AvgDaysBetweenOrders',
                                nbins=50,
                                title="Average Days Between Orders",
                                color_discrete_sequence=['#ec4899'],
                                template=chart_template
                            )
                            st.plotly_chart(fig_inter, use_container_width=True)

                    else:
                        st.warning("⚠️ Advanced Analytics are disabled. Please enable them in the sidebar.")

                # =================================================================
                # TAB 5: EXPORT DATA
                # =================================================================
                with tab_export:
                    st.markdown("### 📥 Download Results")
                    st.success("Analysis complete. You can download the processed datasets below for further use in Excel, PowerBI, or CRM tools.")
                    
                    col_dl1, col_dl2 = st.columns(2)
                    
                    with col_dl1:
                        # Prepare Main Export
                        csv_main = final_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📄 Download Segmented Customer Data",
                            data=csv_main,
                            file_name=f'customer_segments_{dt.datetime.now().strftime("%Y%m%d")}.csv',
                            mime='text/csv',
                            key='dl_main'
                        )
                        st.markdown("**Contains:** Customer ID, RFM Scores, Clusters, Segment Names.")

                    with col_dl2:
                        # Prepare Summary Export
                        csv_sum = summary_display.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📊 Download Segment Summary Stats",
                            data=csv_sum,
                            file_name=f'segment_summary_{dt.datetime.now().strftime("%Y%m%d")}.csv',
                            mime='text/csv',
                            key='dl_sum'
                        )
                        st.markdown("**Contains:** Aggregated metrics, averages, and total revenue per segment.")

                    st.markdown("---")
                    st.markdown("### 👁️ Preview Data")
                    st.dataframe(final_df.head(100), use_container_width=True)

    else:
        # =====================================================================
        # EMPTY STATE (NO FILE UPLOADED)
        # =====================================================================
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">📂</div>
            <div class="empty-state-title">Ready to Analyze</div>
            <div class="empty-state-subtitle">
                Upload your transaction data (CSV or Excel) using the sidebar to generate insights.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🚀 Capabilities Overview")
        
        col_feat1, col_feat2, col_feat3 = st.columns(3)
        
        with col_feat1:
            st.markdown("""
            <div class="stat-item">
                <div style="font-size: 2rem;">🧠</div>
                <div style="font-weight: 600; margin-bottom: 0.5rem;">AI Segmentation</div>
                <div style="font-size: 0.9rem; color: #666;">
                    Automatic customer grouping using K-Means clustering and RFM logic.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_feat2:
            st.markdown("""
            <div class="stat-item">
                <div style="font-size: 2rem;">🔥</div>
                <div style="font-weight: 600; margin-bottom: 0.5rem;">Sales Heatmaps</div>
                <div style="font-size: 0.9rem; color: #666;">
                    Identify peak trading hours and days instantly.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_feat3:
            st.markdown("""
            <div class="stat-item">
                <div style="font-size: 2rem;">💡</div>
                <div style="font-weight: 600; margin-bottom: 0.5rem;">Strategic Insights</div>
                <div style="font-size: 0.9rem; color: #666;">
                    Get actionable, segment-specific marketing tactics instantly.
                </div>
            </div>
            """, unsafe_allow_html=True)

# =============================================================================
# 4. RUN APPLICATION
# =============================================================================
if __name__ == "__main__":
    main()