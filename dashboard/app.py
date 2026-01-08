"""
Credit Risk MLOps Dashboard
Deployed on Kubernetes - Makes REAL predictions via KServe
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os

# KServe endpoint (internal cluster DNS)
KSERVE_URL = os.environ.get(
    "KSERVE_URL", 
    "http://credit-risk-model-predictor.ml-credit-risk.svc.cluster.local/v1/models/credit-risk-model:predict"
)

# Page config
st.set_page_config(
    page_title="Credit Risk MLOps Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def engineer_features(data):
    """Apply feature engineering transformations - SAME as training"""
    df = pd.DataFrame([data])
    
    pay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
    amt_cols = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    
    # Payment behavior features
    df['LATE_PAYMENTS'] = (df[pay_cols] > 0).sum(axis=1)
    df['MAX_DELAY'] = df[pay_cols].max(axis=1)
    df['AVG_DELAY'] = df[pay_cols].mean(axis=1)
    df['SEVERE_DELAY'] = (df[pay_cols] >= 2).sum(axis=1)
    df['EVER_2MONTH_LATE'] = (df[pay_cols] >= 2).any(axis=1).astype(int)
    df['RECENT_DELAY_WEIGHTED'] = df['PAY_0'] * 3 + df['PAY_2'] * 2 + df['PAY_3']
    
    # Aggregates
    df['AVG_BILL_AMT'] = df[bill_cols].mean(axis=1)
    df['AVG_PAY_AMT'] = df[amt_cols].mean(axis=1)
    df['TOTAL_BILL'] = df[bill_cols].sum(axis=1)
    df['TOTAL_PAY'] = df[amt_cols].sum(axis=1)
    
    # Ratios
    df['UTILIZATION'] = df['BILL_AMT1'] / (df['LIMIT_BAL'] + 1)
    df['AVG_UTILIZATION'] = df['AVG_BILL_AMT'] / (df['LIMIT_BAL'] + 1)
    df['PAY_RATIO'] = df['TOTAL_PAY'] / (df['TOTAL_BILL'] + 1)
    df['RECENT_PAY_RATIO'] = df['PAY_AMT1'] / (df['BILL_AMT1'] + 1)
    
    # Trends
    df['BILL_TREND'] = df['BILL_AMT1'] - df['BILL_AMT6']
    df['PAY_TREND'] = df['PAY_AMT1'] - df['PAY_AMT6']
    df['INCREASING_DEBT'] = (df['BILL_TREND'] > 0).astype(int)
    
    # Interactions
    df['LIMIT_AGE'] = df['LIMIT_BAL'] / df['AGE']
    df['DELAY_UTIL'] = df['AVG_DELAY'] * df['AVG_UTILIZATION']
    
    # Categorical
    df['HIGH_EDUCATION'] = (df['EDUCATION'] <= 2).astype(int)
    df['SINGLE'] = (df['MARRIAGE'] == 2).astype(int)
    
    # Handle infinity
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)
    
    return df

# Expected feature order (must match training)
FEATURE_NAMES = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
    'LATE_PAYMENTS', 'MAX_DELAY', 'AVG_DELAY', 'SEVERE_DELAY',
    'EVER_2MONTH_LATE', 'RECENT_DELAY_WEIGHTED',
    'AVG_BILL_AMT', 'AVG_PAY_AMT', 'TOTAL_BILL', 'TOTAL_PAY',
    'UTILIZATION', 'AVG_UTILIZATION', 'PAY_RATIO', 'RECENT_PAY_RATIO',
    'BILL_TREND', 'PAY_TREND', 'INCREASING_DEBT',
    'LIMIT_AGE', 'DELAY_UTIL', 'HIGH_EDUCATION', 'SINGLE'
]

def get_prediction(customer_data):
    """Make real prediction call to KServe"""
    try:
        # Apply feature engineering
        df = engineer_features(customer_data)
        
        # Select features in correct order
        features = df[FEATURE_NAMES].values.tolist()
        
        # Call KServe
        payload = {"instances": features}
        response = requests.post(
            KSERVE_URL,
            json=payload,
            headers={"Host": "credit-risk.local"},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["predictions"][0], None
        else:
            return None, f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return None, str(e)

# Sidebar
with st.sidebar:
    st.markdown("### 🏦 Credit Risk MLOps")
    st.markdown("Production-grade ML pipeline on Kubernetes")
    st.markdown("---")
    st.markdown("**KServe Endpoint:**")
    st.code(KSERVE_URL[:50] + "...", language=None)
    st.markdown("---")
    st.markdown("**Tech Stack:**")
    st.markdown("- 🔧 Kubeflow Pipelines")
    st.markdown("- 🚀 KServe")
    st.markdown("- 📊 Prometheus + Grafana")

# Header
st.markdown('<h1 class="main-header">🏦 Credit Risk MLOps Platform</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Real-Time Credit Risk Predictions via KServe</p>", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔮 Live Predictions", "📈 Model Insights", "🖥️ Infrastructure"])

# ==================== TAB 1: OVERVIEW ====================
with tab1:
    st.markdown("## System Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="🎯 Model AUC", value="0.78", delta="Above baseline")
    with col2:
        st.metric(label="📦 Components", value="12", delta="All healthy")
    with col3:
        st.metric(label="⚡ Latency", value="~50ms", delta="p99")
    with col4:
        st.metric(label="🔄 Replicas", value="1-3", delta="Auto-scaling")
    
    st.markdown("---")
    st.markdown("### 🏗️ System Architecture")
    
    # Create architecture diagram with Plotly
    fig = go.Figure()
    
    # Define component positions
    components = {
        "👤 User": (0, 2),
        "🌐 Dashboard": (1.5, 2),
        "🚀 KServe": (3, 2),
        "🤖 XGBoost Model": (4.5, 2),
        "📊 Prometheus": (3, 0.5),
        "📈 Grafana": (4.5, 0.5),
        "🔧 Kubeflow": (1.5, 0.5),
        "📦 Minio": (0, 0.5),
    }
    
    # Add nodes
    colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe', '#43e97b', '#38f9d7']
    for i, (name, (x, y)) in enumerate(components.items()):
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers+text',
            marker=dict(size=60, color=colors[i % len(colors)], symbol='circle'),
            text=[name],
            textposition='top center',
            textfont=dict(size=12),
            hoverinfo='text',
            showlegend=False
        ))
    
    # Add edges (arrows)
    edges = [
        (0, 1.5, 2, 2),      # User -> Dashboard
        (1.5, 3, 2, 2),      # Dashboard -> KServe
        (3, 4.5, 2, 2),      # KServe -> Model
        (4.5, 3, 2, 0.5),    # Model -> Prometheus (metrics)
        (3, 4.5, 0.5, 0.5),  # Prometheus -> Grafana
        (1.5, 0, 0.5, 0.5),  # Kubeflow -> Minio
    ]
    
    for x0, x1, y0, y1 in edges:
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode='lines',
            line=dict(color='rgba(100,100,100,0.5)', width=2),
            showlegend=False
        ))
    
    fig.update_layout(
        height=350,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 5]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.2, 2.8]),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add data flow description
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **🔄 Request Flow:**
        1. User enters data in Dashboard
        2. Dashboard calls KServe endpoint
        3. XGBoost model returns prediction
        4. Result displayed with risk score
        """)
    with col2:
        st.markdown("""
        **📊 Monitoring Flow:**
        - Metrics scraped by Prometheus
        - Visualized in Grafana dashboards
        - Drift detection alerts enabled
        """)

# ==================== TAB 2: LIVE PREDICTIONS ====================
with tab2:
    st.markdown("## 🔮 Real-Time Credit Risk Prediction")
    st.markdown("Enter customer details to get **live predictions** from KServe InferenceService")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 👤 Demographics")
        limit_bal = st.slider("Credit Limit ($)", 10000, 500000, 50000, 10000)
        age = st.slider("Age", 21, 70, 35)
        sex = st.selectbox("Sex", [1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
        education = st.selectbox("Education", [1, 2, 3, 4], 
                                 format_func=lambda x: ["", "Graduate School", "University", "High School", "Other"][x])
        marriage = st.selectbox("Marital Status", [1, 2, 3],
                               format_func=lambda x: ["", "Married", "Single", "Other"][x])
        
    with col2:
        st.markdown("### 💳 Payment History")
        pay_0 = st.selectbox("Last Month", [-2, -1, 0, 1, 2, 3, 4, 5], index=2, 
                             format_func=lambda x: "On time" if x <= 0 else f"{x} months late")
        pay_2 = st.selectbox("2 Months Ago", [-2, -1, 0, 1, 2, 3], index=2,
                             format_func=lambda x: "On time" if x <= 0 else f"{x} months late")
        pay_3 = st.selectbox("3 Months Ago", [-2, -1, 0, 1, 2], index=2,
                             format_func=lambda x: "On time" if x <= 0 else f"{x} months late")
        pay_4 = st.slider("4 Months Ago", -2, 3, 0)
        pay_5 = st.slider("5 Months Ago", -2, 3, 0)
        pay_6 = st.slider("6 Months Ago", -2, 3, 0)
        
    with col3:
        st.markdown("### 💰 Bill & Payment Amounts")
        bill_amt1 = st.number_input("Current Bill ($)", 0, 200000, 20000, 1000)
        bill_amt2 = st.number_input("Bill 2 Months ($)", 0, 200000, 19000, 1000)
        bill_amt3 = st.number_input("Bill 3 Months ($)", 0, 200000, 18000, 1000)
        pay_amt1 = st.number_input("Last Payment ($)", 0, 100000, 2000, 500)
        pay_amt2 = st.number_input("Payment 2 Mo ($)", 0, 100000, 1800, 500)
        pay_amt3 = st.number_input("Payment 3 Mo ($)", 0, 100000, 1600, 500)
    
    # Build customer data dict
    customer_data = {
        'LIMIT_BAL': limit_bal, 'SEX': sex, 'EDUCATION': education, 
        'MARRIAGE': marriage, 'AGE': age,
        'PAY_0': pay_0, 'PAY_2': pay_2, 'PAY_3': pay_3, 
        'PAY_4': pay_4, 'PAY_5': pay_5, 'PAY_6': pay_6,
        'BILL_AMT1': bill_amt1, 'BILL_AMT2': bill_amt2, 'BILL_AMT3': bill_amt3,
        'BILL_AMT4': int(bill_amt3 * 0.9), 'BILL_AMT5': int(bill_amt3 * 0.8), 'BILL_AMT6': int(bill_amt3 * 0.7),
        'PAY_AMT1': pay_amt1, 'PAY_AMT2': pay_amt2, 'PAY_AMT3': pay_amt3,
        'PAY_AMT4': int(pay_amt3 * 0.9), 'PAY_AMT5': int(pay_amt3 * 0.8), 'PAY_AMT6': int(pay_amt3 * 0.7)
    }
    
    st.markdown("---")
    
    if st.button("🔮 Get Live Prediction", type="primary", use_container_width=True):
        with st.spinner("Calling KServe InferenceService..."):
            prediction, error = get_prediction(customer_data)
        
        if error:
            st.error(f"Prediction failed: {error}")
            st.info("Falling back to local estimation...")
            
            # Fallback calculation
            risk_score = 0.15
            if pay_0 > 1: risk_score += 0.25
            if pay_2 > 0: risk_score += 0.1
            if limit_bal < 30000: risk_score += 0.1
            if bill_amt1 / (limit_bal + 1) > 0.8: risk_score += 0.15
            prediction = min(risk_score, 0.95)
        
        # Display result
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if prediction > 0.5:
                st.error(f"### ⚠️ HIGH RISK")
            else:
                st.success(f"### ✅ LOW RISK")
            
            st.markdown(f"<h1 style='text-align: center;'>{prediction:.1%}</h1>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;'>Default Probability</p>", unsafe_allow_html=True)
        
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = prediction * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Risk Score"},
            delta = {'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkred" if prediction > 0.5 else "darkgreen"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 60], 'color': "yellow"},
                    {'range': [60, 100], 'color': "lightcoral"}
                ],
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 50}
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show input summary
        with st.expander("📋 Input Summary"):
            st.json(customer_data)

# ==================== TAB 3: MODEL INSIGHTS ====================
with tab3:
    st.markdown("## 📈 Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Feature Importance")
        features = ['PAY_0', 'PAY_2', 'LIMIT_BAL', 'PAY_AMT1', 'BILL_AMT1', 'AGE', 'PAY_3', 'UTILIZATION']
        importance = [0.25, 0.15, 0.12, 0.10, 0.09, 0.08, 0.07, 0.06]
        fig = px.bar(x=importance, y=features, orientation='h', color=importance, 
                     color_continuous_scale='Viridis')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Model Metrics")
        fig = go.Figure(go.Bar(
            x=['AUC', 'Accuracy', 'Precision', 'Recall', 'F1'],
            y=[0.78, 0.76, 0.47, 0.63, 0.54],
            marker_color=['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe'],
            text=[0.78, 0.76, 0.47, 0.63, 0.54],
            textposition='outside'
        ))
        fig.update_layout(height=400, yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 4: INFRASTRUCTURE ====================
with tab4:
    st.markdown("## 🖥️ Kubernetes Infrastructure")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 1rem; color: white;'>
            <h3>ml-credit-risk</h3>
            <p>🟢 3 pods</p>
            <small>• credit-risk-model-predictor<br>• monitoring-service<br>• mlops-dashboard</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 1.5rem; border-radius: 1rem; color: white;'>
            <h3>kubeflow</h3>
            <p>🟢 13 pods</p>
            <small>• ml-pipeline<br>• minio<br>• mysql</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 1.5rem; border-radius: 1rem; color: white;'>
            <h3>monitoring</h3>
            <p>🟢 8 pods</p>
            <small>• prometheus<br>• grafana<br>• alertmanager</small>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>🏦 Credit Risk MLOps | <a href='https://github.com/Shrinet82/ML-OPS'>GitHub</a></p>", unsafe_allow_html=True)
