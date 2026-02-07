import streamlit as st
import pandas as pd
import boto3
import joblib
import plotly.express as px
import json
from io import BytesIO
from snowflake.connector import connect
import os

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(page_title="Churn Prediction Engine", page_icon="⚡", layout="wide")

st.markdown("""
<style>
    .metric-card {background-color: #1e1e1e; padding: 20px; border-radius: 10px; border: 1px solid #333;}
    .stButton>button {width: 100%;}
    .risk-tag {padding: 2px 8px; border-radius: 4px; font-size: 0.8em; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# CONFIGURATION
# ==========================================
def get_config(key):
    return os.environ.get(key) or st.secrets.get(key)

try:
    SNOWFLAKE_USER = get_config("SNOWFLAKE_USER")
    SNOWFLAKE_PASSWORD = get_config("SNOWFLAKE_PASSWORD")
    SNOWFLAKE_ACCOUNT = get_config("SNOWFLAKE_ACCOUNT")
    SNOWFLAKE_WH = "CHURN_PROJECT_WH_M"
    SNOWFLAKE_DB = "RETAIL_ANALYTICS"
    SNOWFLAKE_SCHEMA = "CHURN_PREDICTION"
    BUCKET_NAME = get_config("BUCKET_NAME")
    MODEL_KEY = "models/best_churn_model.pkl"
    LEADERBOARD_KEY = "models/leaderboard.json"
    AWS_REGION = get_config("AWS_REGION") or "us-east-1"
    AWS_ACCESS_KEY = get_config("AWS_ACCESS_KEY_ID")
    AWS_SECRET_KEY = get_config("AWS_SECRET_ACCESS_KEY")
except:
    st.error("Configuration Error. Check Secrets.")
    st.stop()

# ==========================================
# DATA LOADING
# ==========================================
def get_s3_client():
    if AWS_ACCESS_KEY and AWS_SECRET_KEY:
        return boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY, region_name=AWS_REGION)
    return boto3.client('s3', region_name=AWS_REGION)

@st.cache_resource(ttl=60)
def load_resources_from_s3():
    s3 = get_s3_client()
    resources = {"model": None, "leaderboard": []}
    try:
        with st.spinner('Fetching Champion Model...'):
            response = s3.get_object(Bucket=BUCKET_NAME, Key=MODEL_KEY)
            resources["model"] = joblib.load(BytesIO(response['Body'].read()))
        try:
            response_lb = s3.get_object(Bucket=BUCKET_NAME, Key=LEADERBOARD_KEY)
            resources["leaderboard"] = json.loads(response_lb['Body'].read().decode('utf-8'))
        except: pass
    except Exception as e:
        st.warning(f"Waiting for Model... Error: {e}")
    return resources

def get_snowflake_data():
    conn = connect(
        user=SNOWFLAKE_USER,
        password=SNOWFLAKE_PASSWORD,
        account=SNOWFLAKE_ACCOUNT,
        warehouse=SNOWFLAKE_WH,
        database=SNOWFLAKE_DB,
        schema=SNOWFLAKE_SCHEMA
    )
    # Order by interaction to prioritize Active Users
    query = """
    SELECT * FROM CUSTOMER_FEATURES_VIEW 
    ORDER BY RT_TOTAL_INTERACTIONS DESC, CUSTOMERID 
    LIMIT 2000
    """
    df = pd.read_sql(query, conn)
    conn.close()
    df.columns = [x.upper() for x in df.columns]
    
    if not df.empty:
        # Recreate Text Labels
        def get_contract(row):
            if row.get('IS_MONTH_TO_MONTH') == 1: return 'Month-to-month'
            if row.get('IS_TWO_YEAR') == 1: return 'Two year'
            return 'One year'
        df['CONTRACT_TEXT'] = df.apply(get_contract, axis=1)
        df['PAYMENT_TEXT'] = df['PAYS_VIA_ECHECK'].apply(lambda x: 'Electronic Check' if x == 1 else 'Auto-Pay / Other')
        df['CHURN_TEXT'] = df['CHURN_LABEL'].apply(lambda x: 'Churned' if x == 1 else 'Active')

        # --- GENERATE EXPLAINABLE RISK REASONS ---
        def get_risk_reason(row):
            reasons = []
            # Real-Time Signals
            if row.get('RT_CHECKOUT_ERRORS', 0) > 0: reasons.append("⚠️ Checkout Error")
            if row.get('RT_CANCELLATION_INTENT', 0) == 1: reasons.append("🛑 Visited Cancel Page")
            if row.get('RT_TOTAL_INTERACTIONS', 0) > 5: reasons.append("👀 High Activity Spike")
            
            # Static Signals
            if row.get('IS_MONTH_TO_MONTH') == 1: reasons.append("📅 No Contract")
            if row.get('MONTHLY_CHARGES', 0) > 90: reasons.append("💸 High Bill")
            if row.get('TENURE_MONTHS', 0) < 3: reasons.append("👶 New Customer")
            
            return " | ".join(reasons) if reasons else "Normal Behavior"
            
        df['RISK_FACTORS'] = df.apply(get_risk_reason, axis=1)

    return df.fillna(0)

# ==========================================
# MAIN UI
# ==========================================
st.title("⚡ Real-Time Churn Analytics Engine")

col1, col2 = st.columns([1, 6])
with col1:
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
with col2:
    st.caption("Last updated: Just now | Connected to Snowflake & AWS S3")

resources = load_resources_from_s3()
model = resources["model"]
leaderboard = resources["leaderboard"]
df = get_snowflake_data()

# Prediction Logic
if model is not None and not df.empty:
    X = df.drop(columns=['CUSTOMERID', 'CHURN_LABEL', 'CONTRACT_TEXT', 'PAYMENT_TEXT', 'CHURN_TEXT', 'RISK_FACTORS'], errors='ignore')
    try:
        df['PREDICTED_CHURN'] = model.predict(X)
        df['CHURN_PROBABILITY'] = model.predict_proba(X)[:, 1]
    except Exception as e:
        st.error(f"Prediction Error: {e}")

# ==========================================
# TABS LAYOUT
# ==========================================
# NEW ORDER: 1. Data -> 2. Model -> 3. Action
tab1, tab2, tab3 = st.tabs(["📊 Data Explorer (EDA)", "🧠 Model Performance", "🚨 Live Operations"])

# --- TAB 1: EDA ---
with tab1:
    st.header("Exploratory Data Analysis")
    st.markdown("Understanding **Who** is leaving and **Why**.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Rows Fetched", len(df))
    c2.metric("Features Available", len(df.columns))
    c3.metric("Historical Churn Rate", f"{df['CHURN_LABEL'].mean():.1%}")

    st.subheader("1. What drives Churn?")
    row2_1, row2_2 = st.columns(2)
    
    with row2_1:
        # VISUAL 1: Churn Rate by Payment Method
        pay_churn = df.groupby('PAYMENT_TEXT')['CHURN_LABEL'].mean().reset_index()
        pay_churn['CHURN_LABEL'] = pay_churn['CHURN_LABEL'] * 100 
        
        fig_pay = px.bar(pay_churn, x="PAYMENT_TEXT", y="CHURN_LABEL", 
                         title="Churn Rate % by Payment Method",
                         color="PAYMENT_TEXT",
                         text_auto='.1f',
                         labels={'CHURN_LABEL': 'Churn Rate (%)'})
        st.plotly_chart(fig_pay, use_container_width=True)

    with row2_2:
        # VISUAL 2: Churn Rate by Contract
        cont_churn = df.groupby('CONTRACT_TEXT')['CHURN_LABEL'].mean().reset_index()
        cont_churn['CHURN_LABEL'] = cont_churn['CHURN_LABEL'] * 100
        
        fig_contract = px.bar(cont_churn, x="CONTRACT_TEXT", y="CHURN_LABEL", 
                              title="Churn Rate % by Contract",
                              color="CONTRACT_TEXT",
                              text_auto='.1f',
                              labels={'CHURN_LABEL': 'Churn Rate (%)'})
        st.plotly_chart(fig_contract, use_container_width=True)

    st.markdown("---")
    st.subheader("2. Financial Impact Analysis")
    # VISUAL 3: Monthly Charges
    fig_hist = px.histogram(df, x="MONTHLY_CHARGES", color="CHURN_TEXT", 
                            title="Churn by Monthly Bill Amount (Bimodal Distribution)",
                            barmode="overlay", opacity=0.7, nbins=50,
                            color_discrete_map={'Churned': '#ef4444', 'Active': '#3b82f6'})
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # RESTORED: Context Labels
    col_desc1, col_desc2 = st.columns(2)
    with col_desc1:
        st.info("📉 **Left Cluster ($20):** **Budget Users**. Likely Phone-only service. They have low churn because the service is cheap.")
    with col_desc2:
        st.error("📈 **Right Cluster ($70-$100):** **Premium Users**. Fiber Optic + Streaming bundles. This group has **High Churn** and represents the biggest revenue risk.")

# --- TAB 2: MODEL PERFORMANCE ---
with tab2:
    st.header("AI Model Performance")
    st.markdown("How the **Multi-Model Tournament** picked the winner.")

    if leaderboard:
        st.subheader("🏆 Tournament Leaderboard")
        lb_df = pd.DataFrame(leaderboard)
        def highlight_winner(row):
            return ['background-color: #10b981; color: white' if row['Selected'] == '🏆 WINNER' else '' for _ in row]
        st.dataframe(lb_df.style.apply(highlight_winner, axis=1), use_container_width=True, hide_index=True)
    
    st.subheader("Feature Importance")
    col_fi, col_risk = st.columns(2)
    with col_fi:
        try:
            estimator = model.named_steps['clf']
            if hasattr(estimator, 'feature_importances_'):
                feat_imp = pd.DataFrame({'Feature': X.columns, 'Importance': estimator.feature_importances_}).sort_values(by='Importance', ascending=False).head(10)
                fig_imp = px.bar(feat_imp, x='Importance', y='Feature', orientation='h', title="Top Predictors")
                st.plotly_chart(fig_imp, use_container_width=True)
            elif hasattr(estimator, 'coef_'):
                 coeffs = estimator.coef_[0]
                 feat_imp = pd.DataFrame({'Feature': X.columns, 'Importance': abs(coeffs)}).sort_values(by='Importance', ascending=False).head(10)
                 fig_imp = px.bar(feat_imp, x='Importance', y='Feature', orientation='h', title="Feature Impact (Coefficients)")
                 st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.info("Visuals not available for this model type.")
        except:
            st.info("Feature importance data unavailable.")

    with col_risk:
        st.markdown("##### Risk vs. Customer Tenure")
        fig_scatter = px.scatter(df, x="TENURE_MONTHS", y="CHURN_PROBABILITY", color="PREDICTED_CHURN", 
                               opacity=0.6, title="Does Loyalty Reduce Risk?")
        st.plotly_chart(fig_scatter, use_container_width=True)

# --- TAB 3: LIVE OPERATIONS (ACTIONABLE) ---
with tab3:
    st.header("Live Operations Center")
    
    # FILTER: ONLY Active Customers
    active_customers_df = df[df['CHURN_LABEL'] == 0].copy()
    
    # 1. KEY METRICS
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Avg Risk (Active Users)", f"{active_customers_df['CHURN_PROBABILITY'].mean():.1%}")
    
    # High Risk Count
    high_risk_df = active_customers_df[active_customers_df['CHURN_PROBABILITY'] > 0.7]
    m3.metric("High Risk Users (>70%)", len(high_risk_df), delta_color="inverse")
    
    # Active Sessions
    rt_col = 'RT_TOTAL_INTERACTIONS' if 'RT_TOTAL_INTERACTIONS' in df.columns else df.columns[0]
    rt_traffic = df[df[rt_col] > 0].copy()
    m4.metric("Active Sessions (Now)", len(rt_traffic), delta_color="normal")

    # 2. DRILL DOWN: HIGH RISK USERS
    # This allows clicking to see the 25 users mentioned in your query
    with st.expander(f"⚠️ View Details: {len(high_risk_df)} High Risk Customers (>70%)", expanded=False):
        st.dataframe(
            high_risk_df[['CUSTOMERID', 'CHURN_PROBABILITY', 'RISK_FACTORS', 'TENURE_MONTHS', 'MONTHLY_CHARGES']]
            .sort_values(by='CHURN_PROBABILITY', ascending=False)
            .style.background_gradient(subset=['CHURN_PROBABILITY'], cmap='Reds')
            .format({'CHURN_PROBABILITY': "{:.1%}", 'MONTHLY_CHARGES': "${:.2f}"}),
            use_container_width=True
        )

    # 3. REAL-TIME TABLE
    st.markdown("---")
    st.subheader("🔴 Real-Time Active Sessions")
    st.caption("Users currently interacting with the platform. Sorted by activity level.")
    
    if not rt_traffic.empty:
        rt_display_cols = ['CUSTOMERID', 'CHURN_PROBABILITY', 'RISK_FACTORS', 'RT_TOTAL_INTERACTIONS', 'RT_CANCELLATION_INTENT', 'TENURE_MONTHS', 'MONTHLY_CHARGES']
        rt_valid_cols = [c for c in rt_display_cols if c in rt_traffic.columns]
        
        st.dataframe(
            rt_traffic[rt_valid_cols].sort_values(by='RT_TOTAL_INTERACTIONS', ascending=False)
            .style.background_gradient(subset=['CHURN_PROBABILITY'], cmap='Reds')
            .format({'CHURN_PROBABILITY': "{:.1%}", 'MONTHLY_CHARGES': "${:.2f}"}),
            use_container_width=True
        )
    else:
        st.warning("No active users detected. Run the generator script to simulate traffic.")

    # 4. VIP WATCHLIST
    st.markdown("---")
    st.subheader("💎 VIP Watchlist Alerts")
    st.caption("High-Value Customers (Tenure > 2 Years & Bill > $70) showing signs of leaving (Prob > 50%).")
    
    # VIP Logic
    vip_mask = (active_customers_df['TENURE_MONTHS'] > 24) & (active_customers_df['MONTHLY_CHARGES'] > 70)
    vips = active_customers_df[vip_mask].copy()
    risky_vips = vips[vips['CHURN_PROBABILITY'] > 0.5]
    
    if not risky_vips.empty:
        st.error(f"🚨 {len(risky_vips)} VIPs need attention!")
        vip_cols = ['CUSTOMERID', 'CHURN_PROBABILITY', 'RISK_FACTORS', 'TENURE_MONTHS', 'MONTHLY_CHARGES']
        valid_vip_cols = [c for c in vip_cols if c in risky_vips.columns]
        
        st.dataframe(
            risky_vips[valid_vip_cols].sort_values(by='CHURN_PROBABILITY', ascending=False)
            .style.background_gradient(subset=['CHURN_PROBABILITY'], cmap='Reds')
            .format({'CHURN_PROBABILITY': "{:.1%}", 'MONTHLY_CHARGES': "${:.2f}"}),
            use_container_width=True
        )
    else:
        st.success("No VIPs currently exhibiting churn behavior.")
