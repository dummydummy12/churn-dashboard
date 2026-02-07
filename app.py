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
    
    # --- RECREATE TEXT LABELS FOR EDA ---
    if not df.empty:
        # Contract Text
        def get_contract(row):
            if row.get('IS_MONTH_TO_MONTH') == 1: return 'Month-to-month'
            if row.get('IS_TWO_YEAR') == 1: return 'Two year'
            return 'One year'
        df['CONTRACT_TEXT'] = df.apply(get_contract, axis=1)
        
        # Payment Text
        df['PAYMENT_TEXT'] = df['PAYS_VIA_ECHECK'].apply(lambda x: 'Electronic Check' if x == 1 else 'Auto-Pay / Other')
        
        # Churn Text
        df['CHURN_TEXT'] = df['CHURN_LABEL'].apply(lambda x: 'Churned' if x == 1 else 'Active')

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

if model is not None and not df.empty:
    X = df.drop(columns=['CUSTOMERID', 'CHURN_LABEL', 'CONTRACT_TEXT', 'PAYMENT_TEXT', 'CHURN_TEXT'], errors='ignore')
    try:
        df['PREDICTED_CHURN'] = model.predict(X)
        df['CHURN_PROBABILITY'] = model.predict_proba(X)[:, 1]
    except Exception as e:
        st.error(f"Prediction Error: {e}")

# ==========================================
# TABS LAYOUT
# ==========================================
tab1, tab2, tab3 = st.tabs(["📊 Data Explorer (EDA)", "🚨 Live Monitoring", "🧠 Model Performance"])

# --- TAB 1: EDA (CORRECTED STACKED BARS) ---
with tab1:
    st.header("Exploratory Data Analysis")
    st.markdown("Understanding **Who** is leaving and **Why**.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Rows Fetched", len(df))
    c2.metric("Features Available", len(df.columns))
    c3.metric("Historical Churn Rate", f"{df['CHURN_LABEL'].mean():.1%}")

    st.subheader("1. Where is the Churn coming from?")
    row2_1, row2_2 = st.columns(2)
    
    with row2_1:
        # VISUAL 1: Payment Method (Corrected: Stacked)
        # Removed barmode='group' -> Now it stacks correctly to show 100% split
        fig_pay = px.histogram(df, x="PAYMENT_TEXT", color="CHURN_TEXT", 
                               barnorm='percent',
                               title="Churn Rate by Payment Method",
                               color_discrete_map={'Churned': '#ef4444', 'Active': '#10b981'},
                               height=400)
        st.plotly_chart(fig_pay, use_container_width=True)
        st.info("💡 **Insight:** Notice the large Red bar for 'Electronic Check'. Manual payers leave faster.")

    with row2_2:
        # VISUAL 2: Contract Type (Corrected: Stacked)
        fig_contract = px.histogram(df, x="CONTRACT_TEXT", color="CHURN_TEXT", 
                                  barnorm='percent',
                                  title="Churn Rate by Contract Type", 
                                  color_discrete_map={'Churned': '#ef4444', 'Active': '#10b981'},
                                  height=400)
        st.plotly_chart(fig_contract, use_container_width=True)
        st.info("💡 **Insight:** Month-to-month contracts are ~40-50% Red (High Churn), while Two-Year contracts are almost entirely Green.")

    st.markdown("---")
    st.subheader("2. Financial Impact: Are we losing high-value customers?")
    
    # VISUAL 3: Monthly Charges
    fig_hist = px.histogram(df, x="MONTHLY_CHARGES", color="CHURN_TEXT", 
                            title="Distribution of Monthly Charges (Churn vs. Active)",
                            barmode="overlay", opacity=0.7, nbins=50,
                            color_discrete_map={'Churned': '#ef4444', 'Active': '#3b82f6'})
    st.plotly_chart(fig_hist, use_container_width=True)
    
    col_desc1, col_desc2 = st.columns(2)
    with col_desc1:
        st.caption("📉 **Left ($20):** Basic users. Mostly Blue (Active).")
    with col_desc2:
        st.caption("📈 **Right ($70+):** Premium users. Notice the Red spikes indicating higher churn.")

# --- TAB 2: LIVE MONITORING ---
with tab2:
    st.header("Live Operations Center")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Avg Churn Risk", f"{df['CHURN_PROBABILITY'].mean():.1%}")
    m3.metric("High Risk Users (>70%)", df[df['CHURN_PROBABILITY'] > 0.7].shape[0], delta_color="inverse")
    
    rt_col = 'RT_TOTAL_INTERACTIONS' if 'RT_TOTAL_INTERACTIONS' in df.columns else df.columns[0]
    active_users = df[df[rt_col] > 0].copy()
    m4.metric("Active Sessions (Now)", len(active_users), delta_color="normal")

    st.markdown("---")
    st.subheader("🔴 Real-Time Active Sessions")
    
    if not active_users.empty:
        rt_display_cols = ['CUSTOMERID', 'CHURN_PROBABILITY', 'RT_TOTAL_INTERACTIONS', 'RT_CHECKOUT_ERRORS', 'RT_CANCELLATION_INTENT']
        rt_valid_cols = [c for c in rt_display_cols if c in active_users.columns]
        
        st.dataframe(
            active_users[rt_valid_cols].sort_values(by='RT_TOTAL_INTERACTIONS', ascending=False)
            .style.background_gradient(subset=['CHURN_PROBABILITY'], cmap='Reds'),
            use_container_width=True
        )
    else:
        st.warning("No active users detected in the current window.")

    st.markdown("---")
    st.subheader("💎 VIP Watchlist Alerts")
    
    vip_mask = (df['TENURE_MONTHS'] > 24) & (df['MONTHLY_CHARGES'] > 70)
    vips = df[vip_mask].copy()
    risky_vips = vips[vips['CHURN_PROBABILITY'] > 0.5]
    
    if not risky_vips.empty:
        st.error(f"🚨 {len(risky_vips)} High-Value Customers at Risk!")
        vip_cols = ['CUSTOMERID', 'CHURN_PROBABILITY', 'TENURE_MONTHS', 'MONTHLY_CHARGES', 'RT_CHECKOUT_ERRORS']
        valid_vip_cols = [c for c in vip_cols if c in risky_vips.columns]
        st.dataframe(
            risky_vips[valid_vip_cols].sort_values(by='CHURN_PROBABILITY', ascending=False)
            .style.background_gradient(subset=['CHURN_PROBABILITY'], cmap='Reds')
            .format({'CHURN_PROBABILITY': "{:.1%}", 'MONTHLY_CHARGES': "${:.2f}"}),
            use_container_width=True
        )
    else:
        st.success("No VIPs currently exhibiting churn behavior.")

# --- TAB 3: MODEL PERFORMANCE ---
with tab3:
    st.header("AI Model Performance")

    if leaderboard:
        st.subheader("🏆 Model Tournament Results")
        lb_df = pd.DataFrame(leaderboard)
        def highlight_winner(row):
            return ['background-color: #10b981; color: white' if row['Selected'] == '🏆 WINNER' else '' for _ in row]
        st.dataframe(lb_df.style.apply(highlight_winner, axis=1), use_container_width=True, hide_index=True)
    
    st.subheader("Feature Importance Analysis")
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
                st.info("Visuals not available for this model type (Black Box).")
        except:
            st.info("Feature importance data unavailable.")

    with col_risk:
        st.markdown("##### Risk vs. Customer Tenure")
        fig_scatter = px.scatter(df, x="TENURE_MONTHS", y="CHURN_PROBABILITY", color="PREDICTED_CHURN", 
                               opacity=0.6, title="Does Loyalty Reduce Risk?")
        st.plotly_chart(fig_scatter, use_container_width=True)
