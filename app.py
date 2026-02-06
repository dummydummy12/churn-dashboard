import streamlit as st
import pandas as pd
import boto3
import joblib
import plotly.express as px
from io import BytesIO
from snowflake.connector import connect
import os

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Churn Prediction Engine", page_icon="⚡", layout="wide")

st.markdown("""
<style>
    .metric-card {background-color: #f0f2f6; padding: 20px; border-radius: 10px;}
    .stButton>button {width: 100%;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# CONFIGURATION
# ==========================================
def get_config(key):
    """Gets config from Environment or Streamlit Secrets"""
    # Check standard environment variable first (for AWS/Docker)
    if key in os.environ:
        return os.environ[key]
    # Check Streamlit secrets (for Community Cloud)
    elif key in st.secrets:
        return st.secrets[key]
    return None

try:
    # Snowflake Credentials
    SNOWFLAKE_USER = get_config("SNOWFLAKE_USER")
    SNOWFLAKE_PASSWORD = get_config("SNOWFLAKE_PASSWORD")
    SNOWFLAKE_ACCOUNT = get_config("SNOWFLAKE_ACCOUNT")
    # Hardcoded values for non-sensitive config
    SNOWFLAKE_WH = "CHURN_PROJECT_WH_M"
    SNOWFLAKE_DB = "RETAIL_ANALYTICS"
    SNOWFLAKE_SCHEMA = "CHURN_PREDICTION"
    
    # AWS Credentials
    BUCKET_NAME = get_config("BUCKET_NAME")
    MODEL_KEY = "models/best_churn_model.pkl"
    AWS_REGION = get_config("AWS_REGION") or "us-east-1"
    
    # AWS Keys (Required for Streamlit Cloud)
    AWS_ACCESS_KEY = get_config("AWS_ACCESS_KEY_ID")
    AWS_SECRET_KEY = get_config("AWS_SECRET_ACCESS_KEY")

except Exception:
    st.error("Configuration Error. Please check Streamlit Secrets.")
    st.stop()

# ==========================================
# DATA LOADING
# ==========================================
@st.cache_resource(ttl=60)
def load_model_from_s3():
    # Streamlit Cloud needs explicit keys
    if AWS_ACCESS_KEY and AWS_SECRET_KEY:
        s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY, region_name=AWS_REGION)
    else:
        # Fallback for local/IAM roles
        s3 = boto3.client('s3', region_name=AWS_REGION)

    try:
        with st.spinner('Fetching Champion Model from S3...'):
            response = s3.get_object(Bucket=BUCKET_NAME, Key=MODEL_KEY)
            model_bytes = response['Body'].read()
            model = joblib.load(BytesIO(model_bytes))
        return model
    except Exception as e:
        st.warning(f"Waiting for Model... (Run Training Lambda). Error: {e}")
        return None

def get_snowflake_data():
    conn = connect(
        user=SNOWFLAKE_USER,
        password=SNOWFLAKE_PASSWORD,
        account=SNOWFLAKE_ACCOUNT,
        warehouse=SNOWFLAKE_WH,
        database=SNOWFLAKE_DB,
        schema=SNOWFLAKE_SCHEMA
    )
    # Get recent data sample
    query = "SELECT * FROM CUSTOMER_FEATURES_VIEW SAMPLE(2000 ROWS)" 
    df = pd.read_sql(query, conn)
    conn.close()
    
    # Standardize column names
    df.columns = [x.upper() for x in df.columns]
    return df.fillna(0)

# ==========================================
# MAIN UI
# ==========================================
st.title("⚡ Real-Time Churn Analytics Engine")
st.markdown("Monitoring **Behavioral Signals** merged with **Historical Profiles**.")

col1, col2 = st.columns([1, 3])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/4149/4149663.png", width=100)
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()

model = load_model_from_s3()
df = get_snowflake_data()

if model is not None and not df.empty:
    X = df.drop(columns=['CUSTOMERID', 'CHURN_LABEL'], errors='ignore')
    
    try:
        # Predict using Pipeline (Scaler + Model)
        predictions = model.predict(X)
        probs = model.predict_proba(X)[:, 1]
        
        df['PREDICTED_CHURN'] = predictions
        df['CHURN_PROBABILITY'] = probs

        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Customers Analyzed", len(df))
        m2.metric("Avg Churn Risk", f"{df['CHURN_PROBABILITY'].mean():.1%}")
        m3.metric("High Risk (>70%)", df[df['CHURN_PROBABILITY'] > 0.7].shape[0], delta_color="inverse")
        rt_active = df[df['RT_TOTAL_INTERACTIONS'] > 0].shape[0]
        m4.metric("Real-Time Active", rt_active, delta_color="normal")

        # VIP Watchlist
        st.markdown("---")
        st.header("💎 VIP Watchlist: Anomaly Detection")
        
        vip_mask = (df['TENURE_MONTHS'] > 24) & (df['MONTHLY_CHARGES'] > 70)
        vips = df[vip_mask].copy()
        risky_vips = vips[vips['CHURN_PROBABILITY'] > 0.5]
        
        if not risky_vips.empty:
            st.error(f"🚨 ALERT: {len(risky_vips)} VIPs are at risk of churning!")
            cols = ['CUSTOMERID', 'CHURN_PROBABILITY', 'RT_CANCELLATION_INTENT', 'RT_CHECKOUT_ERRORS', 'TENURE_MONTHS', 'MONTHLY_CHARGES']
            valid_cols = [c for c in cols if c in risky_vips.columns]
            st.dataframe(risky_vips[valid_cols].sort_values(by='CHURN_PROBABILITY', ascending=False).style.background_gradient(subset=['CHURN_PROBABILITY'], cmap='Reds'))
        else:
            st.success("No VIPs currently exhibiting churn behavior.")

        # Visuals
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Risk vs. Tenure")
            fig = px.scatter(df, x="TENURE_MONTHS", y="CHURN_PROBABILITY", color="PREDICTED_CHURN", opacity=0.6)
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            st.subheader("Feature Importance")
            estimator = model.named_steps['clf']
            if hasattr(estimator, 'feature_importances_'):
                feat_imp = pd.DataFrame({'Feature': X.columns, 'Importance': estimator.feature_importances_}).sort_values(by='Importance', ascending=False).head(10)
                fig2 = px.bar(feat_imp, x='Importance', y='Feature', orientation='h')
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Visuals not available for this model type.")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
