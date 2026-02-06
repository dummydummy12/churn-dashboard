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
    """Gets config from Environment (Cloud) or Secrets (Local)"""
    return os.environ.get(key) or st.secrets.get(key)

try:
    # Snowflake Credentials
    SNOWFLAKE_USER = get_config("SNOWFLAKE_USER")
    SNOWFLAKE_PASSWORD = get_config("SNOWFLAKE_PASSWORD")
    SNOWFLAKE_ACCOUNT = get_config("SNOWFLAKE_ACCOUNT")
    SNOWFLAKE_WH = "CHURN_PROJECT_WH_M"
    SNOWFLAKE_DB = "RETAIL_ANALYTICS"
    SNOWFLAKE_SCHEMA = "CHURN_PREDICTION"
    
    # AWS Credentials
    BUCKET_NAME = get_config("BUCKET_NAME")
    MODEL_KEY = "models/best_churn_model.pkl"
    LEADERBOARD_KEY = "models/leaderboard.json" # New file key
    AWS_REGION = get_config("AWS_REGION") or "us-east-1"
    
    # AWS Keys (Required for Streamlit Cloud)
    AWS_ACCESS_KEY = get_config("AWS_ACCESS_KEY_ID")
    AWS_SECRET_KEY = get_config("AWS_SECRET_ACCESS_KEY")

except Exception:
    st.error("Configuration Error. Please check Environment Variables or Secrets.")
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
        # 1. Load Model
        with st.spinner('Fetching Champion Model...'):
            response = s3.get_object(Bucket=BUCKET_NAME, Key=MODEL_KEY)
            model_bytes = response['Body'].read()
            resources["model"] = joblib.load(BytesIO(model_bytes))
            
        # 2. Load Leaderboard (JSON)
        try:
            response_lb = s3.get_object(Bucket=BUCKET_NAME, Key=LEADERBOARD_KEY)
            content = response_lb['Body'].read().decode('utf-8')
            resources["leaderboard"] = json.loads(content)
        except Exception as e:
            # It's okay if leaderboard is missing initially
            print(f"Leaderboard not found: {e}")
            
    except Exception as e:
        st.warning(f"Waiting for Model... (Run Training Lambda). Error: {e}")
        
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

col1, col2 = st.columns([1, 3])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/4149/4149663.png", width=100)
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()

# Load everything
resources = load_resources_from_s3()
model = resources["model"]
leaderboard = resources["leaderboard"]
df = get_snowflake_data()

# --- SECTION 1: MODEL TOURNAMENT RESULTS ---
if leaderboard:
    st.subheader("🏆 Model Tournament Leaderboard")
    lb_df = pd.DataFrame(leaderboard)
    
    # Highlight the winner row
    def highlight_winner(row):
        return ['background-color: #10b981; color: white; font-weight: bold' if row['Selected'] == '🏆 WINNER' else '' for _ in row]

    st.dataframe(
        lb_df.style.apply(highlight_winner, axis=1),
        use_container_width=True,
        hide_index=True
    )
elif model is not None:
    st.info("Leaderboard data not yet generated. Run the training Lambda again to see all model scores.")

# --- SECTION 2: PREDICTIONS ---
if model is not None and not df.empty:
    X = df.drop(columns=['CUSTOMERID', 'CHURN_LABEL'], errors='ignore')
    
    try:
        # Predict using Pipeline
        predictions = model.predict(X)
        probs = model.predict_proba(X)[:, 1]
        
        df['PREDICTED_CHURN'] = predictions
        df['CHURN_PROBABILITY'] = probs

        # Metrics
        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Analyzed", len(df))
        m2.metric("Avg Risk", f"{df['CHURN_PROBABILITY'].mean():.1%}")
        m3.metric("High Risk (>70%)", df[df['CHURN_PROBABILITY'] > 0.7].shape[0], delta_color="inverse")
        
        # Check RT columns safely
        rt_col = 'RT_TOTAL_INTERACTIONS' if 'RT_TOTAL_INTERACTIONS' in df.columns else df.columns[0]
        rt_active = df[df[rt_col] > 0].shape[0] if 'RT_TOTAL_INTERACTIONS' in df.columns else 0
        m4.metric("Real-Time Active", rt_active, delta_color="normal")

        # --- SECTION 3: VIP WATCHLIST ---
        st.markdown("---")
        st.header("💎 VIP Watchlist")
        
        # High Value: High Tenure & High Spend
        vip_mask = (df['TENURE_MONTHS'] > 24) & (df['MONTHLY_CHARGES'] > 70)
        vips = df[vip_mask].copy()
        risky_vips = vips[vips['CHURN_PROBABILITY'] > 0.5]
        
        if not risky_vips.empty:
            st.error(f"🚨 ALERT: {len(risky_vips)} VIPs are at risk!")
            # Display safely
            display_cols = ['CUSTOMERID', 'CHURN_PROBABILITY', 'RT_CANCELLATION_INTENT', 'RT_CHECKOUT_ERRORS', 'TENURE_MONTHS', 'MONTHLY_CHARGES']
            valid_cols = [c for c in display_cols if c in risky_vips.columns]
            
            st.dataframe(
                risky_vips[valid_cols].sort_values(by='CHURN_PROBABILITY', ascending=False)
                .style.background_gradient(subset=['CHURN_PROBABILITY'], cmap='Reds')
                .format({'CHURN_PROBABILITY': "{:.1%}", 'MONTHLY_CHARGES': "${:.2f}"})
            )
        else:
            st.success("No VIPs currently exhibiting churn behavior.")

        # --- SECTION 4: VISUALS ---
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Risk vs. Tenure")
            fig = px.scatter(df, x="TENURE_MONTHS", y="CHURN_PROBABILITY", color="PREDICTED_CHURN", opacity=0.6)
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            st.subheader("Feature Importance")
            # Logic to extract importance from Pipeline safely
            try:
                estimator = model.named_steps['clf']
                if hasattr(estimator, 'feature_importances_'):
                    feat_imp = pd.DataFrame({'Feature': X.columns, 'Importance': estimator.feature_importances_}).sort_values(by='Importance', ascending=False).head(10)
                    fig2 = px.bar(feat_imp, x='Importance', y='Feature', orientation='h')
                    st.plotly_chart(fig2, use_container_width=True)
                elif hasattr(estimator, 'coef_'):
                     # For Linear Models
                     coeffs = estimator.coef_[0]
                     feat_imp = pd.DataFrame({'Feature': X.columns, 'Importance': abs(coeffs)}).sort_values(by='Importance', ascending=False).head(10)
                     fig2 = px.bar(feat_imp, x='Importance', y='Feature', orientation='h', title="Feature Impact (Magnitude)")
                     st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info("Current model (Black Box) does not provide simple feature importance.")
            except:
                st.info("Could not extract feature importance from pipeline.")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
