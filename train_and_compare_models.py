import os
import json
import boto3
import joblib
import pandas as pd
from io import BytesIO
from snowflake.connector import connect
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ==========================================
# CONFIGURATION
# ==========================================
SNOWFLAKE_USER = os.environ.get('SNOWFLAKE_USER', '').strip()
SNOWFLAKE_PASSWORD = os.environ.get('SNOWFLAKE_PASSWORD', '').strip()
SNOWFLAKE_ACCOUNT = os.environ.get('SNOWFLAKE_ACCOUNT', '').strip()
SNOWFLAKE_WH = os.environ.get('SNOWFLAKE_WH', 'CHURN_PROJECT_WH_M').strip()
SNOWFLAKE_DB = os.environ.get('SNOWFLAKE_DB', 'RETAIL_ANALYTICS').strip()
SNOWFLAKE_SCHEMA = os.environ.get('SNOWFLAKE_SCHEMA', 'CHURN_PREDICTION').strip()
BUCKET_NAME = os.environ.get('BUCKET_NAME', '').strip()
MODEL_KEY = 'models/best_churn_model.pkl'
LEADERBOARD_KEY = 'models/leaderboard.json'

def get_data():
    """Fetches the Customer 360 view from Snowflake"""
    print("🔌 Connecting to Snowflake...")
    ctx = connect(
        user=SNOWFLAKE_USER,
        password=SNOWFLAKE_PASSWORD,
        account=SNOWFLAKE_ACCOUNT,
        warehouse=SNOWFLAKE_WH,
        database=SNOWFLAKE_DB,
        schema=SNOWFLAKE_SCHEMA
    )
    
    print(f"📥 Querying CUSTOMER_FEATURES_VIEW using {SNOWFLAKE_WH}...")
    df = pd.read_sql("SELECT * FROM CUSTOMER_FEATURES_VIEW", ctx)
    ctx.close()
    
    # Uppercase columns are standard in Snowflake
    df.columns = [x.upper() for x in df.columns]
    return df

def lambda_handler(event, context):
    print("🚀 Starting Multi-Model Tournament (Updated)...")
    
    # 1. Load Data
    try:
        df = get_data()
        print(f"✅ Loaded {len(df)} rows.")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return {'statusCode': 500, 'body': str(e)}

    # 2. Preprocessing
    X = df.drop(columns=['CUSTOMERID', 'CHURN_LABEL'], errors='ignore')
    y = df['CHURN_LABEL']
    
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

    # 3. Define Contenders
    def make_pipeline(classifier):
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf', classifier)
        ])

    models = {
        "Logistic Regression": make_pipeline(LogisticRegression(max_iter=1000)),
        "Random Forest": make_pipeline(RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)),
        "Gradient Boosting": make_pipeline(GradientBoostingClassifier(n_estimators=100, random_state=42)),
        "AdaBoost": make_pipeline(AdaBoostClassifier(n_estimators=50, random_state=42)),
        "Support Vector Machine": make_pipeline(SVC(probability=True, random_state=42)) 
    }

    best_name = None
    best_score = 0.0 
    best_model = None
    leaderboard = []

    # 4. The Tournament Loop
    print("🥊 Fighting...")
    for name, pipeline in models.items():
        try:
            print(f"   Training {name}...")
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)
            
            # Metrics
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds)
            
            # Add to leaderboard list
            leaderboard.append({
                "Model": name,
                "Accuracy": f"{acc:.2%}",
                "F1_Score": f"{f1:.2%}",
                "Selected": False
            })
            
            print(f"   👉 {name} | Acc: {acc:.2f} | F1: {f1:.2f}")
            
            if f1 > best_score:
                best_score = f1
                best_name = name
                best_model = pipeline
                
        except Exception as e:
            print(f"   ⚠️ Error training {name}: {e}")

    # Mark the winner in the leaderboard
    for entry in leaderboard:
        if entry["Model"] == best_name:
            entry["Selected"] = "🏆 WINNER"
        else:
            entry["Selected"] = ""

    print(f"🏆 WINNER: {best_name} with F1-Score: {best_score:.2%}")

    # 5. Save Leaderboard JSON & Model to S3
    if best_model:
        s3 = boto3.client('s3')
        
        # Save Model
        buffer = BytesIO()
        joblib.dump(best_model, buffer)
        buffer.seek(0)
        s3.put_object(Bucket=BUCKET_NAME, Key=MODEL_KEY, Body=buffer)
        
        # Save Leaderboard
        s3.put_object(
            Bucket=BUCKET_NAME, 
            Key=LEADERBOARD_KEY, 
            Body=json.dumps(leaderboard),
            ContentType='application/json'
        )
        
        print(f"💾 Saved model and leaderboard to s3://{BUCKET_NAME}/")

        return {
            'statusCode': 200,
            'body': json.dumps({
                "message": "Tournament Complete",
                "winner": best_name,
                "leaderboard": leaderboard
            })
        }
    else:
        return {'statusCode': 500, 'body': "No models trained successfully."}
