import json
import boto3
import time
import random
import uuid
import os
import csv
from datetime import datetime

# ==========================================
# CONFIGURATION
# ==========================================
# Ensure these match your environment variables
BUCKET_NAME = os.environ.get('BUCKET_NAME') 
REGION = os.environ.get('AWS_REGION', 'us-east-1')

# Path to the CSV you uploaded in Phase 1
# Make sure this path matches exactly where you put the file in S3
CSV_FILE_KEY = 'csv_data/WA_Fn-UseC_-Telco-Customer-Churn.csv' 

s3 = boto3.client('s3', region_name=REGION)

# Global variable to cache IDs (so we don't read S3 every single second)
CACHED_CUSTOMER_IDS = []

def get_all_customer_ids():
    """
    Downloads the full CSV from S3 once and extracts all 7043 IDs.
    """
    global CACHED_CUSTOMER_IDS
    
    # If we already have them in memory, return them
    if CACHED_CUSTOMER_IDS:
        return CACHED_CUSTOMER_IDS
        
    print(f"📥 Fetching Customer List from s3://{BUCKET_NAME}/{CSV_FILE_KEY} ...")
    
    try:
        response = s3.get_object(Bucket=BUCKET_NAME, Key=CSV_FILE_KEY)
        # Read the content as a string
        lines = response['Body'].read().decode('utf-8').splitlines()
        
        # Parse CSV to get the first column (customerID)
        reader = csv.reader(lines)
        header = next(reader) # Skip header
        
        ids = [row[0] for row in reader if row] # Extract first column
        
        print(f"✅ Loaded {len(ids)} unique Customer IDs.")
        CACHED_CUSTOMER_IDS = ids
        return ids
        
    except Exception as e:
        print(f"❌ Error loading CSV from S3: {e}")
        # Fallback to a few IDs if S3 read fails, just to keep running
        return ['7590-VHVEG', '5575-GNVDE']

# Simulated Data Lists
EVENT_TYPES = ['page_view', 'add_to_cart', 'checkout_error', 'view_cancellation_policy', 'contact_support']
PAGES = ['/home', '/dashboard', '/billing', '/support', '/cancel-confirm']

def generate_event(all_ids):
    """Generates a single event for a RANDOM real customer"""
    event = {
        "event_id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        
        # PICK FROM THE FULL LIST OF 7000+ IDs
        "customer_id": random.choice(all_ids), 
        
        "event_type": random.choice(EVENT_TYPES),
        "url_visited": random.choice(PAGES),
        "session_duration": random.randint(10, 600),
        "device": random.choice(['mobile', 'desktop', 'tablet'])
    }
    return event

def lambda_handler(event, context):
    """
    AWS Lambda Entry Point.
    """
    print("🚀 Starting Cloud Event Simulation...")
    
    # 1. Get the full list of IDs
    all_ids = get_all_customer_ids()
    
    # 2. Generate Events
    num_events = random.randint(5, 15) # Generate a batch
    uploaded_files = []

    for _ in range(num_events):
        data = generate_event(all_ids)
        file_name = f"events/event_{int(time.time())}_{str(uuid.uuid4())[:8]}.json"
        
        try:
            json_string = json.dumps(data)
            s3.put_object(
                Bucket=BUCKET_NAME,
                Key=file_name,
                Body=json_string
            )
            print(f"✅ Uploaded {file_name} for {data['customer_id']}")
            uploaded_files.append(file_name)
        except Exception as e:
            print(f"❌ Error uploading to S3: {e}")

    return {
        'statusCode': 200,
        'body': json.dumps(f"Generated {len(uploaded_files)} events across {len(all_ids)} potential customers.")
    }
