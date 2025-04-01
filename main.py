import joblib
import threading
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


# Initialize FastAPI app
app = FastAPI()
print("FastAPI app initialized")

try:
    # Load the saved model, encoder, and scaler
    print("Loading model files...")
    model = joblib.load("nfraud_detection_model.pkl")
    print(f"Model loaded successfully: {type(model)}")
    
    woe_encoder = joblib.load("nwoe_encoder.pkl")
    print(f"WOE encoder loaded successfully: {type(woe_encoder)}")
    
    scaler = joblib.load("nscaler.pkl")
    print(f"Scaler loaded successfully: {type(scaler)}")
except Exception as e:
    print(f"Error loading model files: {str(e)}")
    raise

class Transaction(BaseModel):
    cc_num_frequency: str
    amt: float
    merchant: str
    category: str
    trans_date_trans_time: str
    dob: str
    merch_lat: float
    merch_long: float
    city: str
    job: str
    is_fraud: str

class TransactionList(BaseModel):
    transactions: List[Transaction]

def preprocess_data(data):
    print(f"Starting preprocessing of data with shape: {data.shape}")
    print(f"Input data columns: {data.columns.tolist()}")
    print(f"First row of input data: {data.iloc[0].to_dict()}")
    
    # Convert date columns to datetime and ensure consistent timezone handling
    print("Converting date columns to datetime...")
    data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])
    data['dob'] = pd.to_datetime(data['dob'])
    
    # Make both datetime columns timezone-aware (UTC) or both timezone-naive
    # Option 1: Make both timezone-naive
    data['trans_date_trans_time'] = data['trans_date_trans_time'].dt.tz_localize(None)
    
    # Alternative Option 2: Make both timezone-aware
    # data['dob'] = data['dob'].dt.tz_localize('UTC')
    
    print(f"Date conversion complete. Sample trans_date: {data['trans_date_trans_time'].iloc[0]}")
    
    # Feature Engineering
    print("Performing feature engineering...")
    data['age'] = (data['trans_date_trans_time'] - data['dob']).dt.days // 365
    print(f"Age feature created. Sample age: {data['age'].iloc[0]}")
    
    # Rest of your preprocessing code remains the same
    data['hour'] = data['trans_date_trans_time'].dt.hour
    print(f"Hour feature created. Sample hour: {data['hour'].iloc[0]}")
    
    data['amt_log'] = np.log1p(data['amt'])
    print(f"Log amount feature created. Original amt: {data['amt'].iloc[0]}, Log amt: {data['amt_log'].iloc[0]}")
    
    # Drop all columns except the ones you want
    required_columns = ['merch_lat', 'age', 'hour', 'amt_log', 'category', 'city', 'job', 'cc_num_frequency', "is_fraud"]
    print(f"Keeping only required columns: {required_columns}")
    data = data[required_columns]
    print(f"Preprocessed data shape: {data.shape}")
    print(f"Preprocessed data columns: {data.columns.tolist()}")
    print(f"First row after preprocessing: {data.iloc[0].to_dict()}")
    
    return data



@app.post("/predict")
async def predict_fraud(transaction_list: TransactionList):
    try:
        print(f"Received prediction request with {len(transaction_list.transactions)} transactions")
        
        # Convert input data to DataFrame
        print("Converting input data to DataFrame...")
        df = pd.DataFrame([t.dict() for t in transaction_list.transactions])
        print(f"Input DataFrame created with shape: {df.shape}")
        
        # Preprocess the data
        print("Starting data preprocessing...")
        df = preprocess_data(df)
        print("Data preprocessing completed")
        
        # Apply WOE encoding
        print("Applying WOE encoding...")
        df_encoded = woe_encoder.transform(df)
        print(f"WOE encoding applied. Encoded data shape: {df_encoded.shape}")
        print(f"Encoded data columns: {df_encoded.columns.tolist()}")

        df_encoded = df_encoded.rename(columns={
            'category': 'category_WOE',
            'city': 'city_WOE',
            'job': 'job_WOE'
        })
        
        # Prepare features for prediction
        features = ['merch_lat', 'age', 'hour', 'amt_log', 'category_WOE', 'city_WOE', 'job_WOE', 'cc_num_frequency']
        print(f"Using features for prediction: {features}")
        X = df_encoded[features]
        print(f"Feature matrix shape: {X.shape}")
        
        # Scale the features
        print("Scaling features...")
        X_scaled = scaler.transform(X)
        print(f"Features scaled. Scaled data shape: {X_scaled.shape}")
        print(f"Sample of scaled data (first row): {X_scaled[0]}")
        
        # Make predictions
        print("Making predictions...")
        predictions = model.predict(X_scaled)
        print(f"Predictions made: {predictions}")
        
        probabilities = model.predict_proba(X_scaled)[:, 1]
        print(f"Prediction probabilities: {probabilities}")
        
        # Prepare the response
        print("Preparing response...")
        results = []
        for i, transaction in enumerate(transaction_list.transactions):
            result = {
                "transaction": transaction.dict(),
                "fraud_prediction": bool(predictions[i]),
                "fraud_probability": float(probabilities[i])
            }
            results.append(result)
            print(f"Transaction {i+1} result: fraud={bool(predictions[i])}, probability={float(probabilities[i]):.4f}")
        
        print(f"Returning results for {len(results)} transactions")
        return {"results": results}
    
    except Exception as e:
        error_msg = f"Error in prediction: {str(e)}"
        print(error_msg)
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)
