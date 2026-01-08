
import boto3
from botocore.client import Config
import os

# Minio config
MINIO_ENDPOINT = "http://localhost:9001"
ACCESS_KEY = "minio"
SECRET_KEY = "minio123"
BUCKET_NAME = "models"
MODEL_FILE = "models/xgboost_baseline.json"
OBJECT_NAME = "credit-risk/model.json"

def upload_model():
    print(f"Connecting to Minio at {MINIO_ENDPOINT}...")
    s3 = boto3.client('s3',
                      endpoint_url=MINIO_ENDPOINT,
                      aws_access_key_id=ACCESS_KEY,
                      aws_secret_access_key=SECRET_KEY,
                      config=Config(signature_version='s3v4'))

    # Create bucket if not exists
    try:
        s3.create_bucket(Bucket=BUCKET_NAME)
        print(f"Created bucket '{BUCKET_NAME}'")
    except Exception as e:
        if "BucketAlreadyOwnedByYou" in str(e) or "BucketAlreadyExists" in str(e):
            print(f"Bucket '{BUCKET_NAME}' already exists")
        else:
            print(f"Error creating bucket: {e}")

    # Upload file
    print(f"Uploading {MODEL_FILE} to {BUCKET_NAME}/{OBJECT_NAME}...")
    try:
        s3.upload_file(MODEL_FILE, BUCKET_NAME, OBJECT_NAME)
        print("✅ Upload successful!")
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        raise e

if __name__ == "__main__":
    upload_model()
