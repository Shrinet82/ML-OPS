
import boto3
from botocore.client import Config

MINIO_ENDPOINT = "http://localhost:9001"
ACCESS_KEY = "minio"
SECRET_KEY = "minio123"
BUCKET_NAME = "models"
OBJECT_TO_DELETE = "credit-risk/xgboost_baseline.json"

def delete_file():
    print(f"Connecting to Minio at {MINIO_ENDPOINT}...")
    s3 = boto3.client('s3',
                      endpoint_url=MINIO_ENDPOINT,
                      aws_access_key_id=ACCESS_KEY,
                      aws_secret_access_key=SECRET_KEY,
                      config=Config(signature_version='s3v4'))

    print(f"Deleting {OBJECT_TO_DELETE}...")
    try:
        s3.delete_object(Bucket=BUCKET_NAME, Key=OBJECT_TO_DELETE)
        print("✅ Deletion successful!")
    except Exception as e:
        print(f"❌ Deletion failed: {e}")

if __name__ == "__main__":
    delete_file()
