import os
from google.cloud import storage
import tensorflow as tf

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'GoogleStorage\Credentials\key.json'

def initialize_gcs_client():
    client = storage.Client()
    return client

def download_model(model_output_path):
    client = initialize_gcs_client()
    bucket_name = 'socsabucket'
    
    bucket = client.bucket(bucket_name)
    
    blob = bucket.blob(model_output_path)

    local_model_fileName = os.path.basename(model_output_path)

    try:
        blob.download_to_filename(local_model_fileName)
        print(f"Model downloaded to {local_model_fileName}")
        return local_model_fileName
    except Exception as e:
        print(f"Error downloading model: {e}")
        return None
 