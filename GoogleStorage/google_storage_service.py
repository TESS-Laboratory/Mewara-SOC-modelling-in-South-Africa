import os
from google.cloud import storage
import tensorflow as tf

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'GoogleStorage\Credentials\key.json'

def initialize_gcs_client():
    client = storage.Client()
    return client

def download_model(model_output_path, local_model_path=None):
    client = initialize_gcs_client()
    bucket_name = 'socsabucket'
    
    bucket = client.bucket(bucket_name)
    
    blob = bucket.blob(model_output_path)

    if local_model_path is None:
        local_model_path = os.path.basename(model_output_path)
    
    try:
        blob.download_to_filename(local_model_path)
        print(f"Model downloaded to {local_model_path}")
        return local_model_path
    except Exception as e:
        print(f"Error downloading model: {e}")
        return None
    
def load_model_cloud(model_path):
    local_model_path = download_model(model_path)
    if local_model_path:
        # Load the model using TensorFlow
        try:
            model = tf.keras.models.load_model(local_model_path)
            print("Model loaded successfully.")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print("Failed to download the model.")
