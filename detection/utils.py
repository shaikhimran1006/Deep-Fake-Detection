import os
import gdown

# Google Drive file ID
FILE_ID = '1iUgxZ1mmk8NrWxb3DS9rTH7P-Vbw-IxF'  # Replace with your actual file ID
MODEL_PATH = os.path.join(os.getcwd(), 'model.keras')  # Save in current working directory

def download_model():
    """Download the model from Google Drive if it doesn't exist."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading the model from Google Drive...")
        url = f'https://drive.google.com/uc?id={FILE_ID}'
        gdown.download(url, MODEL_PATH, quiet=False)
    else:
        print("Model already exists locally.")
