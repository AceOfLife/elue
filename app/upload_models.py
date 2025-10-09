from dotenv import load_dotenv
from huggingface_hub import HfApi, HfFolder, upload_file, hf_hub_download
import os
import joblib

# === Load Environment Variables ===
load_dotenv()  # Reads the .env file in the project root

HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")  # Read from env securely
REPO_ID = "BigDavies/2staged-HalvingRandomSearchCV"
MODEL_DIR = "models"
MODEL_FILES = ["rf_pipeline.pkl", "rf_pipeline2.pkl"]

# === Safety Check ===
if not HF_TOKEN:
    raise ValueError("HUGGINGFACE_HUB_TOKEN environment variable not set.")

# === Authenticate with Hugging Face Hub ===
HfFolder.save_token(HF_TOKEN)
api = HfApi()

# === Create Repo (if it doesn't already exist) ===
api.create_repo(repo_id=REPO_ID, exist_ok=True, token=HF_TOKEN)
print(f"Repository ready: {REPO_ID}")

# === Upload Model Files ===
for model_file in MODEL_FILES:
    model_path = os.path.join(MODEL_DIR, model_file)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_file,
        repo_id=REPO_ID,
        token=HF_TOKEN,
    )
    print(f"Uploaded: {model_file}")

# === Verify Upload by Downloading & Loading ===
loaded_models = {}

for model_file in MODEL_FILES:
    downloaded_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=model_file,
        token=HF_TOKEN
    )
    with open(downloaded_path, "rb") as f:
        loaded_models[model_file] = joblib.load(f)
    print(f"Loaded model: {model_file}")

# === Access the Loaded Models if Needed ===
stage1_model = loaded_models["rf_pipeline.pkl"]
stage2_model = loaded_models["rf_pipeline2.pkl"]

print(" All models uploaded and verified successfully!")
