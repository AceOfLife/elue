# from huggingface_hub import hf_hub_download
# import pandas as pd
# import joblib
# import os
# from typing import Tuple

# # Constants
# HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
# REPO_ID = "BigDavies/2staged-HalvingRandomSearchCV"
# MODEL_FILES = ["rf_pipeline.pkl", "rf_pipeline2.pkl"]

# def load_models() -> Tuple:
#     """
#     Load trained model pipelines from the Hugging Face Hub.
    
#     Returns:
#         Tuple: (stage1_pipeline, stage2_pipeline)
#     """
#     loaded_models = {}
    
#     for model_file in MODEL_FILES:
#         try:
#             path = hf_hub_download(
#                 repo_id=REPO_ID,
#                 filename=model_file,
#                 token=HF_TOKEN
#             )
#             with open(path, "rb") as f:
#                 loaded_models[model_file] = joblib.load(f)
#         except Exception as e:
#             raise RuntimeError(f"Failed to load {model_file} from Hugging Face Hub: {e}")
    
#     return loaded_models["rf_pipeline.pkl"], loaded_models["rf_pipeline2.pkl"]


# def make_prediction(stage1_pipeline, stage2_pipeline, form_data: dict) -> Tuple[float, float]:
#     """
#     Generate TRP and GRP predictions based on form input.

#     Args:
#         stage1_pipeline: Trained pipeline for stage 1 (Spend → TRP)
#         stage2_pipeline: Trained pipeline for stage 2 (TRP + context → GRP)
#         form_data (dict): Form input data from user

#     Returns:
#         Tuple[float, float]: (Predicted TRP, Predicted GRP)
#     """
#     # Prepare input for Stage 1
#     stage1_df = pd.DataFrame([{
#         "Category": form_data["Category"],
#         "Month": form_data["Month"],
#         "Daypart": form_data["Daypart"],
#         "Spend": float(form_data["Spend"]),
#         "Station": form_data["Station"]
#     }])

#     predicted_trp = stage1_pipeline.predict(stage1_df)[0]

#     # Prepare input for Stage 2
#     stage2_df = pd.DataFrame([{
#         "Predicted_TRP": predicted_trp,
#         "Normalize 30s": float(form_data["Normalize_30s"]),
#         "Duration": float(form_data["Duration"])
#     }])

#     predicted_grp = stage2_pipeline.predict(stage2_df)[0]

#     return predicted_trp, predicted_grp


from huggingface_hub import hf_hub_download
import pandas as pd
import joblib
import os
from typing import Tuple

# Constants
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
REPO_ID = "BigDavies/2staged-HalvingRandomSearchCV"
MODEL_FILES = ["rf_pipeline.pkl", "rf_pipeline2.pkl"]

def load_models() -> Tuple:
    """
    Load trained model pipelines from the Hugging Face Hub.
    
    Returns:
        Tuple: (stage1_pipeline, stage2_pipeline)
    """
    loaded_models = {}
    
    for model_file in MODEL_FILES:
        try:
            path = hf_hub_download(
                repo_id=REPO_ID,
                filename=model_file,
                token=HF_TOKEN
            )
            with open(path, "rb") as f:
                loaded_models[model_file] = joblib.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load {model_file} from Hugging Face Hub: {e}")
    
    return loaded_models["rf_pipeline.pkl"], loaded_models["rf_pipeline2.pkl"]


def make_prediction(stage1_pipeline, stage2_pipeline, form_data: dict) -> Tuple[float, float]:
    """
    Generate TRP and GRP predictions based on form input.

    Args:
        stage1_pipeline: Trained pipeline for stage 1 (Spend → TRP)
        stage2_pipeline: Trained pipeline for stage 2 (TRP + context → GRP)
        form_data (dict): Form input data from user

    Returns:
        Tuple[float, float]: (Predicted TRP, Predicted GRP)
    """
    # Prepare input for Stage 1
    stage1_df = pd.DataFrame([{
        "Category": form_data["Category"],
        "Month": form_data["Month"],
        "Daypart": form_data["Daypart"],
        "Spend": float(form_data["Spend"]),
        "Station": form_data["Station"],
        "Audience": form_data["Audience"],
    }])

    predicted_trp = stage1_pipeline.predict(stage1_df)[0]

    # Prepare input for Stage 2
    stage2_df = pd.DataFrame([{
        "Predicted_TRP": predicted_trp,
        "Normalize 30s": float(form_data["Normalize_30s"]),
        "Duration": float(form_data["Duration"])
    }])

    predicted_grp = stage2_pipeline.predict(stage2_df)[0]

    return predicted_trp, predicted_grp