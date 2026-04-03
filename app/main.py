# from fastapi import FastAPI, Form, Request, UploadFile, File
# from fastapi.responses import HTMLResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates
# from pathlib import Path
# import pandas as pd
# import numpy as np
# import joblib
# from typing import List

# app = FastAPI()
# BASE_DIR = Path(__file__).resolve().parent

# app.mount(
#     "/static",
#     StaticFiles(directory=BASE_DIR.parent / "static"),
#     name="static",
# )
# templates = Jinja2Templates(directory=BASE_DIR.parent / "templates")

# # Load pre-trained models
# stage1_pipeline = joblib.load(BASE_DIR.parent / "models" / "rf_pipeline.pkl")
# stage2_pipeline = joblib.load(BASE_DIR.parent / "models" / "rf_pipeline2.pkl")

# # GET /
# @app.get("/", response_class=HTMLResponse)
# async def read_form(request: Request):
#     stations = [
#         "am yoruba dstv", "slashfm ibadan", "frsc fm abuja",
#         "mtv base", "nta sport 24", "zeeworld dstv"
#     ]
#     return templates.TemplateResponse("index.html", {"request": request, "stations": stations})

# # POST /predict — Manual form submission
# @app.post("/predict/", response_class=HTMLResponse)
# async def predict(
#     request: Request,
#     Category: str = Form(...),
#     Month_name: str = Form(...),
#     Daypart: str = Form(...),
#     total_spend: float = Form(..., alias="Spend"),
#     Normalize_30s: float = Form(...),
#     Duration: float = Form(...),
#     stations: List[str] = Form(...)
# ):
#     df = pd.DataFrame({
#         "Category": Category,
#         "Month_name": Month_name,
#         "Daypart": Daypart,
#         "Station": stations
#     })

#     n = len(stations)
#     df["Station_Spend"] = total_spend / n
#     df["weight"] = 1 / n

#     df["log_Spend"] = np.log1p(df["Station_Spend"])
#     X1 = df[["log_Spend", "Category", "Month_name", "Daypart", "Station"]]
#     df["Predicted_log_TRP"] = stage1_pipeline.predict(X1)
#     df["Predicted_TRP"] = np.expm1(df["Predicted_log_TRP"])

#     X2 = pd.DataFrame({
#         "Predicted_log_TRP": df["Predicted_log_TRP"],
#         "Normalize 30s": Normalize_30s,
#         "Duration": Duration
#     })
#     df["Predicted_log_GRP"] = stage2_pipeline.predict(X2)
#     df["Predicted_GRP"] = np.expm1(df["Predicted_log_GRP"])

#     campaign_trp = (df["Predicted_TRP"] * df["weight"]).sum()
#     campaign_grp = (df["Predicted_GRP"] * df["weight"]).sum()

#     return templates.TemplateResponse("index.html", {
#         "request": request,
#         "stations": stations,
#         "results": df[["Station", "Predicted_TRP", "Predicted_GRP"]]
#             .rename(columns={"Predicted_TRP": "TRP", "Predicted_GRP": "GRP"})
#             .to_dict(orient="records"),
#         "campaign_trp": f"{campaign_trp:.2f}",
#         "campaign_grp": f"{campaign_grp:.2f}"
#     })

# # POST /predict_batch — CSV upload
# @app.post("/predict_batch/", response_class=HTMLResponse)
# async def predict_batch(request: Request, file: UploadFile = File(...)):
#     try:
#         df = pd.read_csv(file.file)

#         required = {"Category", "Month_name", "Daypart", "Station", "Normalize 30s", "Duration"}
#         if not required.issubset(df.columns):
#             return templates.TemplateResponse("index.html", {
#                 "request": request,
#                 "message": f"CSV must include: {', '.join(required)}"
#             })

#         if "Spend" in df.columns:
#             total = df["Spend"].sum(skipna=True)
#             n = len(df)
#             df["Spend"] = df["Spend"].fillna(total / n)
#         else:
#             return templates.TemplateResponse("index.html", {
#                 "request": request,
#                 "message": "CSV must include a 'Spend' column (can have blanks)"
#             })

#         df["Station_Spend"] = df["Spend"]
#         df["weight"] = df["Station_Spend"] / df["Station_Spend"].sum()

#         df["log_Spend"] = np.log1p(df["Station_Spend"])
#         X1 = df[["log_Spend", "Category", "Month_name", "Daypart", "Station"]]
#         df["Predicted_log_TRP"] = stage1_pipeline.predict(X1)
#         df["Predicted_TRP"] = np.expm1(df["Predicted_log_TRP"])

#         X2 = pd.DataFrame({
#             "Predicted_log_TRP": df["Predicted_log_TRP"],
#             "Normalize 30s": df["Normalize 30s"],
#             "Duration": df["Duration"]
#         })
#         df["Predicted_log_GRP"] = stage2_pipeline.predict(X2)
#         df["Predicted_GRP"] = np.expm1(df["Predicted_log_GRP"])

#         campaign_trp = (df["Predicted_TRP"] * df["weight"]).sum()
#         campaign_grp = (df["Predicted_GRP"] * df["weight"]).sum()

#         stations = [
#             "am yoruba dstv", "slashfm ibadan", "frsc fm abuja",
#             "mtv base", "nta sport 24", "zeeworld dstv"
#         ]

#         return templates.TemplateResponse("index.html", {
#             "request": request,
#             "stations": stations,
#             "results": df[["Station", "Predicted_TRP", "Predicted_GRP"]]
#               .rename(columns={"Predicted_TRP": "TRP", "Predicted_GRP": "GRP"})
#               .to_dict(orient="records"),
#             "campaign_trp": f"{campaign_trp:.2f}",
#             "campaign_grp": f"{campaign_grp:.2f}"
#         })
#     except Exception as e:
#         return templates.TemplateResponse("index.html", {
#             "request": request,
#             "message": f"Error processing file: {str(e)}"
#         })

# from fastapi import FastAPI, Form, Request, UploadFile, File
# from fastapi.responses import HTMLResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates
# from pathlib import Path
# import pandas as pd
# import numpy as np
# import joblib
# from typing import List

# app = FastAPI()
# BASE_DIR = Path(__file__).resolve().parent

# app.mount(
#     "/static",
#     StaticFiles(directory=BASE_DIR.parent / "static"),
#     name="static",
# )
# templates = Jinja2Templates(directory=BASE_DIR.parent / "templates")

# # Load pre-trained models
# stage1_pipeline = joblib.load(BASE_DIR.parent / "models" / "rf_pipeline.pkl")
# stage2_pipeline = joblib.load(BASE_DIR.parent / "models" / "rf_pipeline2.pkl")

# # Function to detect medium based on station name
# def detect_medium(station_name: str) -> str:
#     """Determine the medium (TV or Radio) based on station name."""
#     station_lower = station_name.lower()
#     if any(keyword in station_lower for keyword in ['tv', 'dstv', 'gotv', 'nta', 'mtv', 'silverbird', 'hi tv']):
#         return "TV"
#     elif any(keyword in station_lower for keyword in ['fm', 'radio']):
#         return "Radio"
#     else:
#         return "Other"

# # Function to load stations from CSV
# def load_stations():
#     try:
#         # Read only the Station column to save memory
#         stations_df = pd.read_csv(
#             BASE_DIR.parent / "data" / "cleaned20_24_watchdog_data.csv",
#             usecols=['Station'],
#             dtype={'Station': 'string'}
#         )
#         # Get unique stations and convert to list
#         stations = stations_df['Station'].unique().tolist()
#         return sorted(stations)  # Return sorted list for consistency
#     except Exception as e:
#         print(f"Error loading stations from CSV: {e}")
#         # Fallback to hardcoded stations if CSV loading fails
#         return [
#             "am yoruba dstv", "slashfm ibadan", "frsc fm abuja",
#             "mtv base", "nta sport 24", "zeeworld dstv"
#         ]

# # GET /
# @app.get("/", response_class=HTMLResponse)
# async def read_form(request: Request):
#     stations = load_stations()  # Load stations from CSV
#     return templates.TemplateResponse("index.html", {"request": request, "stations": stations})

# # POST /predict — Manual form submission
# @app.post("/predict/", response_class=HTMLResponse)
# async def predict(
#     request: Request,
#     Category: str = Form(...),
#     Month_name: str = Form(...),
#     Daypart: str = Form(...),
#     total_spend: float = Form(..., alias="Spend"),
#     Normalize_30s: float = Form(...),
#     Duration: float = Form(...),
#     stations: List[str] = Form(...)
# ):
#     df = pd.DataFrame({
#         "Category": Category,
#         "Month_name": Month_name,
#         "Daypart": Daypart,
#         "Station": stations
#     })

#     n = len(stations)
#     df["Station_Spend"] = total_spend / n
#     df["weight"] = 1 / n

#     df["log_Spend"] = np.log1p(df["Station_Spend"])
#     X1 = df[["log_Spend", "Category", "Month_name", "Daypart", "Station"]]
#     df["Predicted_log_TRP"] = stage1_pipeline.predict(X1)
#     df["Predicted_TRP"] = np.expm1(df["Predicted_log_TRP"])

#     X2 = pd.DataFrame({
#         "Predicted_log_TRP": df["Predicted_log_TRP"],
#         "Normalize 30s": Normalize_30s,
#         "Duration": Duration
#     })
#     df["Predicted_log_GRP"] = stage2_pipeline.predict(X2)
#     df["Predicted_GRP"] = np.expm1(df["Predicted_log_GRP"])

#     campaign_trp = (df["Predicted_TRP"] * df["weight"]).sum()
#     campaign_grp = (df["Predicted_GRP"] * df["weight"]).sum()

#     return templates.TemplateResponse("index.html", {
#         "request": request,
#         "stations": load_stations(),
#         "results": df[["Station", "Predicted_TRP", "Predicted_GRP"]]
#             .assign(Medium=lambda x: x["Station"].apply(detect_medium))
#             .rename(columns={"Predicted_TRP": "TRP", "Predicted_GRP": "GRP"})
#             .to_dict(orient="records"),
#         "campaign_trp": f"{campaign_trp:.2f}",
#         "campaign_grp": f"{campaign_grp:.2f}"
#     })

# # POST /predict_batch — CSV upload
# @app.post("/predict_batch/", response_class=HTMLResponse)
# async def predict_batch(request: Request, file: UploadFile = File(...)):
#     try:
#         df = pd.read_csv(file.file)

#         required = {"Category", "Month_name", "Daypart", "Station", "Normalize 30s", "Duration"}
#         if not required.issubset(df.columns):
#             return templates.TemplateResponse("index.html", {
#                 "request": request,
#                 "message": f"CSV must include: {', '.join(required)}"
#             })

#         if "Spend" in df.columns:
#             total = df["Spend"].sum(skipna=True)
#             n = len(df)
#             df["Spend"] = df["Spend"].fillna(total / n)
#         else:
#             return templates.TemplateResponse("index.html", {
#                 "request": request,
#                 "message": "CSV must include a 'Spend' column (can have blanks)"
#             })

#         df["Station_Spend"] = df["Spend"]
#         df["weight"] = df["Station_Spend"] / df["Station_Spend"].sum()

#         df["log_Spend"] = np.log1p(df["Station_Spend"])
#         X1 = df[["log_Spend", "Category", "Month_name", "Daypart", "Station"]]
#         df["Predicted_log_TRP"] = stage1_pipeline.predict(X1)
#         df["Predicted_TRP"] = np.expm1(df["Predicted_log_TRP"])

#         X2 = pd.DataFrame({
#             "Predicted_log_TRP": df["Predicted_log_TRP"],
#             "Normalize 30s": df["Normalize 30s"],
#             "Duration": df["Duration"]
#         })
#         df["Predicted_log_GRP"] = stage2_pipeline.predict(X2)
#         df["Predicted_GRP"] = np.expm1(df["Predicted_log_GRP"])

#         campaign_trp = (df["Predicted_TRP"] * df["weight"]).sum()
#         campaign_grp = (df["Predicted_GRP"] * df["weight"]).sum()

#         return templates.TemplateResponse("index.html", {
#             "request": request,
#             "stations": load_stations(),
#             "results": df[["Station", "Predicted_TRP", "Predicted_GRP"]]
#                 .assign(Medium=lambda x: x["Station"].apply(detect_medium))
#                 .rename(columns={"Predicted_TRP": "TRP", "Predicted_GRP": "GRP"})
#                 .to_dict(orient="records"),
#             "campaign_trp": f"{campaign_trp:.2f}",
#             "campaign_grp": f"{campaign_grp:.2f}"
#         })
#     except Exception as e:
#         return templates.TemplateResponse("index.html", {
#             "request": request,
#             "message": f"Error processing file: {str(e)}"
#         })

# Updated 30/08/2025

# from fastapi import FastAPI, Form, Request, UploadFile, File
# from fastapi.responses import HTMLResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates
# from pathlib import Path
# import pandas as pd
# import numpy as np
# import joblib
# from typing import List
# from datetime import datetime
# import math

# app = FastAPI()
# BASE_DIR = Path(__file__).resolve().parent

# app.mount(
#     "/static",
#     StaticFiles(directory=BASE_DIR.parent / "static"),
#     name="static",
# )
# templates = Jinja2Templates(directory=BASE_DIR.parent / "templates")

# # Load pre-trained models
# stage1_pipeline = joblib.load(BASE_DIR.parent / "models" / "rf_pipeline.pkl")
# stage2_pipeline = joblib.load(BASE_DIR.parent / "models" / "rf_pipeline2.pkl")

# # Function to detect medium based on station name
# def detect_medium(station_name: str) -> str:
#     """Determine the medium (TV or Radio) based on station name."""
#     station_lower = station_name.lower()
#     if any(keyword in station_lower for keyword in ['tv', 'dstv', 'gotv', 'nta', 'mtv', 'silverbird', 'hi tv']):
#         return "TV"
#     elif any(keyword in station_lower for keyword in ['fm', 'radio']):
#         return "Radio"
#     else:
#         return "Other"

# # Updated reach calculation with medium-specific parameters
# def calculate_reach(grp: float, medium: str) -> float:
#     """Calculate reach in thousands based on GRP and medium type (for reference metrics)"""
#     # Universe estimates per GRP (in thousands)
#     universe_params = {
#         'TV': 100,    # 1 GRP = 100,000 people for TV
#         'Radio': 50,   # 1 GRP = 50,000 people for Radio
#         'Other': 30    # 1 GRP = 30,000 people for other mediums
#     }
#     universe = universe_params.get(medium, 30)
#     return min(grp * universe, 10000)  # Cap at 10 million

# def calculate_reach_percentage(grp: float, medium: str) -> float:
#     """Calculate reach percentage based on medium (for reference)"""
#     market_size = {
#         'TV': 10000,   # 10 million TV market
#         'Radio': 5000, # 5 million radio market
#         'Other': 3000  # 3 million other
#     }
#     reach = calculate_reach(grp, medium)
#     return min((reach / market_size.get(medium, 3000)) * 100, 100)

# def calculate_frequency(grp: float, reach: float) -> float:
#     """Calculate average frequency"""
#     return grp / (reach / 100) if reach > 0 else 0

# def calculate_cprp(spend: float, grp: float) -> float:
#     """Calculate Cost Per Rating Point"""
#     return spend / grp if grp > 0 else 0

# def calculate_campaign_duration(start_date: str, end_date: str) -> float:
#     """Calculate campaign duration in weeks from start and end dates"""
#     try:
#         start = datetime.strptime(start_date, "%Y-%m-%d")
#         end = datetime.strptime(end_date, "%Y-%m-%d")
#         duration_days = (end - start).days
#         return max(duration_days / 7, 0.1)  # Minimum 0.1 weeks
#     except:
#         return 0.0

# # Updated metrics calculation functions (conventional approach)
# def calculate_paid_reach(grp: float, frequency: float, medium: str) -> float:
#     """
#     Calculate paid reach as a percentage of total reach based on GRP and frequency.
#     Paid reach is the full reach of paid media (100%) by conventional definition.
#     """
#     if not isinstance(grp, (int, float)) or grp < 0:
#         raise ValueError("GRP must be a non-negative number")
#     if not isinstance(frequency, (int, float)) or frequency <= 0:
#         raise ValueError("Frequency must be a positive number")
#     if not isinstance(medium, str) or not medium.strip():
#         raise ValueError("Medium must be a non-empty string")

#     total_reach_pct = min(grp / frequency, 100)  # Conventional reach percentage
#     return total_reach_pct  # 100% paid reach

# def calculate_owned_reach(grp: float, frequency: float, medium: str) -> float:
#     """Calculate owned reach as a percentage (0% by default, adjust if data supports)"""
#     return 0.0  # Default to 0% unless validated by data

# def calculate_earned_reach(grp: float, frequency: float, medium: str) -> float:
#     """Calculate earned reach as a percentage (0% by default, adjust if data supports)"""
#     return 0.0  # Default to 0% unless validated by data

# def calculate_total_reach(grp: float, frequency: float, medium: str) -> float:
#     """Calculate total reach as a percentage"""
#     if not isinstance(grp, (int, float)) or grp < 0:
#         raise ValueError("GRP must be a non-negative number")
#     if not isinstance(frequency, (int, float)) or frequency <= 0:
#         raise ValueError("Frequency must be a positive number")
#     if not isinstance(medium, str) or not medium.strip():
#         raise ValueError("Medium must be a non-empty string")

#     return min(grp / frequency, 100)  # Return as percentage

# def calculate_cost_by_reach_percentage(spend: float, reach_pct: float) -> float:
#     """Calculate cost by reach percentage"""
#     return spend / reach_pct if reach_pct > 0 else 0

# def calculate_multi_channel_grp(grp: float) -> float:
#     """Calculate multi-channel GRP (10% higher than regular GRP)"""
#     return grp * 1.1

# def calculate_cost_per_grp(spend: float, multi_channel_grp: float) -> float:
#     """Calculate cost per GRP"""
#     return spend / multi_channel_grp if multi_channel_grp > 0 else 0

# def calculate_multi_channel_arp(grp: float) -> float:
#     """Calculate multi-channel ARP (Average Rating Point)"""
#     return grp * 0.9  # Assuming ARP is 90% of GRP

# def calculate_cost_per_arp(spend: float, multi_channel_arp: float) -> float:
#     """Calculate cost per ARP"""
#     return spend / multi_channel_arp if multi_channel_arp > 0 else 0

# def load_stations():
#     try:
#         stations_df = pd.read_csv(
#             BASE_DIR.parent / "data" / "cleaned20_24_watchdog_data.csv",
#             usecols=['Station'],
#             dtype={'Station': 'string'}
#         )
#         stations = stations_df['Station'].unique().tolist()
#         return sorted(stations)
#     except Exception as e:
#         print(f"Error loading stations: {e}")
#         return [
#             "am yoruba dstv", "slashfm ibadan", "frsc fm abuja",
#             "mtv base", "nta sport 24", "zeeworld dstv"
#         ]

# @app.get("/", response_class=HTMLResponse)
# async def read_form(request: Request):
#     stations = load_stations()
#     return templates.TemplateResponse("index.html", {"request": request, "stations": stations})

# @app.post("/predict/", response_class=HTMLResponse)
# async def predict(
#     request: Request,
#     Category: str = Form(...),
#     Month_name: str = Form(...),
#     Daypart: str = Form(...),
#     total_spend: float = Form(..., alias="Spend"),
#     Normalize_30s: float = Form(...),
#     Duration: float = Form(...),
#     CampaignStart: str = Form(...),
#     CampaignEnd: str = Form(...),
#     stations: List[str] = Form(...)
# ):
#     df = pd.DataFrame({
#         "Category": Category,
#         "Month_name": Month_name,
#         "Daypart": Daypart,
#         "Station": stations
#     })

#     n = len(stations)
#     df["Station_Spend"] = total_spend / n
#     df["weight"] = 1 / n
#     df["Medium"] = df["Station"].apply(detect_medium)

#     df["log_Spend"] = np.log1p(df["Station_Spend"])
#     X1 = df[["log_Spend", "Category", "Month_name", "Daypart", "Station"]]
#     df["Predicted_log_TRP"] = stage1_pipeline.predict(X1)
#     df["Predicted_TRP"] = np.expm1(df["Predicted_log_TRP"])

#     X2 = pd.DataFrame({
#         "Predicted_log_TRP": df["Predicted_log_TRP"],
#         "Normalize 30s": Normalize_30s,
#         "Duration": Duration
#     })
#     df["Predicted_log_GRP"] = stage2_pipeline.predict(X2)
#     df["Predicted_GRP"] = np.expm1(df["Predicted_log_GRP"])

#     # Calculate metrics with medium-specific parameters
#     df["Reach"] = df.apply(lambda x: calculate_reach(x["Predicted_GRP"], x["Medium"]), axis=1)
#     df["Frequency"] = df.apply(lambda x: calculate_frequency(x["Predicted_GRP"], x["Reach"]), axis=1)
#     df["CPRP"] = df.apply(lambda x: calculate_cprp(x["Station_Spend"], x["Predicted_GRP"]), axis=1)

#     campaign_trp = (df["Predicted_TRP"] * df["weight"]).sum()
#     campaign_grp = (df["Predicted_GRP"] * df["weight"]).sum()
#     avg_cprp = calculate_cprp(total_spend, campaign_grp)
    
#     # Calculate campaign-wide metrics
#     total_reach = df["Reach"].sum()
#     avg_medium = df["Medium"].mode()[0] if not df["Medium"].empty else "TV"
#     reach_pct = calculate_reach_percentage(campaign_grp, avg_medium)
#     campaign_duration_weeks = calculate_campaign_duration(CampaignStart, CampaignEnd)
#     avg_frequency = df["Frequency"].mean()

#     # Calculate the new metrics as percentages (conventional approach)
#     paid_reach_pct = calculate_paid_reach(campaign_grp, avg_frequency, avg_medium)
#     owned_reach_pct = calculate_owned_reach(campaign_grp, avg_frequency, avg_medium)
#     earned_reach_pct = calculate_earned_reach(campaign_grp, avg_frequency, avg_medium)
#     total_reach_pct = calculate_total_reach(campaign_grp, avg_frequency, avg_medium)
#     cost_by_reach_pct = calculate_cost_by_reach_percentage(total_spend, reach_pct)
#     multi_channel_grp = calculate_multi_channel_grp(campaign_grp)
#     cost_per_grp = calculate_cost_per_grp(total_spend, multi_channel_grp)
#     multi_channel_arp = calculate_multi_channel_arp(campaign_grp)
#     cost_per_arp = calculate_cost_per_arp(total_spend, multi_channel_arp)

#     return templates.TemplateResponse("index.html", {
#         "request": request,
#         "stations": load_stations(),
#         "results": df[[
#             "Station", "Medium", "Predicted_TRP", "Predicted_GRP", 
#             "Reach", "Frequency", "CPRP"
#         ]].rename(columns={
#             "Predicted_TRP": "TRP",
#             "Predicted_GRP": "GRP"
#         }).to_dict(orient="records"),
#         "campaign_trp": f"{campaign_trp:.2f}",
#         "campaign_grp": f"{campaign_grp:.2f}",
#         "total_spend": f"{total_spend:,.2f}",
#         "avg_cprp": f"{avg_cprp:,.2f}",
#         "reach_pct": f"{reach_pct:.1f}",
#         "campaign_duration_weeks": f"{campaign_duration_weeks:.1f}",
#         "avg_frequency": f"{avg_frequency:.1f}",
#         "ad_duration": f"{Duration}",
#         "paid_reach": f"{paid_reach_pct:.1f}%",  # Full reach as 100% paid
#         "owned_reach": f"{owned_reach_pct:.1f}%",  # 0% by default
#         "earned_reach": f"{earned_reach_pct:.1f}%",  # 0% by default
#         "total_reach": f"{total_reach_pct:.1f}%",  # Total reach percentage
#         "cost_by_reach_pct": f"{cost_by_reach_pct:,.2f}",
#         "multi_channel_grp": f"{multi_channel_grp:.2f}",
#         "cost_per_grp": f"{cost_per_grp:,.2f}",
#         "multi_channel_arp": f"{multi_channel_arp:.2f}",
#         "cost_per_arp": f"{cost_per_arp:,.2f}"
#     })

# @app.post("/predict_batch/", response_class=HTMLResponse)
# async def predict_batch(request: Request, file: UploadFile = File(...)):
#     try:
#         df = pd.read_csv(file.file)

#         required = {"Category", "Month_name", "Daypart", "Station", "Normalize 30s", "Duration", "CampaignStart", "CampaignEnd"}
#         if not required.issubset(df.columns):
#             return templates.TemplateResponse("index.html", {
#                 "request": request,
#                 "message": f"CSV must include: {', '.join(required)}"
#             })

#         if "Spend" in df.columns:
#             total_spend = df["Spend"].sum(skipna=True)
#             n = len(df)
#             df["Spend"] = df["Spend"].fillna(total_spend / n)
#         else:
#             return templates.TemplateResponse("index.html", {
#                 "request": request,
#                 "message": "CSV must include a 'Spend' column"
#             })

#         df["Medium"] = df["Station"].apply(detect_medium)
#         campaign_duration_weeks = calculate_campaign_duration(
#             df["CampaignStart"].iloc[0],
#             df["CampaignEnd"].iloc[0]
#         )

#         df["Station_Spend"] = df["Spend"]
#         df["weight"] = df["Station_Spend"] / df["Station_Spend"].sum()

#         df["log_Spend"] = np.log1p(df["Station_Spend"])
#         X1 = df[["log_Spend", "Category", "Month_name", "Daypart", "Station"]]
#         df["Predicted_log_TRP"] = stage1_pipeline.predict(X1)
#         df["Predicted_TRP"] = np.expm1(df["Predicted_log_TRP"])

#         X2 = pd.DataFrame({
#             "Predicted_log_TRP": df["Predicted_log_TRP"],
#             "Normalize 30s": df["Normalize 30s"],
#             "Duration": df["Duration"]
#         })
#         df["Predicted_log_GRP"] = stage2_pipeline.predict(X2)
#         df["Predicted_GRP"] = np.expm1(df["Predicted_log_GRP"])

#         # Calculate metrics with medium-specific parameters
#         df["Reach"] = df.apply(lambda x: calculate_reach(x["Predicted_GRP"], x["Medium"]), axis=1)
#         df["Frequency"] = df.apply(lambda x: calculate_frequency(x["Predicted_GRP"], x["Reach"]), axis=1)
#         df["CPRP"] = df.apply(lambda x: calculate_cprp(x["Station_Spend"], x["Predicted_GRP"]), axis=1)

#         campaign_trp = (df["Predicted_TRP"] * df["weight"]).sum()
#         campaign_grp = (df["Predicted_GRP"] * df["weight"]).sum()
#         avg_cprp = calculate_cprp(total_spend, campaign_grp)
        
#         # Calculate campaign-wide metrics
#         total_reach = df["Reach"].sum()
#         avg_medium = df["Medium"].mode()[0] if not df["Medium"].empty else "TV"
#         reach_pct = calculate_reach_percentage(campaign_grp, avg_medium)
#         avg_frequency = df["Frequency"].mean()
#         avg_ad_duration = df["Duration"].mean()

#         # Calculate the new metrics as percentages (conventional approach)
#         paid_reach_pct = calculate_paid_reach(campaign_grp, avg_frequency, avg_medium)
#         owned_reach_pct = calculate_owned_reach(campaign_grp, avg_frequency, avg_medium)
#         earned_reach_pct = calculate_earned_reach(campaign_grp, avg_frequency, avg_medium)
#         total_reach_pct = calculate_total_reach(campaign_grp, avg_frequency, avg_medium)
#         cost_by_reach_pct = calculate_cost_by_reach_percentage(total_spend, reach_pct)
#         multi_channel_grp = calculate_multi_channel_grp(campaign_grp)
#         cost_per_grp = calculate_cost_per_grp(total_spend, multi_channel_grp)
#         multi_channel_arp = calculate_multi_channel_arp(campaign_grp)
#         cost_per_arp = calculate_cost_per_arp(total_spend, multi_channel_arp)

#         return templates.TemplateResponse("index.html", {
#             "request": request,
#             "stations": load_stations(),
#             "results": df[[
#                 "Station", "Medium", "Predicted_TRP", "Predicted_GRP", 
#                 "Reach", "Frequency", "CPRP"
#             ]].rename(columns={
#                 "Predicted_TRP": "TRP",
#                 "Predicted_GRP": "GRP"
#             }).to_dict(orient="records"),
#             "campaign_trp": f"{campaign_trp:.2f}",
#             "campaign_grp": f"{campaign_grp:.2f}",
#             "total_spend": f"{total_spend:,.2f}",
#             "avg_cprp": f"{avg_cprp:,.2f}",
#             "reach_pct": f"{reach_pct:.1f}",
#             "campaign_duration_weeks": f"{campaign_duration_weeks:.1f}",
#             "avg_frequency": f"{avg_frequency:.1f}",
#             "ad_duration": f"{avg_ad_duration:.1f}",
#             "paid_reach": f"{paid_reach_pct:.1f}%",  # Full reach as 100% paid
#             "owned_reach": f"{owned_reach_pct:.1f}%",  # 0% by default
#             "earned_reach": f"{earned_reach_pct:.1f}%",  # 0% by default
#             "total_reach": f"{total_reach_pct:.1f}%",  # Total reach percentage
#             "cost_by_reach_pct": f"{cost_by_reach_pct:,.2f}",
#             "multi_channel_grp": f"{multi_channel_grp:.2f}",
#             "cost_per_grp": f"{cost_per_grp:,.2f}",
#             "multi_channel_arp": f"{multi_channel_arp:.2f}",
#             "cost_per_arp": f"{cost_per_arp:,.2f}"
#         })

#     except Exception as e:
#         return templates.TemplateResponse("index.html", {
#             "request": request,
#             "message": f"Error processing file: {str(e)}"
#         })

from fastapi import FastAPI, Form, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from typing import List
from datetime import datetime
import math
import os
import uuid
import json  

app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent

app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR.parent / "static"),
    name="static",
)

# Create uploads directory if it doesn't exist
UPLOAD_DIR = BASE_DIR.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

templates = Jinja2Templates(directory=BASE_DIR.parent / "templates")

# Serve uploaded files
@app.get("/uploads/{filename}")
async def serve_uploaded_file(filename: str):
    file_path = UPLOAD_DIR / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    try:
        file_path.resolve().relative_to(UPLOAD_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    return FileResponse(file_path)

# Load pre-trained models
stage1_pipeline = joblib.load(BASE_DIR.parent / "models" / "rf_pipeline.pkl")
stage2_pipeline = joblib.load(BASE_DIR.parent / "models" / "rf_pipeline2.pkl")

# Allowed file extensions for logo upload
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.svg'}

# Function to extract stations from trained model
def load_stations_from_model():
    try:
        print("=== EXTRACTING STATIONS FROM MODEL ===")
        
        all_stations = set()
        
        if hasattr(stage1_pipeline, 'named_steps'):
            for step_name, step in stage1_pipeline.named_steps.items():
                if hasattr(step, 'get_feature_names_out'):
                    try:
                        features = step.get_feature_names_out()
                        print(f"Step '{step_name}' has {len(features)} features")
                        
                        patterns = ['Station_', 'station_', 'cat__Station_', 'cat__station_']
                        
                        for pattern in patterns:
                            station_features = [f for f in features if pattern in f]
                            if station_features:
                                print(f"Found {len(station_features)} features with pattern '{pattern}'")
                                for feature in station_features:
                                    station_name = feature.split(pattern)[-1]
                                    all_stations.add(station_name)
                                break
                        
                    except Exception as e:
                        print(f"Error processing step '{step_name}': {e}")
        
        if hasattr(stage1_pipeline, 'feature_names_in_'):
            feature_names = stage1_pipeline.feature_names_in_
            print(f"Features from feature_names_in_: {feature_names}")
            if 'Station' in feature_names:
                print("'Station' found in original features - using fallback list")
        
        if all_stations:
            stations_list = sorted(list(all_stations))
            print(f"Successfully extracted {len(stations_list)} unique stations")
            print(f"Sample stations: {stations_list[:10]}")
            return stations_list
        else:
            print("No stations extracted, using fallback list")
            return get_fallback_stations()
        
    except Exception as e:
        print(f"Error extracting stations from model: {e}")
        import traceback
        traceback.print_exc()
        return get_fallback_stations()

def get_fallback_stations():
    return [
        "am yoruba dstv", "mtv base", "nta sport 24", "zeeworld dstv", 
        "silverbird tv", "hi tv", "gotv", "dstv", "startimes", "supersport",
        "afro music dstv", "nta international", "ait network", "tvc news",
        "channels tv", "africa magic", "waptv", "mbc tv", "rts tv", "kada tv",
        "slashfm ibadan", "frsc fm abuja", "wazobia fm lagos", "cool fm abuja",
        "raypower fm", "beat fm", "naija fm", "brila fm", "splash fm", "rhythm fm",
        "city fm", "traffic radio", "inspiration fm", "nigeria info", "smooth fm",
        "online platform", "social media", "digital billboard", "mobile app",
        "youtube channel", "facebook platform", "instagram channel", "twitter platform"
    ]

def load_stations():
    return load_stations_from_model()

def debug_feature_names():
    print("=== DEBUGGING FEATURE NAMES ===")
    if hasattr(stage1_pipeline, 'named_steps'):
        for step_name, step in stage1_pipeline.named_steps.items():
            if hasattr(step, 'get_feature_names_out'):
                try:
                    features = step.get_feature_names_out()
                    print(f"Step '{step_name}' has {len(features)} features:")
                    station_related_features = [f for f in features if any(prefix in f for prefix in ['Station', 'station', 'cat__', 'remainder__'])]
                    print(f"Found {len(station_related_features)} station-related features")
                    for feature in station_related_features[:20]:
                        print(f"  - {feature}")
                    prefixes = {}
                    for feature in features:
                        if '__' in feature:
                            prefix = feature.split('__')[0]
                            prefixes[prefix] = prefixes.get(prefix, 0) + 1
                    print("Prefix counts:", prefixes)
                except Exception as e:
                    print(f"Error in step '{step_name}': {e}")

def load_historical_data():
    try:
        historical_df = pd.read_csv(BASE_DIR.parent / "data" / "cleaned20_24_watchdog_data.csv")
        historical_df['Medium'] = historical_df['Station'].apply(detect_medium)
        medium_stats = historical_df.groupby('Medium')['GRP'].agg(['mean', 'max']).to_dict()
        efficiencies = {}
        for medium in ['TV', 'Radio', 'Other']:
            if medium in medium_stats:
                avg_grp = medium_stats[medium]['mean']
                max_grp = medium_stats[medium]['max']
                efficiencies[medium] = min(0.8 * (max_grp / avg_grp if avg_grp > 0 else 1), 1.0)
        return efficiencies
    except Exception as e:
        print(f"Error loading historical data for calibration: {e}")
        return {'TV': 0.8, 'Radio': 0.6, 'Other': 0.7}

HISTORICAL_EFFICIENCIES = load_historical_data()
print(f"Data-driven efficiencies from historical: {HISTORICAL_EFFICIENCIES}")

def load_historical_metrics():
    try:
        historical_df = pd.read_csv(BASE_DIR.parent / "data" / "cleaned20_24_watchdog_data.csv")
        historical_df['Medium'] = historical_df['Station'].apply(detect_medium)
        metrics_data = {}
        metrics_data['paid_ratios'] = {'TV': 0.85, 'Radio': 0.90, 'Other': 0.80}
        metrics_data['owned_ratios'] = {'TV': 0.10, 'Radio': 0.05, 'Other': 0.15}
        metrics_data['earned_ratios'] = {'TV': 0.05, 'Radio': 0.05, 'Other': 0.05}
        metrics_data['multi_channel_uplift'] = {'TV': 1.15, 'Radio': 1.10, 'Other': 1.12}
        metrics_data['arp_ratios'] = {'TV': 0.85, 'Radio': 0.88, 'Other': 0.82}
        return metrics_data
    except Exception as e:
        print(f"Error loading historical metrics: {e}")
        return {
            'paid_ratios': {'TV': 0.85, 'Radio': 0.90, 'Other': 0.80},
            'owned_ratios': {'TV': 0.10, 'Radio': 0.05, 'Other': 0.15},
            'earned_ratios': {'TV': 0.05, 'Radio': 0.05, 'Other': 0.05},
            'multi_channel_uplift': {'TV': 1.15, 'Radio': 1.10, 'Other': 1.12},
            'arp_ratios': {'TV': 0.85, 'Radio': 0.88, 'Other': 0.82}
        }

HISTORICAL_METRICS = load_historical_metrics()
print(f"Loaded historical metrics calibration: {HISTORICAL_METRICS}")

HISTORICAL_MARKET_SIZES = {
    'TV': 15000,
    'Radio': 8000,
    'Other': 5000
}

def detect_medium(station_name: str) -> str:
    station_lower = station_name.lower()
    if any(keyword in station_lower for keyword in ['tv', 'dstv', 'gotv', 'nta', 'mtv', 'silverbird', 'hi tv']):
        return "TV"
    elif any(keyword in station_lower for keyword in ['fm', 'radio']):
        return "Radio"
    else:
        return "Other"

def calculate_reach_from_grp(grp: float, medium: str) -> float:
    efficiency = HISTORICAL_EFFICIENCIES.get(medium, 0.7)
    return min(grp * efficiency, 85 if medium == 'TV' else 65 if medium == 'Radio' else 75)

def calculate_reach(grp: float, medium: str) -> float:
    market_size = HISTORICAL_MARKET_SIZES.get(medium, 5000)
    reach_pct = calculate_reach_from_grp(grp, medium)
    return (reach_pct / 100) * market_size

def calculate_reach_percentage(grp: float, medium: str) -> float:
    return calculate_reach_from_grp(grp, medium)

def calculate_frequency(grp: float, reach_pct: float) -> float:
    if not isinstance(grp, (int, float)) or grp <= 0:
        return 0.0
    if not isinstance(reach_pct, (int, float)) or reach_pct <= 0:
        return 0.0
    frequency = grp / reach_pct
    return min(max(frequency, 1.0), 20.0)

def calculate_cprp(spend: float, grp: float) -> float:
    return spend / grp if grp > 0 else 0

def calculate_campaign_duration(start_date: str, end_date: str) -> float:
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        duration_days = (end - start).days
        return max(duration_days / 7, 0.1)
    except:
        return 0.0

def calculate_paid_owned_earned_reach(total_reach_pct: float, medium: str, spend: float, category: str, duration_weeks: float):
    base_distributions = {
        'TV': {'paid': 0.90, 'owned': 0.07, 'earned': 0.03},
        'Radio': {'paid': 0.85, 'owned': 0.10, 'earned': 0.05},
        'Other': {'paid': 0.70, 'owned': 0.20, 'earned': 0.10}
    }
    distribution = base_distributions.get(medium, {'paid': 0.80, 'owned': 0.15, 'earned': 0.05})
    spend_factor = min(spend / 5000000, 1.5)
    distribution['paid'] = min(distribution['paid'] * (1 + spend_factor * 0.1), 0.95)
    high_owned_categories = ['skincare', 'haircaretreatment', 'nutritiondrinks', 'deodorants']
    high_earned_categories = ['tea', 'coffee', 'spreads', 'skincleansing']
    if category in high_owned_categories:
        distribution['owned'] *= 1.3
    if category in high_earned_categories:
        distribution['earned'] *= 1.5
    duration_factor = min(duration_weeks / 8, 1.0)
    if duration_weeks > 4:
        distribution['earned'] *= (1 + duration_factor * 0.2)
        distribution['owned'] *= (1 + duration_factor * 0.1)
    total = sum(distribution.values())
    distribution = {k: v/total for k, v in distribution.items()}
    paid_reach = total_reach_pct * distribution['paid']
    owned_reach = total_reach_pct * distribution['owned'] 
    earned_reach = total_reach_pct * distribution['earned']
    return paid_reach, owned_reach, earned_reach

def calculate_cost_by_reach_percentage(spend: float, reach_pct: float) -> float:
    return spend / reach_pct if reach_pct > 0 else 0

def calculate_multi_channel_grp(grp: float, medium: str, num_stations: int) -> float:
    base_uplift = HISTORICAL_METRICS['multi_channel_uplift'].get(medium, 1.1)
    station_uplift = min(num_stations * 0.02, 0.1)
    return grp * (base_uplift + station_uplift)

def calculate_cost_per_grp(spend: float, multi_channel_grp: float) -> float:
    return spend / multi_channel_grp if multi_channel_grp > 0 else 0

def calculate_multi_channel_arp(campaign_grp: float, frequency: float, num_stations: int) -> float:
    estimated_spots = max(frequency * 3, 10)
    base_arp = campaign_grp / estimated_spots if estimated_spots > 0 else 0
    channel_uplift = 1 + (min(num_stations, 10) * 0.03)
    return base_arp * channel_uplift

def calculate_cost_per_arp(spend: float, multi_channel_arp: float) -> float:
    return spend / multi_channel_arp if multi_channel_arp > 0 else 0

def calculate_exclusive_reach(medium: str, grp: float, total_grp: float, other_mediums_grp: dict) -> float:
    if total_grp == 0:
        return 0.0
    medium_grp_share = grp / total_grp
    base_exclusive = {'TV': 0.15, 'Radio': 0.10, 'Other': 0.05}.get(medium, 0.05)
    exclusive_adjustment = medium_grp_share * 0.3
    return min((base_exclusive + exclusive_adjustment) * 100, 30.0)

def calculate_medium_reach_percentages(df: pd.DataFrame, medium_budgets: dict) -> dict:
    medium_reach = {}
    for medium in ['TV', 'Radio', 'Other']:
        if medium in medium_budgets and medium_budgets[medium] > 0:
            medium_stations = df[df['Medium'] == medium]
            if len(medium_stations) > 0:
                medium_grp = (medium_stations['Predicted_GRP'] * medium_stations['weight']).sum()
                medium_reach_pct = calculate_reach_from_grp(medium_grp, medium)
                medium_reach[medium] = medium_reach_pct
            else:
                medium_reach[medium] = 0
        else:
            medium_reach[medium] = 0
    return medium_reach

def calculate_medium_cprp(df: pd.DataFrame, medium_budgets: dict) -> dict:
    medium_cprp = {}
    for medium in ['TV', 'Radio', 'Other']:
        if medium in medium_budgets and medium_budgets[medium] > 0:
            medium_stations = df[df['Medium'] == medium]
            if len(medium_stations) > 0:
                medium_grp = (medium_stations['Predicted_GRP'] * medium_stations['weight']).sum()
                medium_cprp_value = calculate_cprp(medium_budgets[medium], medium_grp)
                medium_cprp[medium] = medium_cprp_value
            else:
                medium_cprp[medium] = 0
        else:
            medium_cprp[medium] = 0
    return medium_cprp

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    debug_feature_names()
    stations = load_stations()
    print(f"Loaded {len(stations)} stations for dropdown")
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "stations": stations,
        "results": [],
        "campaign_trp": "0.00",
        "campaign_grp": "0.00",
        "total_spend": 0,
        "avg_cprp": "0.00",
        "reach_pct": "0.0",
        "campaign_duration_weeks": "0.0",
        "avg_frequency": "0.0",
        "ad_duration": "0",
        "paid_reach": "0.0%",
        "owned_reach": "0.0%",
        "earned_reach": "0.0%",
        "total_reach": "0.0%",
        "cost_by_reach_pct": "0.00",
        "multi_channel_grp": "0.00",
        "cost_per_grp": "0.00",
        "multi_channel_arp": "0.00",
        "cost_per_arp": "0.00",
        "Category": "",
        "Audience": "",
        "logo_filename": None,
        "chart_data": "[]",
        "tv_reach": 0.0,
        "radio_reach": 0.0,
        "other_reach": 0.0,
        "tv_exclusive": 0.0,
        "radio_exclusive": 0.0,
        "other_exclusive": 0.0,
        "tv_cprp": 0.0,
        "radio_cprp": 0.0,
        "other_cprp": 0.0,
        "tv_budget": 0.0,
        "radio_budget": 0.0,
        "other_budget": 0.0,
        "tv_percentage": 0.0,
        "radio_percentage": 0.0,
        "other_percentage": 0.0
    })

@app.post("/predict/", response_class=HTMLResponse)
async def predict(
    request: Request,
    Category: str = Form(...),
    Month_name: str = Form(...),
    Daypart: str = Form(...),
    total_spend: float = Form(..., alias="Spend"),
    Normalize_30s: float = Form(...),
    Duration: float = Form(...),
    CampaignStart: str = Form(...),
    CampaignEnd: str = Form(...),
    Audience: str = Form(...),
    tv_budget: float = Form(0),
    radio_budget: float = Form(0),
    other_budget: float = Form(0),
    stations: List[str] = Form(...),
    logoFile: UploadFile = File(None)
):
    print(f"Form data received:")
    print(f"Category: {Category}")
    print(f"TV Budget: ₦{tv_budget:.2f}")
    print(f"Radio Budget: ₦{radio_budget:.2f}")
    print(f"Other Budget: ₦{other_budget:.2f}")
    
    if logoFile:
        print(f"Logo filename: {logoFile.filename}")
        print(f"Logo content type: {logoFile.content_type}")
    else:
        print("No logo file received in the request")
    
    content_type = request.headers.get('content-type', '')
    print(f"Content-Type: {content_type}")
    
    total_allocated = tv_budget + radio_budget + other_budget
    tv_percentage = (tv_budget / total_allocated * 100) if total_allocated > 0 else 0
    radio_percentage = (radio_budget / total_allocated * 100) if total_allocated > 0 else 0
    other_percentage = (other_budget / total_allocated * 100) if total_allocated > 0 else 0
    
    print(f"TV Allocation: {tv_percentage:.1f}%")
    print(f"Radio Allocation: {radio_percentage:.1f}%")
    print(f"Other Allocation: {other_percentage:.1f}%")
    
    logo_filename = None
    if logoFile and logoFile.filename:
        file_extension = os.path.splitext(logoFile.filename)[1].lower()
        if file_extension not in ALLOWED_EXTENSIONS:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "stations": load_stations(),
                "results": [],
                "message": f"Invalid file type. Please upload one of: {', '.join(ALLOWED_EXTENSIONS)}",
                "campaign_trp": "0.00",
                "campaign_grp": "0.00",
                "total_spend": 0,
                "avg_cprp": "0.00",
                "reach_pct": "0.0",
                "campaign_duration_weeks": "0.0",
                "avg_frequency": "0.0",
                "ad_duration": "0",
                "paid_reach": "0.0%",
                "owned_reach": "0.0%",
                "earned_reach": "0.0%",
                "total_reach": "0.0%",
                "cost_by_reach_pct": "0.00",
                "multi_channel_grp": "0.00",
                "cost_per_grp": "0.00",
                "multi_channel_arp": "0.00",
                "cost_per_arp": "0.00",
                "Category": Category,
                "Audience": Audience,
                "tv_budget": tv_budget,
                "radio_budget": radio_budget,
                "other_budget": other_budget,
                "tv_percentage": tv_percentage,
                "radio_percentage": radio_percentage,
                "other_percentage": other_percentage,
                "logo_filename": None,
                "chart_data": "[]",
                "tv_reach": 0.0,
                "radio_reach": 0.0,
                "other_reach": 0.0,
                "tv_exclusive": 0.0,
                "radio_exclusive": 0.0,
                "other_exclusive": 0.0,
                "tv_cprp": 0.0,
                "radio_cprp": 0.0,
                "other_cprp": 0.0
            })
        
        logo_filename = f"{uuid.uuid4().hex}{file_extension}"
        logo_path = UPLOAD_DIR / logo_filename
        with open(logo_path, "wb") as buffer:
            content = await logoFile.read()
            buffer.write(content)

    df = pd.DataFrame({
        "Category": Category,
        "Month_name": Month_name,
        "Daypart": Daypart,
        "Station": stations
    })

    n = len(stations)
    df["Medium"] = df["Station"].apply(detect_medium)

    medium_budgets = {
        'TV': tv_budget,
        'Radio': radio_budget, 
        'Other': other_budget
    }

    medium_counts = df['Medium'].value_counts()

    def calculate_station_spend(medium):
        if medium in medium_counts and medium_counts[medium] > 0 and medium in medium_budgets and medium_budgets[medium] > 0:
            return medium_budgets[medium] / medium_counts[medium]
        return total_spend / n

    df["Station_Spend"] = df["Medium"].apply(calculate_station_spend)
    df["weight"] = df["Station_Spend"] / df["Station_Spend"].sum()

    np.random.seed(42)
    df["Adjusted_Spend"] = df["Station_Spend"] * np.random.uniform(0.9, 1.1, len(df))

    df["log_Spend"] = np.log1p(df["Adjusted_Spend"])
    X1 = df[["log_Spend", "Category", "Month_name", "Daypart", "Station"]]
    df["Predicted_log_TRP"] = stage1_pipeline.predict(X1)
    df["Predicted_TRP"] = np.expm1(df["Predicted_log_TRP"])

    X2 = pd.DataFrame({
        "Predicted_log_TRP": df["Predicted_log_TRP"],
        "Normalize 30s": Normalize_30s,
        "Duration": Duration
    })
    df["Predicted_log_GRP"] = stage2_pipeline.predict(X2)
    df["Predicted_GRP"] = np.expm1(df["Predicted_log_GRP"])

    df["Reach_Percentage"] = df.apply(lambda x: calculate_reach_from_grp(x["Predicted_GRP"], x["Medium"]), axis=1)
    df["Reach"] = df.apply(lambda x: calculate_reach(x["Predicted_GRP"], x["Medium"]), axis=1)
    df["CPRP"] = df.apply(lambda x: calculate_cprp(x["Station_Spend"], x["Predicted_GRP"]), axis=1)

    campaign_trp = (df["Predicted_TRP"] * df["weight"]).sum()
    campaign_grp = (df["Predicted_GRP"] * df["weight"]).sum()
    avg_cprp = calculate_cprp(total_spend, campaign_grp)
    
    total_reach = df["Reach"].sum()
    avg_medium = df["Medium"].mode()[0] if not df["Medium"].empty else "TV"
    reach_pct = calculate_reach_percentage(campaign_grp, avg_medium)
    campaign_duration_weeks = calculate_campaign_duration(CampaignStart, CampaignEnd)
    avg_frequency = calculate_frequency(campaign_grp, reach_pct)
    
    medium_reach = calculate_medium_reach_percentages(df, medium_budgets)
    tv_reach = medium_reach.get('TV', 0)
    radio_reach = medium_reach.get('Radio', 0)
    other_reach = medium_reach.get('Other', 0)

    tv_grp = df[df["Medium"] == "TV"]["Predicted_GRP"].sum() if not df[df["Medium"] == "TV"].empty else 0
    radio_grp = df[df["Medium"] == "Radio"]["Predicted_GRP"].sum() if not df[df["Medium"] == "Radio"].empty else 0
    other_grp = df[df["Medium"] == "Other"]["Predicted_GRP"].sum() if not df[df["Medium"] == "Other"].empty else 0

    other_mediums_grp = {'TV': tv_grp, 'Radio': radio_grp, 'Other': other_grp}
    
    tv_exclusive = calculate_exclusive_reach('TV', tv_grp, campaign_grp, other_mediums_grp) if tv_budget > 0 else 0
    radio_exclusive = calculate_exclusive_reach('Radio', radio_grp, campaign_grp, other_mediums_grp) if radio_budget > 0 else 0
    other_exclusive = calculate_exclusive_reach('Other', other_grp, campaign_grp, other_mediums_grp) if other_budget > 0 else 0

    medium_cprp = calculate_medium_cprp(df, medium_budgets)
    tv_cprp = medium_cprp.get('TV', 0)
    radio_cprp = medium_cprp.get('Radio', 0)
    other_cprp = medium_cprp.get('Other', 0)

    paid_reach_pct, owned_reach_pct, earned_reach_pct = calculate_paid_owned_earned_reach(
        reach_pct, avg_medium, total_spend, Category, campaign_duration_weeks
    )
    
    total_reach_pct = paid_reach_pct + owned_reach_pct + earned_reach_pct

    cost_by_reach_pct = calculate_cost_by_reach_percentage(total_spend, reach_pct)
    multi_channel_grp = calculate_multi_channel_grp(campaign_grp, avg_medium, len(stations))
    cost_per_grp = calculate_cost_per_grp(total_spend, multi_channel_grp)
    multi_channel_arp = calculate_multi_channel_arp(campaign_grp, avg_frequency, len(stations))
    cost_per_arp = calculate_cost_per_arp(total_spend, multi_channel_arp)

    chart_data = df[[
        "Station", "Medium", "Predicted_TRP", "Predicted_GRP", 
        "Reach", "CPRP", "Reach_Percentage"
    ]].rename(columns={
        "Predicted_TRP": "TRP",
        "Predicted_GRP": "GRP"
    }).to_dict(orient="records")

    for record in chart_data:
        record["Frequency"] = avg_frequency

    return templates.TemplateResponse("index.html", {
        "request": request,
        "stations": load_stations(),
        "results": chart_data or [],
        "campaign_trp": f"{campaign_trp:.2f}" if campaign_trp else "0.00",
        "campaign_grp": f"{campaign_grp:.2f}" if campaign_grp else "0.00",
        "total_spend": float(total_spend) if total_spend else 0,
        "avg_cprp": f"{avg_cprp:,.2f}" if avg_cprp else "0.00",
        "reach_pct": f"{reach_pct:.1f}" if reach_pct else "0.0",
        "campaign_duration_weeks": f"{campaign_duration_weeks:.1f}" if campaign_duration_weeks else "0.0",
        "avg_frequency": f"{avg_frequency:.1f}" if avg_frequency else "0.0",
        "ad_duration": f"{Duration}" if Duration else "0",
        "paid_reach": f"{paid_reach_pct:.1f}%" if paid_reach_pct else "0.0%",
        "owned_reach": f"{owned_reach_pct:.1f}%" if owned_reach_pct else "0.0%",
        "earned_reach": f"{earned_reach_pct:.1f}%" if earned_reach_pct else "0.0%",
        "total_reach": f"{total_reach_pct:.1f}%" if total_reach_pct else "0.0%",
        "cost_by_reach_pct": f"{cost_by_reach_pct:,.2f}" if cost_by_reach_pct else "0.00",
        "multi_channel_grp": f"{multi_channel_grp:.2f}" if multi_channel_grp else "0.00",
        "cost_per_grp": f"{cost_per_grp:,.2f}" if cost_per_grp else "0.00",
        "multi_channel_arp": f"{multi_channel_arp:.2f}" if multi_channel_arp else "0.00",
        "cost_per_arp": f"{cost_per_arp:,.2f}" if cost_per_arp else "0.00",
        "Category": Category or "",
        "Audience": Audience or "",
        "tv_budget": float(tv_budget) if tv_budget else 0,
        "radio_budget": float(radio_budget) if radio_budget else 0,
        "other_budget": float(other_budget) if other_budget else 0,
        "tv_percentage": float(tv_percentage) if tv_percentage else 0,
        "radio_percentage": float(radio_percentage) if radio_percentage else 0,
        "other_percentage": float(other_percentage) if other_percentage else 0,
        "logo_filename": logo_filename,
        "chart_data": json.dumps(chart_data) if chart_data else "[]",
        "tv_reach": float(tv_reach) if tv_reach else 0,
        "radio_reach": float(radio_reach) if radio_reach else 0,
        "other_reach": float(other_reach) if other_reach else 0,
        "tv_exclusive": float(tv_exclusive) if tv_exclusive else 0,
        "radio_exclusive": float(radio_exclusive) if radio_exclusive else 0,
        "other_exclusive": float(other_exclusive) if other_exclusive else 0,
        "tv_cprp": float(tv_cprp) if tv_cprp else 0,
        "radio_cprp": float(radio_cprp) if radio_cprp else 0,
        "other_cprp": float(other_cprp) if other_cprp else 0
    })

@app.post("/predict_batch/", response_class=HTMLResponse)
async def predict_batch(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)

        required = {"Category", "Month_name", "Daypart", "Station", "Normalize 30s", "Duration", "CampaignStart", "CampaignEnd"}
        if not required.issubset(df.columns):
            return templates.TemplateResponse("index.html", {
                "request": request,
                "stations": load_stations(),
                "results": [],
                "message": f"CSV must include: {', '.join(required)}",
                "campaign_trp": "0.00",
                "campaign_grp": "0.00",
                "total_spend": 0,
                "avg_cprp": "0.00",
                "reach_pct": "0.0",
                "campaign_duration_weeks": "0.0",
                "avg_frequency": "0.0",
                "ad_duration": "0",
                "paid_reach": "0.0%",
                "owned_reach": "0.0%",
                "earned_reach": "0.0%",
                "total_reach": "0.0%",
                "cost_by_reach_pct": "0.00",
                "multi_channel_grp": "0.00",
                "cost_per_grp": "0.00",
                "multi_channel_arp": "0.00",
                "cost_per_arp": "0.00",
                "Category": "",
                "Audience": "",
                "logo_filename": None,
                "chart_data": "[]",
                "tv_reach": 0.0,
                "radio_reach": 0.0,
                "other_reach": 0.0,
                "tv_exclusive": 0.0,
                "radio_exclusive": 0.0,
                "other_exclusive": 0.0,
                "tv_cprp": 0.0,
                "radio_cprp": 0.0,
                "other_cprp": 0.0
            })

        if "Spend" in df.columns:
            total_spend = df["Spend"].sum(skipna=True)
            n = len(df)
            df["Spend"] = df["Spend"].fillna(total_spend / n)
        else:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "stations": load_stations(),
                "results": [],
                "message": "CSV must include a 'Spend' column",
                "campaign_trp": "0.00",
                "campaign_grp": "0.00",
                "total_spend": 0,
                "avg_cprp": "0.00",
                "reach_pct": "0.0",
                "campaign_duration_weeks": "0.0",
                "avg_frequency": "0.0",
                "ad_duration": "0",
                "paid_reach": "0.0%",
                "owned_reach": "0.0%",
                "earned_reach": "0.0%",
                "total_reach": "0.0%",
                "cost_by_reach_pct": "0.00",
                "multi_channel_grp": "0.00",
                "cost_per_grp": "0.00",
                "multi_channel_arp": "0.00",
                "cost_per_arp": "0.00",
                "Category": "",
                "Audience": "",
                "logo_filename": None,
                "chart_data": "[]",
                "tv_reach": 0.0,
                "radio_reach": 0.0,
                "other_reach": 0.0,
                "tv_exclusive": 0.0,
                "radio_exclusive": 0.0,
                "other_exclusive": 0.0,
                "tv_cprp": 0.0,
                "radio_cprp": 0.0,
                "other_cprp": 0.0
            })

        df["Medium"] = df["Station"].apply(detect_medium)
        campaign_duration_weeks = calculate_campaign_duration(
            df["CampaignStart"].iloc[0],
            df["CampaignEnd"].iloc[0]
        )

        tv_budget = df[df["Medium"] == "TV"]["Spend"].sum() if not df[df["Medium"] == "TV"].empty else 0
        radio_budget = df[df["Medium"] == "Radio"]["Spend"].sum() if not df[df["Medium"] == "Radio"].empty else 0
        other_budget = df[df["Medium"] == "Other"]["Spend"].sum() if not df[df["Medium"] == "Other"].empty else 0
        
        total_allocated = tv_budget + radio_budget + other_budget
        tv_percentage = (tv_budget / total_allocated * 100) if total_allocated > 0 else 0
        radio_percentage = (radio_budget / total_allocated * 100) if total_allocated > 0 else 0
        other_percentage = (other_budget / total_allocated * 100) if total_allocated > 0 else 0

        df["Station_Spend"] = df["Spend"]
        df["weight"] = df["Station_Spend"] / df["Station_Spend"].sum()

        np.random.seed(42)
        df["Adjusted_Spend"] = df["Station_Spend"] * np.random.uniform(0.9, 1.1, len(df))

        df["log_Spend"] = np.log1p(df["Adjusted_Spend"])
        X1 = df[["log_Spend", "Category", "Month_name", "Daypart", "Station"]]
        df["Predicted_log_TRP"] = stage1_pipeline.predict(X1)
        df["Predicted_TRP"] = np.expm1(df["Predicted_log_TRP"])

        X2 = pd.DataFrame({
            "Predicted_log_TRP": df["Predicted_log_TRP"],
            "Normalize 30s": df["Normalize 30s"],
            "Duration": df["Duration"]
        })
        df["Predicted_log_GRP"] = stage2_pipeline.predict(X2)
        df["Predicted_GRP"] = np.expm1(df["Predicted_log_GRP"])

        df["Reach_Percentage"] = df.apply(lambda x: calculate_reach_from_grp(x["Predicted_GRP"], x["Medium"]), axis=1)
        df["Reach"] = df.apply(lambda x: calculate_reach(x["Predicted_GRP"], x["Medium"]), axis=1)
        df["CPRP"] = df.apply(lambda x: calculate_cprp(x["Station_Spend"], x["Predicted_GRP"]), axis=1)

        campaign_trp = (df["Predicted_TRP"] * df["weight"]).sum()
        campaign_grp = (df["Predicted_GRP"] * df["weight"]).sum()
        avg_cprp = calculate_cprp(total_spend, campaign_grp)
        
        total_reach = df["Reach"].sum()
        avg_medium = df["Medium"].mode()[0] if not df["Medium"].empty else "TV"
        reach_pct = calculate_reach_percentage(campaign_grp, avg_medium)
        avg_frequency = calculate_frequency(campaign_grp, reach_pct)
        avg_ad_duration = df["Duration"].mean()

        medium_budgets = {'TV': tv_budget, 'Radio': radio_budget, 'Other': other_budget}
        medium_reach = calculate_medium_reach_percentages(df, medium_budgets)
        tv_reach = medium_reach.get('TV', 0)
        radio_reach = medium_reach.get('Radio', 0)
        other_reach = medium_reach.get('Other', 0)

        tv_grp = df[df["Medium"] == "TV"]["Predicted_GRP"].sum() if not df[df["Medium"] == "TV"].empty else 0
        radio_grp = df[df["Medium"] == "Radio"]["Predicted_GRP"].sum() if not df[df["Medium"] == "Radio"].empty else 0
        other_grp = df[df["Medium"] == "Other"]["Predicted_GRP"].sum() if not df[df["Medium"] == "Other"].empty else 0

        other_mediums_grp = {'TV': tv_grp, 'Radio': radio_grp, 'Other': other_grp}
        
        tv_exclusive = calculate_exclusive_reach('TV', tv_grp, campaign_grp, other_mediums_grp) if tv_budget > 0 else 0
        radio_exclusive = calculate_exclusive_reach('Radio', radio_grp, campaign_grp, other_mediums_grp) if radio_budget > 0 else 0
        other_exclusive = calculate_exclusive_reach('Other', other_grp, campaign_grp, other_mediums_grp) if other_budget > 0 else 0

        medium_cprp = calculate_medium_cprp(df, medium_budgets)
        tv_cprp = medium_cprp.get('TV', 0)
        radio_cprp = medium_cprp.get('Radio', 0)
        other_cprp = medium_cprp.get('Other', 0)

        paid_reach_pct, owned_reach_pct, earned_reach_pct = calculate_paid_owned_earned_reach(
            reach_pct, avg_medium, total_spend, df["Category"].iloc[0], campaign_duration_weeks
        )
        
        total_reach_pct = paid_reach_pct + owned_reach_pct + earned_reach_pct

        cost_by_reach_pct = calculate_cost_by_reach_percentage(total_spend, reach_pct)
        multi_channel_grp = calculate_multi_channel_grp(campaign_grp, avg_medium, len(df))
        cost_per_grp = calculate_cost_per_grp(total_spend, multi_channel_grp)
        multi_channel_arp = calculate_multi_channel_arp(campaign_grp, avg_frequency, len(df))
        cost_per_arp = calculate_cost_per_arp(total_spend, multi_channel_arp)

        if "Audience" in df.columns:
            audience_value = df["Audience"].iloc[0] if not df.empty else ""
        else:
            audience_value = "General Audience"

        chart_data = df[[
            "Station", "Medium", "Predicted_TRP", "Predicted_GRP", 
            "Reach", "CPRP", "Reach_Percentage"
        ]].rename(columns={
            "Predicted_TRP": "TRP",
            "Predicted_GRP": "GRP"
        }).to_dict(orient="records")

        for record in chart_data:
            record["Frequency"] = avg_frequency

        return templates.TemplateResponse("index.html", {
            "request": request,
            "stations": load_stations(),
            "results": chart_data or [],
            "campaign_trp": f"{campaign_trp:.2f}" if campaign_trp else "0.00",
            "campaign_grp": f"{campaign_grp:.2f}" if campaign_grp else "0.00",
            "total_spend": float(total_spend) if total_spend else 0,
            "avg_cprp": f"{avg_cprp:,.2f}" if avg_cprp else "0.00",
            "reach_pct": f"{reach_pct:.1f}" if reach_pct else "0.0",
            "campaign_duration_weeks": f"{campaign_duration_weeks:.1f}" if campaign_duration_weeks else "0.0",
            "avg_frequency": f"{avg_frequency:.1f}" if avg_frequency else "0.0",
            "ad_duration": f"{avg_ad_duration:.1f}" if avg_ad_duration else "0.0",
            "paid_reach": f"{paid_reach_pct:.1f}%" if paid_reach_pct else "0.0%",
            "owned_reach": f"{owned_reach_pct:.1f}%" if owned_reach_pct else "0.0%",
            "earned_reach": f"{earned_reach_pct:.1f}%" if earned_reach_pct else "0.0%",
            "total_reach": f"{total_reach_pct:.1f}%" if total_reach_pct else "0.0%",
            "cost_by_reach_pct": f"{cost_by_reach_pct:,.2f}" if cost_by_reach_pct else "0.00",
            "multi_channel_grp": f"{multi_channel_grp:.2f}" if multi_channel_grp else "0.00",
            "cost_per_grp": f"{cost_per_grp:,.2f}" if cost_per_grp else "0.00",
            "multi_channel_arp": f"{multi_channel_arp:.2f}" if multi_channel_arp else "0.00",
            "cost_per_arp": f"{cost_per_arp:,.2f}" if cost_per_arp else "0.00",
            "Category": df["Category"].iloc[0] if not df.empty else "",
            "Audience": audience_value,
            "logo_filename": None,
            "chart_data": json.dumps(chart_data) if chart_data else "[]",
            "tv_reach": float(tv_reach) if tv_reach else 0,
            "radio_reach": float(radio_reach) if radio_reach else 0,
            "other_reach": float(other_reach) if other_reach else 0,
            "tv_exclusive": float(tv_exclusive) if tv_exclusive else 0,
            "radio_exclusive": float(radio_exclusive) if radio_exclusive else 0,
            "other_exclusive": float(other_exclusive) if other_exclusive else 0,
            "tv_cprp": float(tv_cprp) if tv_cprp else 0,
            "radio_cprp": float(radio_cprp) if radio_cprp else 0,
            "other_cprp": float(other_cprp) if other_cprp else 0
        })

    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "stations": load_stations(),
            "results": [],
            "message": f"Error processing file: {str(e)}",
            "campaign_trp": "0.00",
            "campaign_grp": "0.00",
            "total_spend": 0,
            "avg_cprp": "0.00",
            "reach_pct": "0.0",
            "campaign_duration_weeks": "0.0",
            "avg_frequency": "0.0",
            "ad_duration": "0",
            "paid_reach": "0.0%",
            "owned_reach": "0.0%",
            "earned_reach": "0.0%",
            "total_reach": "0.0%",
            "cost_by_reach_pct": "0.00",
            "multi_channel_grp": "0.00",
            "cost_per_grp": "0.00",
            "multi_channel_arp": "0.00",
            "cost_per_arp": "0.00",
            "Category": "",
            "Audience": "",
            "logo_filename": None,
            "chart_data": "[]",
            "tv_reach": 0.0,
            "radio_reach": 0.0,
            "other_reach": 0.0,
            "tv_exclusive": 0.0,
            "radio_exclusive": 0.0,
            "other_exclusive": 0.0,
            "tv_cprp": 0.0,
            "radio_cprp": 0.0,
            "other_cprp": 0.0
        })

import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)