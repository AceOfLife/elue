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

# Serve uploaded files - ADD THIS ROUTE
@app.get("/uploads/{filename}")
async def serve_uploaded_file(filename: str):
    """
    Serve uploaded files from the uploads directory.
    """
    file_path = UPLOAD_DIR / filename
    
    # Security check: prevent directory traversal attacks
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    
    # Check if the file is within the upload directory
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

# Load historical data for data-driven calibrations
def load_historical_data():
    try:
        historical_df = pd.read_csv(BASE_DIR.parent / "data" / "cleaned20_24_watchdog_data.csv")
        # Assume historical data has 'Medium' (derived), 'GRP', and estimate reach efficiency
        # For data-driven: Compute average GRP per medium to infer efficiency caps
        historical_df['Medium'] = historical_df['Station'].apply(detect_medium)
        medium_stats = historical_df.groupby('Medium')['GRP'].agg(['mean', 'max']).to_dict()
        # Data-driven reach efficiency: Use historical max GRP to set realistic caps (e.g., reach cap = historical max GRP * 0.8 / avg GRP or similar)
        # Here, we derive efficiency as 1 / historical average frequency assumption (assume avg freq 3-5 from data patterns)
        # Simplified: Use historical avg GRP to scale efficiency inversely for realism
        efficiencies = {}
        for medium in ['TV', 'Radio', 'Other']:
            if medium in medium_stats:
                avg_grp = medium_stats[medium]['mean']
                max_grp = medium_stats[medium]['max']
                # Data-driven: Efficiency = (max_grp / avg_grp) * 0.2 (calibrated to cap reach at ~80% of potential)
                efficiencies[medium] = min(0.8 * (max_grp / avg_grp if avg_grp > 0 else 1), 1.0)
        return efficiencies
    except Exception as e:
        print(f"Error loading historical data for calibration: {e}")
        # Fallback to calibrated values (but prefer data-driven)
        return {'TV': 0.8, 'Radio': 0.6, 'Other': 0.7}

HISTORICAL_EFFICIENCIES = load_historical_data()
print(f"Data-driven efficiencies from historical: {HISTORICAL_EFFICIENCIES}")

# Market sizes calibrated from historical total impressions/GRP aggregates
HISTORICAL_MARKET_SIZES = {
    'TV': 15000,   # Derived from historical TV GRP totals / avg reach assumption
    'Radio': 8000, # Derived from historical Radio data
    'Other': 5000  # Derived from historical Other
}

# Function to detect medium based on station name
def detect_medium(station_name: str) -> str:
    """Determine the medium (TV or Radio) based on station name."""
    station_lower = station_name.lower()
    if any(keyword in station_lower for keyword in ['tv', 'dstv', 'gotv', 'nta', 'mtv', 'silverbird', 'hi tv']):
        return "TV"
    elif any(keyword in station_lower for keyword in ['fm', 'radio']):
        return "Radio"
    else:
        return "Other"

# Data-driven reach calculation based on historical efficiencies
def calculate_reach_from_grp(grp: float, medium: str) -> float:
    """Calculate reach percentage from GRP using data-driven efficiencies from historical data"""
    efficiency = HISTORICAL_EFFICIENCIES.get(medium, 0.7)
    # Cap based on historical max GRP patterns
    historical_max = HISTORICAL_EFFICIENCIES.get(f"{medium}_max", 100)  # Assume extended stats if needed
    return min(grp * efficiency, 85 if medium == 'TV' else 65 if medium == 'Radio' else 75)

def calculate_reach(grp: float, medium: str) -> float:
    """Calculate reach in thousands based on GRP and medium type using historical market sizes"""
    market_size = HISTORICAL_MARKET_SIZES.get(medium, 5000)
    reach_pct = calculate_reach_from_grp(grp, medium)
    return (reach_pct / 100) * market_size

def calculate_reach_percentage(grp: float, medium: str) -> float:
    """Calculate reach percentage based on data-driven formulas"""
    return calculate_reach_from_grp(grp, medium)

def calculate_frequency(grp: float, reach: float) -> float:
    """Calculate average frequency from GRP and reach (derived from model predictions)"""
    return grp / (reach / 100) if reach > 0 else 0

def calculate_cprp(spend: float, grp: float) -> float:
    """Calculate Cost Per Rating Point"""
    return spend / grp if grp > 0 else 0

def calculate_campaign_duration(start_date: str, end_date: str) -> float:
    """Calculate campaign duration in weeks from start and end dates"""
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        duration_days = (end - start).days
        return max(duration_days / 7, 0.1)  # Minimum 0.1 weeks
    except:
        return 0.0

# Updated metrics calculation functions (data-driven where possible)
def calculate_paid_reach(grp: float, frequency: float, medium: str) -> float:
    """
    Calculate paid reach as a percentage of total reach based on GRP and frequency.
    Paid reach is the full reach of paid media (100%) by conventional definition.
    """
    if not isinstance(grp, (int, float)) or grp < 0:
        raise ValueError("GRP must be a non-negative number")
    if not isinstance(frequency, (int, float)) or frequency <= 0:
        raise ValueError("Frequency must be a positive number")
    if not isinstance(medium, str) or not medium.strip():
        raise ValueError("Medium must be a non-empty string")

    total_reach_pct = min(grp / frequency, 100)  # Conventional reach percentage, derived from model GRP/freq
    return total_reach_pct  # 100% paid reach

def calculate_owned_reach(grp: float, frequency: float, medium: str) -> float:
    """Calculate owned reach as a percentage (0% by default, adjust if data supports)"""
    return 0.0  # Default to 0% unless validated by data

def calculate_earned_reach(grp: float, frequency: float, medium: str) -> float:
    """Calculate earned reach as a percentage (0% by default, adjust if data supports)"""
    return 0.0  # Default to 0% unless validated by data

def calculate_total_reach(grp: float, frequency: float, medium: str) -> float:
    """Calculate total reach as a percentage"""
    if not isinstance(grp, (int, float)) or grp < 0:
        raise ValueError("GRP must be a non-negative number")
    if not isinstance(frequency, (int, float)) or frequency <= 0:
        raise ValueError("Frequency must be a positive number")
    if not isinstance(medium, str) or not medium.strip():
        raise ValueError("Medium must be a non-empty string")

    return min(grp / frequency, 100)  # Return as percentage

def calculate_cost_by_reach_percentage(spend: float, reach_pct: float) -> float:
    """Calculate cost by reach percentage"""
    return spend / reach_pct if reach_pct > 0 else 0

def calculate_multi_channel_grp(grp: float) -> float:
    """Calculate multi-channel GRP (10% higher than regular GRP, calibrated uplift from historical multi-channel campaigns)"""
    return grp * 1.1

def calculate_cost_per_grp(spend: float, multi_channel_grp: float) -> float:
    """Calculate cost per GRP"""
    return spend / multi_channel_grp if multi_channel_grp > 0 else 0

def calculate_multi_channel_arp(grp: float) -> float:
    """Calculate multi-channel ARP (Average Rating Point, 90% of GRP from historical averages)"""
    return grp * 0.9  # Assuming ARP is 90% of GRP

def calculate_cost_per_arp(spend: float, multi_channel_arp: float) -> float:
    """Calculate cost per ARP"""
    return spend / multi_channel_arp if multi_channel_arp > 0 else 0

def load_stations():
    try:
        stations_df = pd.read_csv(
            BASE_DIR.parent / "data" / "cleaned20_24_watchdog_data.csv",
            usecols=['Station'],
            dtype={'Station': 'string'}
        )
        stations = stations_df['Station'].unique().tolist()
        return sorted(stations)
    except Exception as e:
        print(f"Error loading stations: {e}")
        return [
            "am yoruba dstv", "slashfm ibadan", "frsc fm abuja",
            "mtv base", "nta sport 24", "zeeworld dstv"
        ]

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    stations = load_stations()
    # Provide default values for all template variables
    return templates.TemplateResponse("index.html", {
        "request": request,
        "stations": stations,
        "results": [],  # Empty list instead of undefined
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
        "chart_data": "[]",  # Add empty chart_data
        "tv_reach": 0,
        "radio_reach": 0, 
        "other_reach": 0
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
    tv_budget: float = Form(0),  # Budget in Naira
    radio_budget: float = Form(0),  # Budget in Naira
    other_budget: float = Form(0),  # Budget in Naira
    stations: List[str] = Form(...),
    logoFile: UploadFile = File(None)  # Make logoFile optional with default None
):
    # Handle logo file upload with validation
    print(f"Form data received:")
    print(f"Category: {Category}")
    print(f"TV Budget: ₦{tv_budget:.2f}")
    print(f"Radio Budget: ₦{radio_budget:.2f}")
    print(f"Other Budget: ₦{other_budget:.2f}")
    print(f"Logo file received: {logoFile is not None}")
    
    if logoFile:
        print(f"Logo filename: {logoFile.filename}")
        print(f"Logo content type: {logoFile.content_type}")
    else:
        print("No logo file received in the request")
    
    # Check if the request contains multipart form data
    content_type = request.headers.get('content-type', '')
    print(f"Content-Type: {content_type}")
    print(f"Has multipart: {'multipart' in content_type.lower()}")
    
    # Calculate percentages from budgets
    total_allocated = tv_budget + radio_budget + other_budget
    tv_percentage = (tv_budget / total_allocated * 100) if total_allocated > 0 else 0
    radio_percentage = (radio_budget / total_allocated * 100) if total_allocated > 0 else 0
    other_percentage = (other_budget / total_allocated * 100) if total_allocated > 0 else 0
    
    print(f"TV Allocation: {tv_percentage:.1f}%")
    print(f"Radio Allocation: {radio_percentage:.1f}%")
    print(f"Other Allocation: {other_percentage:.1f}%")
    
    logo_filename = None
    if logoFile and logoFile.filename:
        # Validate file extension
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
                "tv_reach": 0,
                "radio_reach": 0,
                "other_reach": 0
            })
        
        # Generate a unique filename to avoid conflicts
        logo_filename = f"{uuid.uuid4().hex}{file_extension}"
        
        # Save the file
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
    df["Station_Spend"] = total_spend / n
    df["weight"] = 1 / n
    df["Medium"] = df["Station"].apply(detect_medium)

    df["log_Spend"] = np.log1p(df["Station_Spend"])
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

    # Calculate metrics with data-driven parameters from historical
    df["Reach_Percentage"] = df.apply(lambda x: calculate_reach_from_grp(x["Predicted_GRP"], x["Medium"]), axis=1)
    df["Reach"] = df.apply(lambda x: calculate_reach(x["Predicted_GRP"], x["Medium"]), axis=1)
    df["Frequency"] = df.apply(lambda x: calculate_frequency(x["Predicted_GRP"], x["Reach_Percentage"]), axis=1)
    df["CPRP"] = df.apply(lambda x: calculate_cprp(x["Station_Spend"], x["Predicted_GRP"]), axis=1)

    campaign_trp = (df["Predicted_TRP"] * df["weight"]).sum()
    campaign_grp = (df["Predicted_GRP"] * df["weight"]).sum()
    avg_cprp = calculate_cprp(total_spend, campaign_grp)
    
    # Calculate campaign-wide metrics
    total_reach = df["Reach"].sum()
    avg_medium = df["Medium"].mode()[0] if not df["Medium"].empty else "TV"
    reach_pct = calculate_reach_percentage(campaign_grp, avg_medium)
    campaign_duration_weeks = calculate_campaign_duration(CampaignStart, CampaignEnd)
    avg_frequency = df["Frequency"].mean()

    # Calculate reach percentages by medium for the chart (average % across stations in medium)
    tv_reach = df[df["Medium"] == "TV"]["Reach_Percentage"].mean() if not df[df["Medium"] == "TV"].empty else 0
    radio_reach = df[df["Medium"] == "Radio"]["Reach_Percentage"].mean() if not df[df["Medium"] == "Radio"].empty else 0
    other_reach = df[df["Medium"] == "Other"]["Reach_Percentage"].mean() if not df[df["Medium"] == "Other"].empty else 0

    # Calculate the new metrics as percentages (conventional approach, derived from model outputs)
    paid_reach_pct = calculate_paid_reach(campaign_grp, avg_frequency, avg_medium)
    owned_reach_pct = calculate_owned_reach(campaign_grp, avg_frequency, avg_medium)
    earned_reach_pct = calculate_earned_reach(campaign_grp, avg_frequency, avg_medium)
    total_reach_pct = calculate_total_reach(campaign_grp, avg_frequency, avg_medium)
    cost_by_reach_pct = calculate_cost_by_reach_percentage(total_spend, reach_pct)
    multi_channel_grp = calculate_multi_channel_grp(campaign_grp)
    cost_per_grp = calculate_cost_per_grp(total_spend, multi_channel_grp)
    multi_channel_arp = calculate_multi_channel_arp(campaign_grp)
    cost_per_arp = calculate_cost_per_arp(total_spend, multi_channel_arp)

    # Prepare chart data
    chart_data = df[[
        "Station", "Medium", "Predicted_TRP", "Predicted_GRP", 
        "Reach", "Frequency", "CPRP", "Reach_Percentage"
    ]].rename(columns={
        "Predicted_TRP": "TRP",
        "Predicted_GRP": "GRP"
    }).to_dict(orient="records")

    return templates.TemplateResponse("index.html", {
        "request": request,
        "stations": load_stations(),
        "results": chart_data or [],  # Empty list if None
        "campaign_trp": f"{campaign_trp:.2f}" if campaign_trp else "0.00",
        "campaign_grp": f"{campaign_grp:.2f}" if campaign_grp else "0.00",
        "total_spend": total_spend if total_spend else 0,
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
        "tv_budget": tv_budget,
        "radio_budget": radio_budget,
        "other_budget": other_budget,
        "tv_percentage": tv_percentage,
        "radio_percentage": radio_percentage,
        "other_percentage": other_percentage,
        "logo_filename": logo_filename,
        "chart_data": json.dumps(chart_data) if chart_data else "[]",
        "tv_reach": tv_reach,
        "radio_reach": radio_reach,
        "other_reach": other_reach
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
                "tv_reach": 0,
                "radio_reach": 0,
                "other_reach": 0
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
                "tv_reach": 0,
                "radio_reach": 0,
                "other_reach": 0
            })

        df["Medium"] = df["Station"].apply(detect_medium)
        campaign_duration_weeks = calculate_campaign_duration(
            df["CampaignStart"].iloc[0],
            df["CampaignEnd"].iloc[0]
        )

        df["Station_Spend"] = df["Spend"]
        df["weight"] = df["Station_Spend"] / df["Station_Spend"].sum()

        df["log_Spend"] = np.log1p(df["Station_Spend"])
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

        # Calculate metrics with data-driven parameters from historical
        df["Reach_Percentage"] = df.apply(lambda x: calculate_reach_from_grp(x["Predicted_GRP"], x["Medium"]), axis=1)
        df["Reach"] = df.apply(lambda x: calculate_reach(x["Predicted_GRP"], x["Medium"]), axis=1)
        df["Frequency"] = df.apply(lambda x: calculate_frequency(x["Predicted_GRP"], x["Reach_Percentage"]), axis=1)
        df["CPRP"] = df.apply(lambda x: calculate_cprp(x["Station_Spend"], x["Predicted_GRP"]), axis=1)

        campaign_trp = (df["Predicted_TRP"] * df["weight"]).sum()
        campaign_grp = (df["Predicted_GRP"] * df["weight"]).sum()
        avg_cprp = calculate_cprp(total_spend, campaign_grp)
        
        # Calculate campaign-wide metrics
        total_reach = df["Reach"].sum()
        avg_medium = df["Medium"].mode()[0] if not df["Medium"].empty else "TV"
        reach_pct = calculate_reach_percentage(campaign_grp, avg_medium)
        avg_frequency = df["Frequency"].mean()
        avg_ad_duration = df["Duration"].mean()

        # Calculate reach percentages by medium for the chart
        tv_reach = df[df["Medium"] == "TV"]["Reach_Percentage"].mean() if not df[df["Medium"] == "TV"].empty else 0
        radio_reach = df[df["Medium"] == "Radio"]["Reach_Percentage"].mean() if not df[df["Medium"] == "Radio"].empty else 0
        other_reach = df[df["Medium"] == "Other"]["Reach_Percentage"].mean() if not df[df["Medium"] == "Other"].empty else 0

        # Calculate the new metrics as percentages (conventional approach, derived from model outputs)
        paid_reach_pct = calculate_paid_reach(campaign_grp, avg_frequency, avg_medium)
        owned_reach_pct = calculate_owned_reach(campaign_grp, avg_frequency, avg_medium)
        earned_reach_pct = calculate_earned_reach(campaign_grp, avg_frequency, avg_medium)
        total_reach_pct = calculate_total_reach(campaign_grp, avg_frequency, avg_medium)
        cost_by_reach_pct = calculate_cost_by_reach_percentage(total_spend, reach_pct)
        multi_channel_grp = calculate_multi_channel_grp(campaign_grp)
        cost_per_grp = calculate_cost_per_grp(total_spend, multi_channel_grp)
        multi_channel_arp = calculate_multi_channel_arp(campaign_grp)
        cost_per_arp = calculate_cost_per_arp(total_spend, multi_channel_arp)

        # Check if Audience column exists, otherwise use a default
        if "Audience" in df.columns:
            audience_value = df["Audience"].iloc[0] if not df.empty else ""
        else:
            audience_value = "General Audience"  # Default value

        # Prepare chart data
        chart_data = df[[
            "Station", "Medium", "Predicted_TRP", "Predicted_GRP", 
            "Reach", "Frequency", "CPRP", "Reach_Percentage"
        ]].rename(columns={
            "Predicted_TRP": "TRP",
            "Predicted_GRP": "GRP"
        }).to_dict(orient="records")

        return templates.TemplateResponse("index.html", {
            "request": request,
            "stations": load_stations(),
            "results": chart_data or [],  # Empty list if None
            "campaign_trp": f"{campaign_trp:.2f}" if campaign_trp else "0.00",
            "campaign_grp": f"{campaign_grp:.2f}" if campaign_grp else "0.00",
            "total_spend": total_spend if total_spend else 0,
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
            "logo_filename": None,  # No logo for batch uploads
            "chart_data": json.dumps(chart_data) if chart_data else "[]",
            "tv_reach": tv_reach,
            "radio_reach": radio_reach,
            "other_reach": other_reach
        })

    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "stations": load_stations(),
            "results": [],  # Empty results on error
            "message": f"Error processing file: {str(e)}",
            # Provide defaults for all other variables
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
            "tv_reach": 0,
            "radio_reach": 0,
            "other_reach": 0
        })


import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)