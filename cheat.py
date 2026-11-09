import os

from dotenv import load_dotenv
from roboflow import Roboflow
from ultralytics import YOLO

load_dotenv()

api_key = os.getenv("ROBOFLOW_API_KEY")
workspace = os.getenv("ROBOFLOW_WORKSPACE")
project_name = os.getenv("ROBOFLOW_PROJECT")
if not (api_key and workspace and project_name):
    raise SystemExit(
        "Set ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE and ROBOFLOW_PROJECT "
        "(see .env.example)."
    )

# 1. Download your annotated dataset (Free)
rf = Roboflow(api_key=api_key)
project = rf.workspace(workspace).project(project_name)
dataset = project.version(1).download("yolov8")

# 2. Train locally using Ultralytics
model = YOLO("yolo26s.pt")  # Pretrained YOLO26 Medium
results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=35,            # Lower epoch count; transfer learning converges quickly
    imgsz=960,            # High enough for small volleyballs, ~40% faster than 1280
    device="mps",         # Forces Apple Silicon Metal GPU acceleration
    batch=16,             # Keeps GPU saturated without running out of unified memory
    cache=True,           # Pre-loads dataset into RAM to stop disk reading delays
    patience=8,           # Early stopping: halts automatically when accuracy peaks
    workers=4,            # Optimizes CPU thread handling for Mac
)