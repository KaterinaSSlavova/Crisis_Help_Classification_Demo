import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret")

    DB_PATH = os.getenv("DB_PATH", "instance/demo.sqlite3")

    MODEL_DIR = os.getenv("MODEL_DIR", "model_artifacts/distilbert_crisis_final")
    THRESHOLDS_PATH = os.getenv("THRESHOLDS_PATH", "model_artifacts/best_thresholds.npy")

    MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "5000"))
    TOP_TOKENS = int(os.getenv("TOP_TOKENS", "12"))