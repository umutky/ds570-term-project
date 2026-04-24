from pathlib import Path

# Proje ana dizinini dinamik olarak bulur
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Veri ve Çıktı yolları
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"