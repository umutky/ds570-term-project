from retail_forecast import config

def train():
    print("hello from rf-train! Pipeline is starting...")
    print(f"Models will be saved to: {config.MODELS_DIR}")

def predict():
    print("hello from rf-predict! Making predictions...")

def process_data():
    print("hello from rf-process! Processing raw data...")