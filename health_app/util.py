import joblib
def load_model():
    global model, scaler, label_encoder
    try:
        # Load the trained model and preprocessing objects
        model_objects = joblib.load('trained_model.pkl')
        model = model_objects['model']
        scaler = model_objects['scaler']
        label_encoder = model_objects['label_encoder']
        
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")