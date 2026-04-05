import joblib

# Load model
model = joblib.load('models/Baseline_RandomForest_final.joblib')

def predict(input_data):
    prediction = model.predict(input_data)
    return prediction
