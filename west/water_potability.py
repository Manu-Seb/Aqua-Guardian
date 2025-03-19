from flask import Blueprint, request, jsonify
import os
import logging
import pandas as pd
import numpy as np
from pycaret.classification import load_model, predict_model
from dotenv import load_dotenv
from google import generativeai as genai

load_dotenv()
water_potability_bp = Blueprint('water_potability', __name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "Water_Potability", "xgboost_without_source_month")
try:
    model = load_model(MODEL_PATH)
    logging.info("🚀 Water Potability Model loaded successfully!")
except Exception as e:
    logging.error(f"❌ Error loading water quality model: {e}")
    raise

def generate_water_suggestions(input_data):
    if not GEMINI_API_KEY:
        logging.warning("⚠️ GOOGLE_API_KEY environment variable not found.")
        return "Google API key not configured."
    logging.info("Attempting to generate water quality improvement suggestions with Gemini...")
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"The following water quality parameters were analyzed:\n{input_data}\nProvide actionable steps to improve the water quality if any parameter is out of the optimal range.\nOnly give the response, without formatting or stating that you are an AI. NO FORMATTING OF ANY KIND"
        response = model.generate_content(prompt)
        logging.info("✅ Gemini content generation successful.")
        return response.text
    except Exception as e:
        logging.error(f"❌ Error generating content with Gemini: {e}")
        return "Error generating water quality improvement suggestions."

features = {
    'pH': float, 'Iron': float, 'Nitrate': float, 'Chloride': float, 'Lead': float, 'Zinc': float,
    'Turbidity': float, 'Fluoride': float, 'Copper': float, 'Sulfate': float, 'Chlorine': float,
    'Manganese': float, 'Color': str, 'Odor': float, 'Total Dissolved Solids': float
}

@water_potability_bp.route('/water_potability', methods=['POST'])
def test_potability():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400
        logging.info(f"📩 Received data: {data}")

        logging.info("🔍 Running updated water_potability.py with defaults")

        standardized_data = {}
        for feature, dtype in features.items():
            if feature in data:
                try:
                    standardized_data[feature] = dtype(data[feature])
                except ValueError:
                    return jsonify({"error": f"Invalid value for {feature}. Expected {dtype.__name__}"}), 400
            else:
                if feature == 'Color':
                    standardized_data[feature] = 'Colorless'
                elif feature == 'Odor':
                    standardized_data[feature] = 0.0
                elif feature == 'Total Dissolved Solids':
                    standardized_data[feature] = 500.0
                else:
                    standardized_data[feature] = 0.0

        df = pd.DataFrame([standardized_data])[list(features.keys())]
        logging.info(f"📝 Processed Data Columns: {df.columns.tolist()}")
        logging.info(f"📝 Processed Data Sample: {df.iloc[0].to_dict()}")

        prediction = predict_model(model, data=df)
        prediction_label = int(prediction['prediction_label'][0])
        logging.info(f"💧 Water Potability Prediction: {prediction_label}")

        improvement_suggestions = generate_water_suggestions(standardized_data) if prediction_label == 1 else "No improvements needed."
        
        response = {
            "input_data": standardized_data,
            "potability": "Safe for drinking" if prediction_label == 0 else "Not safe for drinking",
            "improvement_suggestions": improvement_suggestions
        }
        logging.info("✅ Test potability completed, sending response.")
        return jsonify(response)
    except Exception as e:
        logging.error(f"❌ Error in test_potability: {type(e).__name__}: {str(e)}")
        return jsonify({"error": f"{type(e).__name__}: {str(e)}"}), 500
    finally:
        logging.info("🏁 Test potability endpoint execution finished.")
        import time
        time.sleep(1)
        logging.info("⏳ Still alive after response!")

@water_potability_bp.route('/random_predict', methods=['GET'])
def random_predict():
    try:
        test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data", "test_df.csv")
        if not os.path.exists(test_data_path):
            return jsonify({"error": "Test data file not found"}), 500

        test_df = pd.read_csv(test_data_path)
        logging.info(f"📋 Test Data Columns: {test_df.columns.tolist()}")
        if test_df.empty:
            return jsonify({"error": "Test data CSV is empty"}), 500

        sample = test_df.sample(n=1)
        processed_data = sample.to_dict(orient='records')[0]
        logging.info(f"🎲 Random Sample Data: {processed_data}")

        # Drop 'Target' if present, as model shouldn't predict on it
        sample_for_prediction = sample.drop(columns=['Target'], errors='ignore')
        logging.info(f"🔍 Sample for Prediction: {sample_for_prediction.to_dict(orient='records')[0]}")

        prediction = predict_model(model, data=sample_for_prediction)
        logging.info(f"🔬 Prediction DataFrame Columns: {prediction.columns.tolist()}")
        logging.info(f"🔬 Prediction DataFrame Sample: {prediction.to_dict(orient='records')[0]}")
        if 'prediction_label' not in prediction.columns:
            return jsonify({"error": "Prediction missing 'prediction_label' column"}), 500
        
        prediction_label = int(prediction['prediction_label'].iloc[0])
        logging.info(f"🔍 Random Water Potability Prediction: {prediction_label}")

        improvement_suggestions = generate_water_suggestions(processed_data) if prediction_label == 1 else "No improvements needed."
        
        response = {
            "input_data": processed_data,
            "potability": "Safe for drinking" if prediction_label == 0 else "Not safe for drinking",
            "improvement_suggestions": improvement_suggestions
        }
        logging.info("✅ Random prediction completed, sending response.")
        return jsonify(response)
    except Exception as e:
        logging.error(f"❌ Error during random water potability prediction: {type(e).__name__}: {str(e)}")
        return jsonify({"error": f"{type(e).__name__}: {str(e)}"}), 500
    finally:
        logging.info("🏁 Random predict endpoint execution finished.")
        import time
        time.sleep(1)
        logging.info("⏳ Still alive after response!")