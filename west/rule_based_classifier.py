from flask import Blueprint, request, jsonify
import pandas as pd
import os
import logging
from dotenv import load_dotenv
from google import generativeai as genai

# ✅ Load environment variables
load_dotenv()

# ✅ Create a Flask Blueprint for Rule-Based Classification
rbc_bp = Blueprint("rbc", __name__)

# ✅ Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ✅ Get API key from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ✅ Define relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
TEST_DATA_PATH = os.path.join(BASE_DIR, "test_data", "test_df.csv")

# ✅ Load test data with error handling
test_df = None
if os.path.exists(TEST_DATA_PATH):
    try:
        test_df = pd.read_csv(TEST_DATA_PATH, low_memory=False)
        logging.info(f"📂 Test data loaded successfully from {TEST_DATA_PATH}")
    except Exception as e:
        logging.error(f"❌ Error reading test data: {e}")
else:
    logging.warning(f"⚠️ Test data file not found at {TEST_DATA_PATH}. Some features may not work.")

# ✅ Define Water Quality Assessment Criteria
def is_habitable(pH, Iron, Nitrate, Chloride, Lead, Zinc, Turbidity, Fluoride, Copper, Sulfate, Chlorine, Manganese):
    """Check if water is habitable for aquatic life based on given chemical properties."""
    if (
        6.5 <= pH <= 9.0 and Iron < 0.3 and Nitrate < 10 and Chloride < 250 and Lead < 0.015 and 
        Zinc < 5 and Turbidity < 5 and 0.7 <= Fluoride <= 1.5 and Copper < 1.3 and Sulfate < 250 and 
        Chlorine < 4.0 and Manganese < 0.05 
    ):
        return 0  # ✅ Habitable
    else:
        return 1  # ❌ Not habitable

# ✅ Generate Gemini Suggestions
def generate_water_suggestions(input_data):
    """Generates suggestions for improving water quality using Gemini."""
    if not GEMINI_API_KEY:
        logging.warning("⚠️ GOOGLE_API_KEY environment variable not found.")
        return "Google API key not configured."

    logging.info("Attempting to generate water quality improvement suggestions with Gemini...")
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        prompt = f"""
        The following water quality parameters were analyzed:
        {input_data}
        Provide actionable steps to improve the water quality for aquatic life if any parameter is out of the optimal range.
        Only give the response, without formatting or stating that you are an AI. NO FORMATTING OF ANY KIND
        """

        response = model.generate_content(prompt)
        logging.info("✅ Gemini content generation successful.")
        return response.text
    except Exception as e:
        logging.error(f"❌ Error generating content with Gemini: {e}")
        return "Error generating water quality improvement suggestions."

# ✅ Define Features for Water Quality Assessment
features = {
    'pH': float, 'Iron': float, 'Nitrate': float, 'Chloride': float, 'Lead': float, 'Zinc': float,
    'Turbidity': float, 'Fluoride': float, 'Copper': float, 'Sulfate': float, 'Chlorine': float,
    'Manganese': float
}

@rbc_bp.before_request
def check_test_data():
    """Ensure test data is available before requests."""
    if test_df is None:
        return jsonify({"error": "Test data not available"}), 500

@rbc_bp.route('/assess_water_quality', methods=['POST'])
def assess_water_quality():
    """API endpoint to assess water quality for aquatic life."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        # ✅ Standardize field names
        standardized_data = {}
        for feature, dtype in features.items():
            value = data.get(feature)
            if value is None:
                return jsonify({"error": f"Missing value for {feature}"}), 400
            try:
                standardized_data[feature] = dtype(value)
            except ValueError:
                return jsonify({"error": f"Invalid value for {feature}. Expected {dtype.__name__}"}), 400

        # ✅ Ensure function receives correct argument names
        is_good = is_habitable(**standardized_data)
        message = "✅ Water quality is habitable for aquatic life." if is_good == 0 else "❌ Water quality is not habitable for aquatic life."
        
        # ✅ Generate improvement suggestions if water is not habitable
        improvement_suggestions = generate_water_suggestions(standardized_data) if is_good == 1 else "No improvements needed."
        
        return jsonify({"input_data": standardized_data, "aquatic_habitability": is_good, "message": message, "improvement_suggestions": improvement_suggestions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@rbc_bp.route('/random_assessment', methods=['GET'])
def random_assessment():
    """Picks a random row from test data and assesses water quality."""
    try:
        # Ensure required columns exist
        available_columns = [col for col in test_df.columns if col in features]
        if not available_columns:
            return jsonify({"error": "Required columns not found in test data"}), 500

        data = test_df.sample(n=1)[available_columns]
        processed_data = data.to_dict(orient='records')[0]

        # Assess water quality
        is_good = is_habitable(**processed_data)
        message = "✅ Water quality is habitable for aquatic life." if is_good == 0 else "❌ Water quality is not habitable for aquatic life."
        
        # ✅ Generate improvement suggestions if water is not habitable
        improvement_suggestions = generate_water_suggestions(processed_data) if is_good == 1 else "No improvements needed."
        
        return jsonify({"input_data": processed_data, "aquatic_habitability": is_good, "message": message, "improvement_suggestions": improvement_suggestions})
    except Exception as e:
        logging.error(f"❌ Error in random_assessment: {e}")
        return jsonify({"error": str(e)}), 500