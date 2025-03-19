from flask import Flask, render_template
from west.water_potability import water_potability_bp  # ✅ Import Blueprint
from west.inference import inference_bp  # ✅ Import Blueprint
from west.rule_based_classifier import rbc_bp  # ✅ Import Blueprint
from dotenv import load_dotenv
import os
from waitress import serve
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

# ✅ Register Blueprints (API routes)
app.register_blueprint(water_potability_bp)
app.register_blueprint(inference_bp)
app.register_blueprint(rbc_bp)

# 🚀 UI Page Routes (ONLY Forwarding)
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/waste_detection')
def waste_detection_page():
    return render_template("waste_detection.html")

@app.route('/water_quality')
def water_quality_page():
    return render_template("water_quality.html")

@app.route('/water_potability')
def potability_test_page():
    return render_template("water_potability.html")


if __name__ == "__main__":
    logging.info("🌐 Starting Flask app with Waitress...")
    serve(app, host="0.0.0.0", port=5000)
    logging.info("🌐 Flask app stopped.")