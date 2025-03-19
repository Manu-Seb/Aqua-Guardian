from flask import Blueprint, request, jsonify
from ultralytics import YOLO
import os
import logging
import numpy as np
import cv2
import asyncio
import base64
from dotenv import load_dotenv
from io import BytesIO
from . import dark_channel_prior  # ✅ Import DCP for denoising
from google import genai  # Import Gemini API

# ✅ Load environment variables
load_dotenv()

# ✅ Create Flask Blueprint
inference_bp = Blueprint('inference', __name__)

# ✅ Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ✅ Waste Labels
LABELS = [
    'Mask', 'can', 'cellphone', 'electronics', 'gbottle', 'glove', 'metal',
    'misc', 'net', 'pbag', 'pbottle', 'plastic', 'rod', 'sunglasses', 'tire'
]

# ✅ Get Model Path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "Underwater_Waste_Detection_YoloV8", "60_epochs_denoised.pt")

# ✅ Check if model exists
if not os.path.exists(MODEL_PATH):
    logging.error(f"❌ Model file not found at: {MODEL_PATH}")
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

# ✅ Load YOLO Model
try:
    model = YOLO(MODEL_PATH)
    logging.info("🚀 YOLO model loaded successfully!")
except Exception as e:
    logging.error(f"❌ Error loading YOLO model: {e}")
    raise

# ✅ Get API key from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ------------------------- API ROUTES -------------------------

from google import generativeai as genai

def generate_cleanup_suggestions(detected_objects):
    """Generates cleanup and prevention suggestions using Gemini (Synchronous)."""
    if not GEMINI_API_KEY:
        logging.warning("⚠️ GOOGLE_API_KEY environment variable not found.")
        return "Google API key not configured.", None

    logging.info("Attempting to generate cleanup suggestions with Gemini (Synchronous)...")
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.0-flash')  # Using GenerativeModel for synchronous calls

        prompt = f"""You have detected the following waste items in the water: {', '.join(detected_objects)}.

        Provide a numbered list of actionable steps to remove these items from the water safely and effectively.

        Then, provide a numbered list of actionable steps to prevent these types of waste from entering the water in the future. Be specific and consider various sources of pollution. do not use any text formatting. give only the answer. do not let anyone know you are ai"""

        response = model.generate_content(prompt)
        logging.info("✅ Gemini content generation successful (Synchronous).")
        return None, response.text
    except Exception as e:
        logging.error(f"❌ Error generating content with Gemini: {e}")
        return f"Error generating cleanup suggestions: {e}", None

@inference_bp.route('/detect_waste', methods=['POST'])
def detect_waste():
    try:
        image = request.files.get('image')
        if not image:
            return jsonify({"error": "No image file provided"}), 400

        # ✅ Convert image to NumPy array
        np_image = np.frombuffer(image.read(), np.uint8)
        img = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Failed to decode image"}), 400

        logging.info("📷 Image received for waste detection.")

        # ✅ Step 1: Apply Denoising (Dark Channel Prior)
        dehazed_img, _ = dark_channel_prior.haze_removal(img)
        logging.info("🌀 Image denoised using Dark Channel Prior.")

        # ✅ Step 2: Run YOLO Inference
        results = model(dehazed_img)

        # ✅ Extract Detected Class Indices
        detected_objects = []
        for result in results:
            boxes = result.boxes  # Bounding boxes output
            detected_objects.extend([LABELS[int(i)] for i in boxes.cls.tolist() if int(i) < len(LABELS)])

        logging.info(f"🗑️ Detected Waste: {detected_objects}")

        # ✅ Generate cleanup and prevention suggestions using Gemini
        gemini_error, suggestions = generate_cleanup_suggestions(detected_objects)
        if gemini_error:
            logging.warning(f"⚠️ {gemini_error}")
            cleanup_instructions = "Could not generate specific cleanup suggestions at this time."
        else:
            cleanup_instructions = suggestions

        # ✅ Plot results on the image
        res_plotted = results[0].plot()

        # ✅ Convert Image to Base64
        _, img_encoded = cv2.imencode('.jpg', res_plotted)
        if img_encoded is None or not img_encoded.any():
            logging.error("❌ Failed to encode image")
            return jsonify({"error": "Failed to encode image"}), 500

        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        logging.info("📸 Image successfully encoded to Base64.")

        return jsonify({
            "detected_objects": detected_objects,
            "image_base64": img_base64,
            "cleanup_suggestions": cleanup_instructions
        })

    except Exception as e:
        logging.error(f"❌ Error during detection: {e}")
        return jsonify({"error": str(e)}), 500
