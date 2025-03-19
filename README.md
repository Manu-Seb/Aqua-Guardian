# Aqua Guardian

## Overview

Aqua Guardian is a Flask-deployed Machine Learning (ML) application with Large Language Model (LLM) integration. It is designed to analyze water quality by leveraging multiple ML models and providing intelligent insights through Google's Gemini AI.

## Features

### 🔹 Waste Detection in Water (YOLOv8)

- Utilizes YOLOv8 for object detection to identify waste in water bodies.
- Provides insights on detected waste and suggests removal techniques via Gemini AI.

### 🔹 Rule-Based Water Quality Classification

- Implements a rule-based classifier to assess water quality based on predefined thresholds.
- Offers recommendations on improving water quality using Gemini AI.

### 🔹 Water Potability Prediction (Stacked Ensemble)

- Uses a stacked ensemble model to predict whether water is potable.
- Suggests methods to enhance drinkability via Gemini AI.

## Tech Stack

- **Backend**: Flask
- **Machine Learning**: YOLOv8, Rule-Based Classification, Stacked Ensemble Model
- **LLM Integration**: Gemini AI
- **Deployment**: Docker (optional for production)

## Installation

### Prerequisites

Ensure you have Python installed (recommended version: 3.9+).

### Setup

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/git-guardian.git
   cd Aqua-Guardian
   ```
2. Create a virtual environment (optional but recommended):
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```sh
   pip install -r west/requirements.txt
   ```
4. Create a `.env` file inside the `west` folder and add your Gemini API key:
   ```sh
   GEMINI_API_KEY=your_api_key_here
   ```
5. Run the Flask app (as a module):
   ```sh
   python -m west.app
   ```

## Docker Integration

To deploy using Docker, follow these steps:

1. Build the Docker image, making sure to pass the Gemini API key during the build:
   ```sh
   docker build -t YOUR_CONTAINER_NAME .

   ```
2. Run the container:
   ```sh
   docker run -p 5000:5000 -e GEMINI_API_KEY="YOUR_API_KEHY" water
   ```

## Usage

- Upload an image for **waste detection**, and the app will identify waste and suggest removal methods.
- Input water quality parameters for **rule-based classification**, and the app will assess the quality and recommend improvements.
- Provide necessary data for **water potability prediction**, and the app will determine if the water is safe to drink.

## API Endpoints

| Endpoint              | Method | Description                                         |
| --------------------- | ------ | --------------------------------------------------- |
| `/detect_waste`       | `POST` | Upload an image for waste detection                 |
| `/classify_water`     | `POST` | Provide water quality parameters for classification |
| `/predict_potability` | `POST` | Input water quality features to predict potability  |

## Future Enhancements

- Improve waste detection accuracy with additional training data.
- Extend LLM capabilities for more detailed recommendations.
- Deploy as a cloud-based service with a user-friendly frontend.

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss your proposed changes.

## License

MIT License

---

Developed with ❤️ by Me

