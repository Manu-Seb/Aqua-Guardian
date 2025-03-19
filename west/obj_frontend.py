from flask import Flask, render_template, request, jsonify
import matplotlib.pyplot as plt
import seaborn as sns
import os
from . import dark_channel_prior
from . import rule_based_classifier as rbc
from .inference import garbage
from . import water_potability

app = Flask(__name__)

# Waste Labels
labels = [
    'Mask', 'can', 'cellphone', 'electronics', 'gbottle', 'glove', 'metal',
    'misc', 'net', 'pbag', 'pbottle', 'plastic', 'rod', 'sunglasses', 'tire'
]

@app.route("/")
def home():
    """ Home Page """
    return render_template("index.html")

@app.route("/waste_detection")
def waste_detection():
    """ Waste Detection Page """
    return render_template("waste_detection.html")

@app.route("/water_quality")
def water_quality():
    """ Water Quality Assessment Page """
    return render_template("water_quality.html")

@app.route("/water_potability")
def potability_page():
    """ Water Potability Test Page """
    return render_template("water_potability.html")

@app.route("/report")
def generate_report():
    """ Generate Report and Display Results """

    # Waste Label Occurrences
    occurrences = {label: garbage.count(label) for label in labels}

    # Prevent division by zero error
    if not any(occurrences.values()):
        return jsonify({"error": "No waste detection data available"}), 400

    # Waste Bar Chart
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(y=list(occurrences.keys()), x=list(occurrences.values()), ax=ax)
    ax.set_xlabel("Occurrences")
    ax.set_ylabel("Labels")
    ax.set_title("Histogram of Waste Occurrences")
    
    img_path = "static/reports/waste_histogram.png"
    fig.savefig(img_path)  # Save chart

    # Water Quality (Aquatic Life)
    quality_aquatic = rbc.quality_aquatic
    if quality_aquatic:
        aquatic_counts = [quality_aquatic.count(0), quality_aquatic.count(1)]
        labels_h = ['Habitual', 'Not Habitual']

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(aquatic_counts, labels=labels_h, autopct='%1.1f%%', startangle=90, colors=['#cfaca4', '#623337'])
        ax.set_title('Proportions Of Water Quality (Aquatic Life)')
        fig.savefig("static/reports/aquatic_quality.png")

        habitual_status = labels_h[max(set(quality_aquatic), key=quality_aquatic.count)]
    else:
        aquatic_counts = None
        habitual_status = "Unknown"

    # Water Quality (Potability)
    water_quality_results = water_potability.get_potability_results()  # Assuming a function for retrieving results
    if water_quality_results:
        potability_counts = [water_quality_results.count(0), water_quality_results.count(1)]
        labels_wqa = ['Fit for use', 'Polluted']

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(potability_counts, labels=labels_wqa, autopct='%1.1f%%', startangle=90, colors=['#1f77b4', '#ff7f0e'])
        ax.set_title('Proportions Of Water Potability')
        fig.savefig("static/reports/potability.png")

        qwa_status = labels_wqa[max(set(water_quality_results), key=water_quality_results.count)]
    else:
        potability_counts = None
        qwa_status = "Unknown"

    # Find most seen waste
    most_seen_waste, most_seen_count = "None", 0
    if any(occurrences.values()):
        most_seen_waste = max(occurrences, key=occurrences.get)
        most_seen_count = occurrences[most_seen_waste]

    return render_template(
        "report.html",
        waste_img=img_path,
        aquatic_counts=aquatic_counts,
        potability_counts=potability_counts,
        most_seen_waste=most_seen_waste,
        most_seen_count=most_seen_count,
        habitual_status=habitual_status,
        qwa_status=qwa_status
    )

@app.route("/haze_removal", methods=['POST'])
def remove_haze():
    """ Apply haze removal using dark channel prior """
    try:
        image = request.files['image']
        if not image:
            return jsonify({"error": "No image file provided"}), 400

        # Convert image to OpenCV format
        import cv2
        import numpy as np
        from PIL import Image

        img = Image.open(image)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Apply dark channel haze removal
        dehazed_img, _ = dark_channel_prior.haze_removal(img)

        # Save and return image
        dehazed_img_path = "static/reports/dehazed_image.png"
        cv2.imwrite(dehazed_img_path, dehazed_img)

        return jsonify({"dehazed_image": dehazed_img_path})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
