# Quick demo of a webserver driving the oakd_service.py
# Run with: waitress-serve --host 127.0.0.1 demo_webserver:app
# Then visit http://localhost:8080
#
# Scaffolded with ChatGPT, then *heavily* hand edited.

from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image
import io
import json
import base64
import numpy as np

from oakd_service import OakdService

oakd_service = OakdService()
oakd_service.start()
print("start called here")

app = Flask(__name__)

latest_image = None  # store last captured image


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/capture", methods=["GET"])
def capture():
    global latest_image

    latest_image = oakd_service.capture()
    im = Image.fromarray(latest_image.rgb)
    jpeg_io = io.BytesIO()
    im.save(jpeg_io, format="JPEG")
    return send_file(
        io.BytesIO(jpeg_io.getvalue()),
        mimetype='image/jpeg',
    )


@app.route("/measure", methods=["POST"])
def measure():
    p1 = request.json["p1"]
    p2 = request.json["p2"]

    distance = latest_image.distance_between_points(p1["x"], p1["y"], p2["x"], p2["y"])

    return jsonify({
        "distance": distance
    })

@app.route("/save", methods=["POST"])
def save():
    data = request.json
    
    if latest_image is None:
        return jsonify({"error": "No image captured"}), 400
        
    # Serialize image to base64 jpeg
    im = Image.fromarray(latest_image.rgb)
    jpeg_io = io.BytesIO()
    im.save(jpeg_io, format="JPEG")
    jpeg_b64 = base64.b64encode(jpeg_io.getvalue()).decode('ascii')
    
    # Serialize point cloud to base64 npz
    npz_io = io.BytesIO()
    np.savez_compressed(npz_io, point_cloud=latest_image.point_cloud)
    npz_b64 = base64.b64encode(npz_io.getvalue()).decode('ascii')
    
    log_entry = {
        "p1": data.get("p1"),
        "p2": data.get("p2"),
        "calculated_distance_mm": data.get("calculated"),
        "actual_distance_mm": data.get("actual"),
        "accuracy_pct": data.get("accuracy"),
        "comment": data.get("comment", ""),
        "image_jpeg_base64": jpeg_b64,
        "point_cloud_npz_base64": npz_b64
    }
    
    with open("benchmark_log.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")
        
    return jsonify({"status": "success"})

