# Quick demo of a webserver driving the oakd_service.py
# Run with: waitress-serve --host 127.0.0.1 demo_webserver:app
# Then visit http://localhost:8080
#
# Scaffolded with ChatGPT, then *heavily* hand edited.

from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image
import io

from oakd_service import OakdService

oakd_service = OakdService()
oakd_service.start()

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
