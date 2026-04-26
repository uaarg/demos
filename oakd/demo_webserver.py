# Quick demo of a webserver driving the oakd_service.py
# Run with: waitress-serve --host 127.0.0.1 demo_webserver:app
# Then visit http://localhost:8080
#
# Scaffolded with ChatGPT, then *heavily* hand edited.
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
import os
import numpy as np
import cv2
import open3d as o3d

from oakd_service import OakdService

oakd_service = OakdService()
oakd_service.start()
print("start called here")

app = Flask(__name__)

latest_image = None  # store last captured image


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/3d")
def view_3d():
    return render_template("3d_viewer.html")


@app.route("/capture", methods=["GET"])
def capture():
    global latest_image

    is_burst = request.args.get('burst', 'false').lower() == 'true'
    burst_frames = int(request.args.get('frames', '5'))

    if is_burst:
        print(f"Webserver requested burst capture of {burst_frames} frames.")
        latest_image = oakd_service.capture_burst(burst_frames)
    else:
        latest_image = oakd_service.capture()
        
    im = Image.fromarray(latest_image.rgb)
    jpeg_io = io.BytesIO()
    im.save(jpeg_io, format="JPEG")
    return send_file(
        io.BytesIO(jpeg_io.getvalue()),
        mimetype='image/jpeg',
    )


@app.route("/depth", methods=["GET"])
def depth():
    global latest_image

    if latest_image is None:
        return jsonify({"error": "No image captured"}), 400

    # Extract Z-coordinates (depth in mm) from the point cloud
    # point_cloud is flat: (width * height, 3). We extract the Z column (index 2)
    depth_z = latest_image.point_cloud[:, 2]

    # Reshape it to 2D
    depth_map = depth_z.reshape((latest_image.height, latest_image.width))

    # Identify valid depth readings (ignoring exactly 0 or NaNs)
    valid_mask = (depth_map > 0) & (~np.isnan(depth_map))

    # Initialize a clean canvas for our colorized depth map
    colorized_depth = np.zeros((latest_image.height, latest_image.width, 3), dtype=np.uint8)

    if np.any(valid_mask):
        # Determine the min and max depth for normalization
        min_depth = np.min(depth_map[valid_mask])
        max_depth = np.max(depth_map[valid_mask])
        
        # Avoid division by zero if all points are at the exact same distance
        if max_depth > min_depth:
            # Normalize the valid depths to 0-255 range (uint8)
            # We invert it so closer objects are lighter/warmer
            normalized_depth = ((max_depth - depth_map) / (max_depth - min_depth) * 255).astype(np.uint8)
        else:
            normalized_depth = np.full_like(depth_map, 128, dtype=np.uint8)

        # Apply a colormap (JET is classic for depth map visualization: red=close, blue=far)
        # We only apply it where we have valid data.
        tmp_colored = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)
        
        # applyColorMap returns BGR, we need RGB for Pillow
        tmp_colored = cv2.cvtColor(tmp_colored, cv2.COLOR_BGR2RGB)

        # Map back only the valid pixels to our black canvas
        colorized_depth[valid_mask] = tmp_colored[valid_mask]


    # Convert to image and return as JPEG
    im = Image.fromarray(colorized_depth)
    jpeg_io = io.BytesIO()
    im.save(jpeg_io, format="JPEG")
    return send_file(
        io.BytesIO(jpeg_io.getvalue()),
        mimetype='image/jpeg',
    )


@app.route("/pointcloud_data", methods=["GET"])
def pointcloud_data():
    global latest_image

    if latest_image is None:
        return jsonify({"error": "No image captured"}), 400

    # The point cloud is of shape (N, 3) where N = width * height
    points = latest_image.point_cloud
    
    # We also have the RGB image, shape (height, width, 3). Flatten it to (N, 3)
    colors = latest_image.rgb.reshape(-1, 3)

    # Filter out invalid points. A Z of <= 0 or NaN is invalid.
    # We also cap it at 15000mm (15 meters) because noisy extreme depths
    # will cause the 3D bounding box to explode and push the camera too far back.
    z_coords = points[:, 2]
    valid_mask = (z_coords > 0) & (z_coords < 15000) & (~np.isnan(z_coords))

    valid_points = points[valid_mask].astype(np.float32)
    valid_colors = colors[valid_mask].astype(np.uint8)

    # Convert arrays to raw bytes
    points_bytes = valid_points.tobytes()
    colors_bytes = valid_colors.tobytes()
    
    # Concatenate the buffers: floats first, then uint8s
    final_buffer = points_bytes + colors_bytes

    return send_file(
        io.BytesIO(final_buffer),
        mimetype='application/octet-stream',
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


@app.route("/test_config", methods=["POST"])
def test_config():
    global latest_image
    data = request.json
    config = data.get("config", {})
    p1 = data.get("p1")
    p2 = data.get("p2")
    actual = float(data.get("actual"))

    # Restart pipeline with new config
    oakd_service.restart(config)

    # Capture
    if config.get("burst_mode"):
        latest_image = oakd_service.capture_burst(5)
    else:
        latest_image = oakd_service.capture()

    if not latest_image:
        return jsonify({"error": "Failed to capture image"}), 500

    try:
        calc_dist = latest_image.distance_between_points(int(p1["x"]), int(p1["y"]), int(p2["x"]), int(p2["y"]))
        calc_dist = float(calc_dist)
            
        error = abs(calc_dist - actual)
        return jsonify({
            "calculated": calc_dist,
            "error": error
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/save_suite", methods=["POST"])
def save_suite():
    data = request.json
    results = data.get("results", [])
    if not results:
        return jsonify({"error": "No results to save"}), 400
        
    import csv, os
    from datetime import datetime
    
    filename = "test_bench_results_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"
    filepath = os.path.join(os.path.expanduser("~"), "Desktop", "demos", "oakd", filename)
    
    keys = ["resolution", "temporal_filter", "spatial_filter", "burst_mode", "confidence", "calculated", "error"]
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in results:
            filtered_r = {k: r.get(k) for k in keys}
            writer.writerow(filtered_r)
            
    return jsonify({"status": "success", "filename": filename})


@app.route("/api/logs/json", methods=["GET"])
def get_json_logs():
    logs = []
    filepath = "benchmark_log.jsonl"
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    entry["log_idx"] = len(logs)
                    # Remove the bulky point cloud data to save bandwidth
                    if "point_cloud_npz_base64" in entry:
                        del entry["point_cloud_npz_base64"]
                    logs.append(entry)
                except Exception:
                    pass
    logs.reverse() # Show newest first
    return jsonify(logs)

@app.route("/api/logs/json/<int:log_idx>/pointcloud_data", methods=["GET"])
def get_log_pointcloud(log_idx):
    filepath = "benchmark_log.jsonl"
    if not os.path.exists(filepath):
        return jsonify({"error": "No logs"}), 404
        
    try:
        with open(filepath, "r") as f:
            for i, line in enumerate(f):
                if i == log_idx:
                    entry = json.loads(line)
                    if "point_cloud_npz_base64" not in entry or "image_jpeg_base64" not in entry:
                        return jsonify({"error": "Missing 3D data"}), 400
                        
                    import base64, io, numpy as np
                    from PIL import Image
                    
                    npz_data = base64.b64decode(entry["point_cloud_npz_base64"])
                    npz_io = io.BytesIO(npz_data)
                    npz = np.load(npz_io)
                    points = npz["point_cloud"]
                    
                    jpeg_data = base64.b64decode(entry["image_jpeg_base64"])
                    jpeg_io = io.BytesIO(jpeg_data)
                    im = Image.open(jpeg_io)
                    colors = np.array(im).reshape(-1, 3)
                    
                    z_coords = points[:, 2]
                    valid_mask = (z_coords > 0) & (z_coords < 15000) & (~np.isnan(z_coords))
                    
                    valid_points = points[valid_mask].astype(np.float32)
                    valid_colors = colors[valid_mask].astype(np.uint8)

                    final_buffer = valid_points.tobytes() + valid_colors.tobytes()

                    return send_file(
                        io.BytesIO(final_buffer),
                        mimetype='application/octet-stream',
                    )
            return jsonify({"error": "Index out of bounds"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/logs/csv", methods=["GET"])
def get_csv_list():
    import glob
    dir_path = os.path.join(os.path.expanduser("~"), "Desktop", "demos", "oakd")
    search_pattern = os.path.join(dir_path, "test_bench_results_*.csv")
    files = glob.glob(search_pattern)
    filenames = [os.path.basename(f) for f in files]
    filenames.sort(reverse=True)
    return jsonify(filenames)

@app.route("/api/logs/csv/<filename>", methods=["GET"])
def get_csv_file(filename):
    if ".." in filename or "/" in filename:
        return jsonify({"error": "Invalid filename"}), 400
        
    filepath = os.path.join(os.path.expanduser("~"), "Desktop", "demos", "oakd", filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404
        
    import csv
    rows = []
    try:
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        return jsonify({"columns": reader.fieldnames, "rows": rows})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
