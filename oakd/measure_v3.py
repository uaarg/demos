#!/usr/bin/env python3

import depthai as dai
import numpy as np
import cv2

# -------------------------------
# Mouse click handling
# -------------------------------
clicked_pixels = []
snapshot_rgb = None
snapshot_points = None
snapshot_width = None
snapshot_height = None

def mouse_callback(event, x, y, flags, param):
    global clicked_pixels, snapshot_rgb

    if event == cv2.EVENT_LBUTTONDOWN and snapshot_rgb is not None:
        clicked_pixels.append((x, y))
        print(f"Clicked pixel: ({x}, {y})")

        cv2.circle(snapshot_rgb, (x, y), 5, (0, 0, 255), -1)

        if len(clicked_pixels) == 2:
            compute_distance()

def compute_distance():
    global clicked_pixels, snapshot_points, snapshot_width

    (x1, y1), (x2, y2) = clicked_pixels

    idx1 = y1 * snapshot_width + x1
    idx2 = y2 * snapshot_width + x2

    p1 = snapshot_points[idx1]
    p2 = snapshot_points[idx2]

    if np.any(np.isnan(p1)) or np.any(np.isnan(p2)):
        print("Invalid depth at one of the points.")
        return

    dist = np.linalg.norm(p1 - p2)

    print(f"Point 1 (mm): {p1}")
    print(f"Point 2 (mm): {p2}")
    print(f"Euclidean distance: {dist:.2f} mm")


# -------------------------------
# DepthAI V3 pipeline
# -------------------------------
pipeline = dai.Pipeline()

# RGB camera (still correct in V3)
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setIspScale(1, 3)

# Mono cameras (V3 Camera node)
monoLeft = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
monoRight = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

# Stereo + PointCloud
stereo = pipeline.create(dai.node.StereoDepth)
pointcloud = pipeline.create(dai.node.PointCloud)

stereo.setDefaultProfilePreset(
    dai.node.StereoDepth.PresetMode.FAST_DENSITY
)
stereo.setLeftRightCheck(True)
stereo.setSubpixel(True)
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

# Linking
monoLeft.requestFullResolutionOutput().link(stereo.left)
monoRight.requestFullResolutionOutput().link(stereo.right)
stereo.depth.link(pointcloud.inputDepth)

# Outputs (V3 queues)
rgbQueue = camRgb.isp.createOutputQueue(maxSize=4, blocking=False)
pclQueue = pointcloud.outputPointCloud.createOutputQueue(maxSize=4, blocking=False)

# -------------------------------
# Run (V3 lifecycle)
# -------------------------------
cv2.namedWindow("color")
cv2.setMouseCallback("color", mouse_callback)

print("Press 'c' to capture snapshot")
print("Click two points to measure distance")
print("Press 'r' to reset")
print("Press 'q' to quit")

with pipeline:
    pipeline.start()

    while pipeline.isRunning():
        rgbMsg = rgbQueue.get()
        pclMsg = pclQueue.tryGet()

        rgb = rgbMsg.getCvFrame()

        key = cv2.waitKey(1)

        if key == ord('q'):
            pipeline.stop()
            break

        if key == ord('c') and pclMsg is not None:
            snapshot_rgb = rgb.copy()
            snapshot_points = pclMsg.getPoints().astype(np.float64)
            snapshot_height, snapshot_width, _ = snapshot_rgb.shape
            clicked_pixels.clear()
            print("Snapshot captured.")

        if key == ord('r'):
            clicked_pixels.clear()
            print("Reset.")

        if snapshot_rgb is not None:
            cv2.imshow("color", snapshot_rgb)
        else:
            cv2.imshow("color", rgb)

cv2.destroyAllWindows()

