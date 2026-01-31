# Version of depth_cloud.py for measuring distances
# Generated, in part, by ChatGPT

import depthai as dai
import numpy as np
import cv2

FPS = 30

# -------------------------------
# Mouse click handling
# -------------------------------
clicked_pixels = []
snapshot_rgb = None
snapshot_points = None
snapshot_width = None
snapshot_height = None

def mouse_callback(event, x, y, flags, param):
    global clicked_pixels

    if event == cv2.EVENT_LBUTTONDOWN and snapshot_rgb is not None:
        clicked_pixels.append((x, y))
        print(f"Clicked pixel: ({x}, {y})")

        # Draw point
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
# DepthAI pipeline
# -------------------------------
pipeline = dai.Pipeline()

camRgb = pipeline.create(dai.node.ColorCamera)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
depth = pipeline.create(dai.node.StereoDepth)
pointcloud = pipeline.create(dai.node.PointCloud)
sync = pipeline.create(dai.node.Sync)
xOut = pipeline.create(dai.node.XLinkOut)

camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
camRgb.setIspScale(1, 3)
camRgb.setFps(FPS)

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setCamera("left")
monoLeft.setFps(FPS)

monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setCamera("right")
monoRight.setFps(FPS)

depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
depth.setLeftRightCheck(True)
depth.setSubpixel(True)
depth.setDepthAlign(dai.CameraBoardSocket.CAM_A)

monoLeft.out.link(depth.left)
monoRight.out.link(depth.right)
depth.depth.link(pointcloud.inputDepth)

camRgb.isp.link(sync.inputs["rgb"])
pointcloud.outputPointCloud.link(sync.inputs["pcl"])

sync.out.link(xOut.input)
xOut.setStreamName("out")

# -------------------------------
# Run
# -------------------------------
with dai.Device(pipeline) as device:
    q = device.getOutputQueue("out", maxSize=4, blocking=False)

    cv2.namedWindow("color")
    cv2.setMouseCallback("color", mouse_callback)

    print("Press 'c' to capture snapshot")
    print("Click two points to measure distance")
    print("Press 'r' to reset")
    print("Press 'q' to quit")

    while True:
        msg = q.get()
        rgb = msg["rgb"].getCvFrame()
        pcl = msg["pcl"]

        key = cv2.waitKey(1)

        if key == ord('q'):
            break

        if key == ord('c') and pcl is not None:
            snapshot_rgb = rgb.copy()
            snapshot_points = pcl.getPoints().astype(np.float64)
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
