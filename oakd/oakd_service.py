#!/usr/bin/env python3
# Refactored to DepthAI V3
# Original v2 code by Aidan Olsen

from dataclasses import dataclass
import numpy as np
import depthai as dai
import cv2
import json


# -------------------------------
# Capture data container
# -------------------------------
@dataclass
class Capture:
    rgb: np.ndarray
    point_cloud: np.ndarray
    width: int
    height: int

    def get_point(self, x: int, y: int) -> np.ndarray:
        """Get the 3D coordinates (mm) of a pixel in the RGB frame."""
        idx = y * self.width + x
        p = self.point_cloud[idx]

        assert not np.any(np.isnan(p)), "Invalid depth at this point"
        return p

    def distance_between_points(
        self, x1: int, y1: int, x2: int, y2: int
    ) -> float:
        """Euclidean distance between two RGB pixels (mm)."""
        p1 = self.get_point(x1, y1)
        p2 = self.get_point(x2, y2)
        return np.linalg.norm(p1 - p2)
    
    def save(self, filename):
        a = {
            "rgb": self.rgb.tolist(),
            "point_cloud": self.point_cloud.tolist(),
            "width": self.width,
            "height": self.height
        }
        s = json.dumps(a)
        with open(filename, "w") as f:
            f.write(s)


# -------------------------------
# OAK-D service (V3)
# -------------------------------
class OakdService:
    """Manages an OAK-D device with on-demand 3D captures (DepthAI V3)."""

    def __init__(self, fps: int = 30):
        self.fps = fps
        self._init_pipeline()
        self.running = False

    def _init_pipeline(self):
        self.pipeline = dai.Pipeline()

        # RGB camera (unchanged in V3)
        self.camRgb = self.pipeline.create(dai.node.ColorCamera)
        self.camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        self.camRgb.setResolution(
            dai.ColorCameraProperties.SensorResolution.THE_1080_P
        )
        self.camRgb.setIspScale(1, 3)
        self.camRgb.setFps(self.fps)

        # Mono cameras (V3 Camera node)
        self.monoLeft = (
            self.pipeline.create(dai.node.Camera)
            .build(dai.CameraBoardSocket.CAM_B)
        )
        self.monoRight = (
            self.pipeline.create(dai.node.Camera)
            .build(dai.CameraBoardSocket.CAM_C)
        )

        # Stereo depth + point cloud
        self.stereo = self.pipeline.create(dai.node.StereoDepth)
        self.pointcloud = self.pipeline.create(dai.node.PointCloud)

        self.stereo.setDefaultProfilePreset(
            dai.node.StereoDepth.PresetMode.HIGH_DETAIL
        )
        self.stereo.setLeftRightCheck(True)
        self.stereo.setSubpixel(True)
        self.stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

        # Linking
        self.monoLeft.requestFullResolutionOutput().link(self.stereo.left)
        self.monoRight.requestFullResolutionOutput().link(self.stereo.right)
        self.stereo.depth.link(self.pointcloud.inputDepth)

        # Output queues (V3 style)
        self.rgbQueue = self.camRgb.isp.createOutputQueue(
            maxSize=2, blocking=False
        )
        self.pclQueue = self.pointcloud.outputPointCloud.createOutputQueue(
            maxSize=2, blocking=False
        )

    def start(self):
        """Start the pipeline."""
        if self.running:
            return

        print("Starting OAK-D pipeline (V3)")
        self.pipeline.start()
        self.running = True

    def stop(self):
        """Stop the pipeline."""
        if not self.running:
            return

        print("Stopping OAK-D pipeline")
        self.pipeline.stop()
        self.running = False

    def capture(self) -> Capture | None:
        """Capture a single RGB + point cloud frame."""
        if not self.running:
            return None

        rgbMsg = self.rgbQueue.get()
        pclMsg = self.pclQueue.tryGet()

        if pclMsg is None:
            return None

        # RGB
        cv_frame = rgbMsg.getCvFrame()
        rgb = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)

        # Point cloud
        point_cloud = pclMsg.getPoints().astype(np.float64)
        height, width, _ = rgb.shape

        c = Capture(
            rgb=rgb,
            point_cloud=point_cloud,
            width=width,
            height=height,
        )
        return c


# -------------------------------
# Standalone run
# -------------------------------
if __name__ == "__main__":
    service = OakdService()
    service.start()

    print("Starting service")

    import time
    time.sleep(3)

    for _ in range(10):
        print("Capturing")
        capture = service.capture()

        if capture is None:
            print("No capture available")
            continue

        x1, y1 = 100, 100
        x2, y2 = 200, 200

        print("Point 1:", capture.get_point(x1, y1))
        print("Point 2:", capture.get_point(x2, y2))
        print(
            "Distance (mm):",
            capture.distance_between_points(x1, y1, x2, y2),
        )

        time.sleep(0.1)

    service.stop()
    print("bye!")

