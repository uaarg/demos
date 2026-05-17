# Service class wrapper. Can be run on it's own (see the if __name__ == "__main__" below).
# Handwritten by Aidan Olsen

from dataclasses import dataclass
import numpy as np
import depthai as dai
import cv2


@dataclass
class Capture:
    rgb: np.ndarray
    point_cloud: np.ndarray
    width: int
    height: int

    def get_point(self, x: int, y: int) -> np.ndarray:
        """Get the 3D coordinates relative to the camera frame (in mm) of a pixel).

        (x, y) are in pixel coordinates in the self.rgb frame."""
        idx = y * self.width + x
        p = self.point_cloud[idx]

        # I am not sure if this is possible...
        assert not np.any(np.isnan(p)), "Invalid depth at this point"

        return p

    def distance_between_points(self, x1: int, y1: int, x2: int, y2: int):
        """Get the physical distance between points from pixels in the rgb
        frame (x1, y1) and (x2, y2).

        Resulting distance is in mm."""
        p1 = self.get_point(x1, y1)
        p2 = self.get_point(x2, y2)

        dist = np.linalg.norm(p1 - p2)

        return dist


class OakdService:
    """Manages an OAK-D device with on-demand capturing of 3D pictures (see Capture)."""

    def __init__(self, fps: int = 30, config: dict = None):
        self._init_pipeline(fps, config)

    def _init_pipeline(self, fps: int, config: dict = None):
        if config is None:
            config = {
                "resolution": "400p",
                "temporal_filter": True,
                "spatial_filter": True,
                "confidence": 245
            }
        """Initialize the Depth AI pipeline (will be run on the OAK-D)"""
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
        camRgb.setFps(fps)

        res = dai.MonoCameraProperties.SensorResolution.THE_800_P if config.get("resolution", "400p").lower() == "800p" else dai.MonoCameraProperties.SensorResolution.THE_400_P
        
        monoLeft.setResolution(res)
        monoLeft.setCamera("left")
        monoLeft.setFps(fps)

        monoRight.setResolution(res)
        monoRight.setCamera("right")
        monoRight.setFps(fps)

        depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        depth.setLeftRightCheck(True)
        depth.setSubpixel(True)
        depth.setDepthAlign(dai.CameraBoardSocket.CAM_A)

        # Configure hardware filters based on config dict
        cfg = depth.initialConfig.get()
        
        cfg.costMatching.confidenceThreshold = config.get("confidence", 245)
        
        if config.get("temporal_filter", True):
            cfg.postProcessing.temporalFilter.enable = True
        else:
            cfg.postProcessing.temporalFilter.enable = False
            
        if config.get("spatial_filter", True):
            cfg.postProcessing.spatialFilter.enable = True
            cfg.postProcessing.spatialFilter.holeFillingRadius = 2
            cfg.postProcessing.spatialFilter.alpha = 0.5
            cfg.postProcessing.spatialFilter.delta = 20
        else:
            cfg.postProcessing.spatialFilter.enable = False
            
        # Add decimation if 800P + filters to save performance, else clear it
        use_decimation = config.get("resolution", "400p").lower() == "800p" and (config.get("temporal_filter") or config.get("spatial_filter"))
        cfg.postProcessing.decimationFilter.decimationFactor = 2 if use_decimation else 1
            
        depth.initialConfig.set(cfg)

        monoLeft.out.link(depth.left)
        monoRight.out.link(depth.right)
        depth.depth.link(pointcloud.inputDepth)

        camRgb.isp.link(sync.inputs["rgb"])
        pointcloud.outputPointCloud.link(sync.inputs["pcl"])

        sync.out.link(xOut.input)
        xOut.setStreamName("out")

        self.pipeline = pipeline

    def capture(self) -> Capture | None:
        """Capture a current 3D frame on the OAK-D.

        NOTE: .start() must have been called first. If it has not, this will only return None."""
        if not self.device or self.device.isClosed():
            print("seems like device is still closed")
            return None

        msg = self.queue.get()
        rgbFrame = msg["rgb"]
        cv_frame = rgbFrame.getCvFrame()
        rgb_frame = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)
        rgb = np.array(rgb_frame)
        pcl = msg["pcl"]

        point_cloud = pcl.getPoints().astype(np.float64)
        height, width, _ = rgb.shape

        capture = Capture(rgb, point_cloud, width, height)
        return capture

    def capture_burst(self, num_frames: int = 5) -> Capture | None:
        """Capture multiple frames and average their valid depth points."""
        if not self.device or self.device.isClosed():
            print("seems like device is still closed")
            return None

        print(f"Starting burst capture of {num_frames} frames...")
        # Flush the queue to ensure we get fresh frames from this moment
        while self.queue.has():
            self.queue.get()

        rgb_frames = []
        pcl_frames = []
        
        for i in range(num_frames):
            msg = self.queue.get()
            cv_frame = msg["rgb"].getCvFrame()
            rgb = np.array(cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB))
            rgb_frames.append(rgb)
            
            pcl = msg["pcl"].getPoints().astype(np.float64)
            pcl_frames.append(pcl)
            print(f"Captured burst frame {i+1}/{num_frames}")

        final_rgb = rgb_frames[-1]
        height, width, _ = final_rgb.shape
        
        pcl_stack = np.stack(pcl_frames, axis=0) # shape: (num_frames, N, 3)
        z_coords = pcl_stack[:, :, 2]
        
        # create mask of valid points (where Z > 0 and Z is not NaN)
        valid_mask = (z_coords > 0) & (~np.isnan(z_coords)) # (num_frames, N)
        valid_mask_expanded = np.expand_dims(valid_mask, axis=-1)
        
        # set invalid points to 0 to safely sum them up
        valid_points = np.where(valid_mask_expanded, pcl_stack, 0.0)
        sum_points = np.sum(valid_points, axis=0) # (N, 3)
        
        # count how many frames each pixel was valid in
        valid_count = np.sum(valid_mask, axis=0) # (N,)
        valid_count_expanded = np.expand_dims(valid_count, axis=-1)
        
        # avoid division by zero
        safe_count = np.where(valid_count_expanded > 0, valid_count_expanded, 1)
        
        # average
        avg_pcl = sum_points / safe_count
        
        # explicitly zero out pixels that had no valid frames
        invalid_final_mask = (valid_count == 0)
        avg_pcl[invalid_final_mask] = 0.0

        print("Burst capture complete.")
        return Capture(final_rgb, avg_pcl, width, height)

    def start(self):
        """Start the depth-perception process on the OAK-D"""
        print("Starting OAK-D Connection...")
        try:
            self.device = dai.Device(self.pipeline)
            self.queue = self.device.getOutputQueue("out", maxSize=1, blocking=False)
        except RuntimeError as e:
            print("\n" + "="*50)
            print("🚨 ERROR: Cannot find the OAK-D Camera!")
            print("This usually happens if:")
            print("  1. The camera is physically unplugged.")
            print("  2. Another python script is already running and owns the camera.")
            print("  3. The camera hardware hung from a previous ungraceful shutdown.")
            print("\nFIX: Unplug the OAK-D USB cable and plug it back in, then try again.")
            print("="*50 + "\n")
            raise e

    def restart(self, config: dict):
        """Restart the pipeline with new configuration."""
        print(f"Restarting OAK-D with config: {config}")
        if hasattr(self, 'device') and self.device and not self.device.isClosed():
            self.stop()
            import time
            print("Cooling down for 1 second to ensure USB handle is fully released...")
            time.sleep(1.0)
            
        self._init_pipeline(fps=30, config=config)
        self.start()
        
        import time
        time.sleep(1.5) # Allow sensors to settle
        
    def stop(self):
        """Stop the depth-perception process"""
        if hasattr(self, 'device') and self.device:
            self.device.close()
        self.queue = None


if __name__ == "__main__":
    service = OakdService()
    service.start()

    print("Starting service")

    import time
    time.sleep(3)

    for _ in range(10):
        print("Capturing")
        capture = service.capture()

        x1, y1 = 100, 100
        print("point 1", (x1, y1), capture.get_point(x1, y1))

        x2, y2 = 200, 200
        print("point 2", (x2, y2), capture.get_point(x2, y2))

        print("distance", capture.distance_between_points(x1, y1, x2, y2))

        time.sleep(0.1)

    print("stopping")
    service.stop()

    print("bye!")
