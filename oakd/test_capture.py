from oakd_service import OakdService

service = OakdService()
service.start()
try:
    capture = service.capture()
    if capture:
        print(f"RGB shape: {capture.rgb.shape}")
        print(f"Point Cloud shape: {capture.point_cloud.shape}")
    else:
        print("Capture failed")
finally:
    service.stop()
