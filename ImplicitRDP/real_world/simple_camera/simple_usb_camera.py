import cv2
from loguru import logger
import pyudev


class SimpleUSBCamera:
    def __init__(self,
                 camera_index=0,
                 width=640,
                 height=360,
                 fps=10,
                 exposure=-6,
                 contrast=100):
        self.context = pyudev.Context()
        self.camera_index = camera_index
        self.cap = None
        self.width = width
        self.height = height
        self.fps = fps
        self.exposure = exposure
        self.contrast = contrast

    def start(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.camera_index)
            self.set_camera_intrisics(self.cap, self.width, self.height, self.contrast, self.exposure)
            if not self.cap.isOpened():
                logger.error("Could not open video device")
                raise Exception("Could not open video device")
            logger.info("Camera started")
        else:
            logger.warning("Camera is already running")
    
    def set_camera_intrisics(self, camera, width, height, contrast, exposure):
        '''
        set the resolution, contarst and resolution of the camera
        '''
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        camera.set(cv2.CAP_PROP_CONTRAST, contrast)  # contrast
        camera.set(cv2.CAP_PROP_EXPOSURE, exposure)  # exposure
        camera.set(cv2.CAP_PROP_FPS, self.fps)
        # Log the actual resolution set
        logger.debug(f"Setting camera resolution to ({width}, {height}), contrast: {contrast}, exposure: {exposure}, fps: {self.fps}")
        actual_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = camera.get(cv2.CAP_PROP_FPS)
        logger.debug(f"Actual camera resolution: ({actual_width}, {actual_height}), fps: {actual_fps}")

    def stop(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("Camera stopped")
        else:
            logger.warning("Camera is not running")

    def get_rgb_frame(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Failed to capture image")
                raise Exception("Failed to capture image")
            return frame
        else:
            logger.error("Camera is not running")
            raise Exception("Camera is not running")


# Example usage:
if __name__ == "__main__":
    camera_index = 0
    width = 640
    height = 360
    exposure = -6
    contrast = 100
    fps = 10
    camera = SimpleUSBCamera(camera_index=camera_index,
                             width=width,
                             height=height,
                             exposure=exposure,
                             contrast=contrast,
                             fps=fps)
    
    try:
        camera.start()
        while True:
            frame = camera.get_rgb_frame()
            # Display the frame
            cv2.imshow('RGB Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        logger.exception(e)
    finally:
        camera.stop()
        cv2.destroyAllWindows()