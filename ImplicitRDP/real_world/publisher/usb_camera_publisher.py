import numpy as np
import rclpy
import bson
import socket
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image, PointCloud2, PointField
import json
from loguru import logger
import math
import uuid
import time
import cv2
import time as tm
import struct
import os


class UsbCameraPublisher(Node):
    '''
    Usb Camera publisher Class
    '''

    def __init__(self,
                 camera_index: int = 0,
                 camera_type: str = 'USB',
                 width: int = 640,
                 height: int = 360,
                 fps: int = 30,
                 exposure: int = -6,
                 contrast: int = 100,
                 camera_name: str = 'left_wrist_camera',
                 debug=False,
                 recorded=False,
                 image_folder='data/realworld',
                 video_path="../../../data/tactile_video/video_001.mp4",
                 enable_streaming: bool = False,
                 streaming_server_ip: str = '127.0.0.1',
                 streaming_server_port: int = 10004,
                 streaming_quality: int = 10,
                 streaming_chunk_size: int = 1024,
                 streaming_display_params_list: list = None,
                 ):
        node_name = f'{camera_name}_publisher_{camera_index}'
        super().__init__(node_name)
        self.camera_index = camera_index
        self.camera_name = camera_name
        self.cap = None
        self.img = None
        self.marker_img = None
        self.fps = fps
        self.contrast = contrast
        self.exposure = exposure

        self.width = width
        self.height = height
        self.debug = debug
        self.recorded = recorded
        self.camera_type = camera_type

        self.color_publisher_ = self.create_publisher(Image, f'/{camera_name}/color/image_raw', 10)
        self.marker_publisher = self.create_publisher(PointCloud2, f'/{camera_name}/marker_offset/information', 10)
        self.timer = self.create_timer(1 / fps, self.timer_callback)
        self.timestamp_offset = None

        self.last_print_time = tm.time()  # Add a variable to keep track of the last print time
        self.fps_list = []
        self.frame_intervals = []
        self.last_frame_time = None

        self.prev_time = time.time()
        self.frame_count = 0

        self.video_path = video_path
        if recorded:
            assert os.path.exists(self.video_path), f"Video path {self.video_path} does not exist!"

        # streaming configuration
        self.enable_streaming = enable_streaming
        if self.enable_streaming:
            self.id = uuid.uuid4()
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.streaming_server_ip = streaming_server_ip
            self.streaming_server_port = streaming_server_port
            self.streaming_quality = streaming_quality
            self.streaming_chunk_size = streaming_chunk_size
            streaming_display_params_list = [{k: list(v) for k, v in d.items()} for d in
                                             streaming_display_params_list]
            self.streaming_display_params_list = streaming_display_params_list

        # start the camera
        self.start()

        # Create a socket
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def start(self):
        '''
        Start the usb camera
        Usb camera has no internal time,
        so we use the time we get the frame as the initial time of the topic
        '''
        if not self.recorded:
            if self.cap is None:
                self.cap = cv2.VideoCapture(self.camera_index)
                self.set_camera_intrisics(self.cap, self.width, self.height, self.contrast, self.exposure)

                if not self.cap.isOpened():
                    self.cap.open(self.camera_index)
                    self.set_camera_intrisics(self.cap, self.width, self.height, self.contrast, self.exposure)
                    if not self.cap.isOpened():
                        logger.error("Could not open video device")
                        raise Exception("Could not open video device")

                logger.info(f"{self.camera_name} started")
            else:
                logger.warning("Camera is already running")
        else:
            self.cap = cv2.VideoCapture(self.video_path)

    def set_camera_intrisics(self, camera, width, height, contrast, exposure):
        '''
        set the resolution, contarst and resolution of the camera
        '''
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        camera.set(cv2.CAP_PROP_CONTRAST, contrast)  # contrast
        camera.set(cv2.CAP_PROP_EXPOSURE, exposure)  # exposure

        actual_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        logger.debug(f"Requested resolution: ({width}, {height}), Actual resolution: ({actual_width}, {actual_height})")

    def stop(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info(f"Camera {self.camera_index} stopped")
        else:
            logger.warning("Camera is not running")

    def get_rgb_frame(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            timestamp = self.get_clock().now()
            if not ret:
                logger.error(f"Failed to capture image from camera {self.camera_index}")
                raise Exception("Failed to capture image")
            else:
                self.img = frame
            return frame, timestamp
        else:
            logger.error("Camera is not running")
            raise Exception("Camera is not running")

    def publish_marker_offset(self, camera_timestamp: Time):
        # Dummy Data, Only for data recorder
        # Fill the message
        msg = PointCloud2()
        msg.header.stamp = camera_timestamp.to_msg()
        msg.header.frame_id = f'camera_marker_offset_{self.camera_name}'
        msg.is_bigendian = False
        msg.point_step = 16
        msg.is_dense = True
        msg.fields = [
            PointField(name='marker_location_x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='marker_location_y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='marker_offset_x', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='marker_offset_y', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        # Fill the data
        pointcloud_data = b''.join(
            map(lambda row: struct.pack('ffff', row[0], row[1], row[2], row[3]), [[0, 0, 0, 0]])
        )
        msg.data = pointcloud_data

        self.marker_publisher.publish(msg)

    def publish_color_image(self, color_image, camera_timestamp: Time):
        '''
        publish the color image and track markers
        '''
        success, encoded_image = cv2.imencode('.jpg', color_image)

        # Fill the message
        msg = Image()
        msg.header.stamp = camera_timestamp.to_msg()
        msg.header.frame_id = f"camera_color_frame_{self.camera_index}"
        msg.height, msg.width, _ = color_image.shape
        msg.encoding = "bgr8"
        msg.step = msg.width * 3
        if success:
            image_bytes = encoded_image.tobytes()
            msg.data = image_bytes
        else:
            logger.warning('fail to image encoding!')
            msg.data = color_image.tobytes()
        self.color_publisher_.publish(msg)

    def send_streaming_msg(self, color_image):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.streaming_quality]
        ret, color_image_encoded = cv2.imencode('.jpg', color_image, encode_param)
        color_image_bytes = color_image_encoded.tobytes()
        packed_data_dict = {"images": [{"id": self.id,
                                        "inHeadSpace": False,
                                        **display_params,
                                        **{"image": color_image_bytes}}
                                        for display_params in self.streaming_display_params_list]}
        packed_data = bson.dumps(packed_data_dict)

        arrow_address = (self.streaming_server_ip, self.streaming_server_port)
        chunk_size = self.streaming_chunk_size

        self.socket.sendto(len(packed_data).to_bytes(length=4, byteorder='little', signed=False), arrow_address)
        if self.debug:
            logger.debug(f"Sending streaming image to VR server with size {len(packed_data)}")

        self.socket.sendto(chunk_size.to_bytes(length=4, byteorder='little', signed=False), arrow_address)
        count = math.ceil(len(packed_data) / chunk_size)
        if self.debug:
            logger.debug(f"Sending streaming image to VR server with {count} chunks of size {chunk_size}")

        for i in range(count):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            if end > len(packed_data):
                end = len(packed_data)
            self.socket.sendto(packed_data[start:end], arrow_address)
        if self.debug:
            logger.debug(f"Sent streaming image to VR server")

    def timer_callback(self):
        '''
        Publish the color frames
        '''
        while True:
            # get color frames
            color_frame, initial_time = self.get_rgb_frame()

            # If rgb frame or marker information not availble, continue
            if color_frame is None:
                continue

            # get the internal camera timestamp of the color frame
            camera_timestamp = initial_time

            # publish
            self.publish_marker_offset(camera_timestamp)

            # publish the color image
            self.publish_color_image(color_frame, camera_timestamp)

            # send streaming image
            if self.enable_streaming:
                color_image = color_frame.copy()
                self.send_streaming_msg(color_image)

            # calculate fps
            self.frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - self.prev_time
            if elapsed_time >= 1.0:
                frame_rate = self.frame_count / elapsed_time
                self.fps_list.append(frame_rate)
                logger.debug(f"Frame rate: {frame_rate:.2f} FPS")
                self.prev_time = current_time
                self.frame_count = 0

            # calculate the interval between two frames
            if self.last_frame_time is not None:
                frame_interval = (current_time - self.last_frame_time) * 1000
                self.frame_intervals.append(frame_interval)
            self.last_frame_time = current_time

            # Print info and make plot every 5 seconds
            if current_time - self.last_print_time >= 5:
                logger.info(f"Publishing image from {self.camera_name} at timestamp (s): {initial_time.nanoseconds / 1e9}")
                self.last_print_time = current_time

            break


def main(args=None):
    rclpy.init(args=args)
    node = UsbCameraPublisher(camera_index=0, camera_name='left_wrist_camera',
                              debug=False,
                              recorded=False,
                              camera_type='USB',
                              video_path="data/usb_camera_video_v1/video_001.mp4")

    try:
        rclpy.spin(node)
    except IndentationError as e:
        logger.exception(e)
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()