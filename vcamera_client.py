import argparse
import numpy as np
import cv2
import requests
import time
from ImplicitRDP.common.precise_sleep import precise_sleep

class vCameraClient:
    def __init__(self, vcamera_server_ip, vcamera_server_port, fps=30, resize_factor=1.0):
        self.vcamera_server_ip = vcamera_server_ip
        self.vcamera_server_port = vcamera_server_port
        self.fps = fps
        self.display_interval_time = 1.0 / fps
        self.resize_factor = resize_factor
        self.session = requests.Session()

    def run(self):
        while True:
            start_t = time.time()
            response = self.session.get(f'http://{self.vcamera_server_ip}:{self.vcamera_server_port}/get_capture')
            if response.status_code == 200:
                if len(response.content) == 0:
                    # handle None
                    continue
                img = np.frombuffer(response.content, np.uint8)
                img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                if self.resize_factor != 1.0:
                    img = cv2.resize(img, (0, 0), fx=self.resize_factor, fy=self.resize_factor)
                cv2.imshow('Camera Streaming', img)
                cv2.waitKey(1)
            cur_time = time.time()
            precise_sleep(max(0., self.display_interval_time - (cur_time - start_t)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vcamera_server_ip', type=str, default='192.168.2.187')
    parser.add_argument('--vcamera_server_port', type=int, default=8102)
    parser.add_argument('--fps', type=int, default=60)
    parser.add_argument('--resize_factor', type=float, default=0.4)
    args = parser.parse_args()

    vcamera_client = vCameraClient(vcamera_server_ip=args.vcamera_server_ip,
                                   vcamera_server_port=args.vcamera_server_port,
                                   fps=args.fps,
                                   resize_factor=args.resize_factor)
    vcamera_client.run()