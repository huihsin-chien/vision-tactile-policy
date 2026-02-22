import threading
import time
import sys
import termios
import fcntl
import os
import requests
from typing import List, Optional
from loguru import logger
from ImplicitRDP.common.data_models import BimanualRobotStates, RobotControlMode, NominalStiffness
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
from ImplicitRDP.common.precise_sleep import precise_sleep
import numpy as np
from pynput import keyboard
from scipy.spatial.transform import Rotation, Slerp

class KineteachController:
    left_tracking_state: Optional[bool] = None
    right_tracking_state: Optional[bool] = None
    recorded_left_robot_tcp: Optional[List[float]] = None
    recorded_right_robot_tcp: Optional[List[float]] = None
    filtered_left_tcp: Optional[List[float]] = None
    filtered_right_tcp: Optional[List[float]] = None
    
    def __init__(self,
                 robot_server_ip: str,
                 robot_server_port: int,
                 host_ip: str = '192.168.2.187',
                 port: int = 8082,
                 fps: int = 100,
                 robot_control_mode: str = 'left_arm_6DOF',
                 use_external_control: bool = False,
                 stiffness_ratio: List[float] = [0.1, 0.1, 0.1, 0.005, 0.005, 0.005],
                 damping: List[float] = [0.6, 0.6, 0.6, 0.6, 0.6, 0.6],
                 filter_alpha: float = 0.5,
                 key_debounce_time: float = 0.5,
                 hold_to_activate: bool = True):
        self.robot_server_ip = robot_server_ip
        self.robot_server_port = robot_server_port
        self.host_ip = host_ip
        self.port = port
        self.robot_control_mode = RobotControlMode[robot_control_mode]
        self.fps = fps
        self.control_cycle_time = 1 / fps
        self.session = requests.session()

        self.use_external_control = use_external_control
        self.nominal_stiffness = self._get_nominal_stiffness(robot_control_mode)
        self.stiffness = [self.nominal_stiffness[i] * stiffness_ratio[i] for i in range(len(self.nominal_stiffness))]
        self.damping = list(damping)
        
        self.left_compliance_enabled = False
        self.right_compliance_enabled = False
        self._stop = False
        
        self.app = FastAPI()
        self.setup_routes()

        # Set initial tracking state
        self.left_tracking_state = False
        self.right_tracking_state = False

        # Set initial robot tcp
        if self.is_arm_enable('left'):
            self.recorded_left_robot_tcp = self._get_current_robot_states().leftRobotTCP
        if self.is_arm_enable('right'):
            self.recorded_right_robot_tcp = self._get_current_robot_states().rightRobotTCP

        # Low-pass filter parameters
        self.filter_alpha = filter_alpha
        self.filtered_left_tcp = self.recorded_left_robot_tcp.copy() if self.recorded_left_robot_tcp is not None else None
        self.filtered_right_tcp = self.recorded_right_robot_tcp.copy() if self.recorded_right_robot_tcp is not None else None
        
        # Add keyboard debounce related variables
        self.key_debounce_time = key_debounce_time
        self.last_key_press_time = {'pause': 0.0}
        # Hold-to-activate mode
        self.hold_to_activate = hold_to_activate
        self.key_held = False
        # Initialize keyboard listener
        self.keyboard_listener = None

    def setup_routes(self):
        @self.app.get('/get_controller_type')
        async def get_controller_type():
            return {"controller_type": "kineteach"}

        @self.app.get('/get_tracking_state')
        async def get_tracking_state():
            return {
                "left": self.left_tracking_state,
                "right": self.right_tracking_state
            }

        @self.app.post('/enable_external_control')
        async def enable_external_control():
            self.use_external_control = True
            logger.info("External control enabled")
            return {"message": "External control enabled"}

        @self.app.post('/disable_external_control')
        async def disable_external_control():
            self.use_external_control = False
            logger.info("External control disabled")
            return {"message": "External control disabled"}
    
    def run(self):
        kineteach_thread = threading.Thread(target=self.process_cmd, daemon=True)
        try:
            kineteach_thread.start()
            logger.info("Start Fast-API Kinematic Teaching Server!")
            
            def on_press(key):
                try:
                    # Check if pause key is pressed
                    if key == keyboard.Key.pause:
                        if self.hold_to_activate:
                            self.handle_key_hold('pause', True)
                        else:
                            self.handle_key_press('pause')
                except AttributeError:
                    pass
                    
            def on_release(key):
                try:
                    # Check if pause key is released
                    if key == keyboard.Key.pause:
                        if self.hold_to_activate:
                            self.handle_key_hold('pause', False)
                except AttributeError:
                    pass
                    
            self.keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
            self.keyboard_listener.daemon = True
            self.keyboard_listener.start()
            logger.info("Start Keyboard Listener!")
            uvicorn.run(self.app, host=self.host_ip, port=self.port)
            kineteach_thread.join()
        except Exception as e:
            logger.exception(e)
            self._stop = True
            if self.keyboard_listener:
                self.keyboard_listener.stop()
            raise e

    def _get_current_robot_states(self):
        url = f"http://{self.robot_server_ip}:{self.robot_server_port}/get_current_robot_states"
        response = self.session.get(url)
        response.raise_for_status()
        return BimanualRobotStates.model_validate(response.json())

    def _send_move_tcp(self, robot_side: str, tcp: List[float]):
        url = f"http://{self.robot_server_ip}:{self.robot_server_port}/move_tcp/{robot_side}"
        data = {"target_tcp": tcp}
        try:
            response = self.session.post(url, json=data, timeout=0.001)
        except requests.exceptions.ReadTimeout:
            # Ignore the timeout error for low-level control commands to reduce latency
            response = None
        if response is not None:
            response.raise_for_status()
            return response.json()
        else:
            return dict()

    def _get_nominal_stiffness(self, robot_control_mode: str) -> List[float]:
        robot_type = 'left' if 'left' in robot_control_mode or 'dual' in robot_control_mode else 'right'
        url = f"http://{self.robot_server_ip}:{self.robot_server_port}/get_nominal_stiffness/{robot_type}"
        response = self.session.get(url)
        response.raise_for_status()
        return NominalStiffness.model_validate(response.json()).stiffness

    def _set_cartesian_impedance(self, robot_side: str, stiffness: List[float], damping: List[float]):
        url = f"http://{self.robot_server_ip}:{self.robot_server_port}/set_cartesian_impedance/{robot_side}"
        data = {"stiffness": stiffness, "damping": damping, "source": "user"}
        try:
            response = self.session.post(url, json=data, timeout=0.001)
        except requests.exceptions.ReadTimeout:
            # Ignore the timeout error for low-level control commands to reduce latency
            response = None
        if response is not None:
            response.raise_for_status()
            return response.json()
        else:
            return dict()

    def _reset_cartesian_impedance(self, robot_side: str):
        url = f"http://{self.robot_server_ip}:{self.robot_server_port}/reset_cartesian_impedance/{robot_side}"
        try:
            response = self.session.post(url, timeout=0.001)
        except requests.exceptions.ReadTimeout:
            # Ignore the timeout error for low-level control commands to reduce latency
            response = None
        if response is not None:
            response.raise_for_status()
            return response.json()
        else:
            return dict()

    def is_arm_enable(self, robot_type: str) -> bool:
        """
        Check if the robot arm is enabled.
        :param robot_type: 'left' or 'right'
        :return: True if the arm is enabled, False otherwise
        """
        if robot_type == 'left':
            return 'left' in str(self.robot_control_mode) or 'dual' in str(self.robot_control_mode)
        elif robot_type == 'right':
            return 'right' in str(self.robot_control_mode) or 'dual' in str(self.robot_control_mode)
        else:
            raise ValueError("robot_type must be 'left' or 'right'")

    def get_adjusted_stiffness(self, robot_side: str) -> List[float]:
        """
        Adjust stiffness parameters based on robot control mode
        """
        adjusted_stiffness = self.stiffness.copy()
        
        if robot_side == 'left':
            if self.robot_control_mode == RobotControlMode.left_arm_3D_translation:
                # Only allow xyz translation, lock rotation
                adjusted_stiffness[3:] = self.nominal_stiffness[3:]
            elif self.robot_control_mode == RobotControlMode.left_arm_3D_translation_Y_rotation:
                # Allow xyz translation and y-axis rotation, lock other rotations
                adjusted_stiffness[3] = self.nominal_stiffness[3]  # Lock x-axis rotation
                adjusted_stiffness[5] = self.nominal_stiffness[5]  # Lock z-axis rotation
            elif self.robot_control_mode == RobotControlMode.dual_arm_3D_translation:
                # Only allow xyz translation, lock rotation
                adjusted_stiffness[3:] = self.nominal_stiffness[3:]
        elif robot_side == 'right':
            if self.robot_control_mode == RobotControlMode.right_arm_3D_translation:
                # Only allow xyz translation, lock rotation
                adjusted_stiffness[3:] = self.nominal_stiffness[3:]
            elif self.robot_control_mode == RobotControlMode.dual_arm_3D_translation:
                # Only allow xyz translation, lock rotation
                adjusted_stiffness[3:] = self.nominal_stiffness[3:]
                
        return adjusted_stiffness
    

    def apply_low_pass_filter(self, current_value: List[float], previous_value: List[float]) -> List[float]:
        """
        Apply low-pass filter to smooth TCP position data
        
        Args:
            current_value: Current TCP position (7D array: [x,y,z,qw,qx,qy,qz])
            previous_value: Previously filtered TCP position
            
        Returns:
            Filtered TCP position with linear interpolation for position and SLERP for quaternion
        """             
        filtered_value = []
        
        # Linear interpolation for position (first 3 elements)
        for i in range(3):
            filtered_value.append(self.filter_alpha * current_value[i] + (1 - self.filter_alpha) * previous_value[i])
        
        # Create a Slerp object
        times = [0, 1]
        rotations = Rotation.from_quat(np.vstack([
            [previous_value[4], previous_value[5], previous_value[6], previous_value[3]],  # scipy uses [x,y,z,w] format
            [current_value[4], current_value[5], current_value[6], current_value[3]]   # scipy uses [x,y,z,w] format
        ]))
        slerp = Slerp(times, rotations)
        
        # Interpolate at alpha
        interp_rot = slerp([self.filter_alpha])[0]
        
        # Convert back to [w,x,y,z] format
        interp_quat = interp_rot.as_quat()  # returns [x,y,z,w]
        result_q = [interp_quat[3], interp_quat[0], interp_quat[1], interp_quat[2]]  # convert to [w,x,y,z]
        
        # Append quaternion to result
        filtered_value.extend(result_q)
        
        return filtered_value
    
    def combine_tcp_by_mode(self, robot_side: str, current_tcp: List[float], stored_tcp: List[float]) -> List[float]:
        """
        Combine current TCP and stored TCP based on robot control mode
        
        Args:
            robot_side: 'left' or 'right'
            current_tcp: Current TCP from robot state
            stored_tcp: Stored TCP from previous state
            
        Returns:
            Combined TCP based on control mode
        """
        result_tcp = current_tcp.copy()
        
        if robot_side == 'left':
            if self.robot_control_mode == RobotControlMode.left_arm_6DOF:
                pass
            elif self.robot_control_mode == RobotControlMode.left_arm_3D_translation:
                # Only allow xyz translation, use stored rotation
                result_tcp[3:] = stored_tcp[3:]
            elif self.robot_control_mode == RobotControlMode.left_arm_3D_translation_Y_rotation:
                # Allow xyz translation and y-axis rotation, lock other rotations
                result_tcp[3] = stored_tcp[3]  # Use stored x-axis rotation
                result_tcp[5] = stored_tcp[5]  # Use stored z-axis rotation
            elif self.robot_control_mode == RobotControlMode.dual_arm_3D_translation:
                # Only allow xyz translation, use stored rotation
                result_tcp[3:] = stored_tcp[3:]
            else:
                raise ValueError(f"Unsupported robot control mode: {self.robot_control_mode}")

        elif robot_side == 'right':
            if self.robot_control_mode == RobotControlMode.right_arm_6DOF:
                pass
            elif self.robot_control_mode == RobotControlMode.right_arm_3D_translation:
                # Only allow xyz translation, use stored rotation
                result_tcp[3:] = stored_tcp[3:]
            elif self.robot_control_mode == RobotControlMode.dual_arm_3D_translation:
                # Only allow xyz translation, use stored rotation
                result_tcp[3:] = stored_tcp[3:]
            else:
                raise ValueError(f"Unsupported robot control mode: {self.robot_control_mode}")
                
        return result_tcp
    
    # Legacy method, no longer used
    def read_single_keypress(self):
        """Read a single keypress without requiring Pause key (legacy method)"""
        fd = sys.stdin.fileno()
        oldterm = termios.tcgetattr(fd)
        newattr = termios.tcgetattr(fd)
        newattr[3] = newattr[3] & ~termios.ICANON & ~termios.ECHO
        termios.tcsetattr(fd, termios.TCSANOW, newattr)
        oldflags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, oldflags | os.O_NONBLOCK)
        try:
            try:
                c = sys.stdin.read(1)
            except IOError:
                c = ''
        finally:
            termios.tcsetattr(fd, termios.TCSAFLUSH, oldterm)
            fcntl.fcntl(fd, fcntl.F_SETFL, oldflags)
        return c

    def handle_key_press(self, key: str):
        """Process key press with debounce functionality (toggle mode)"""
        # Only process Pause key
        if key != 'pause':
            return

        current_time = time.time()
        # Check if within debounce time
        if current_time - self.last_key_press_time.get('pause', 0) < self.key_debounce_time:
            return
            
        # Update last key press time
        self.last_key_press_time['pause'] = current_time
        
        # Toggle both arms' states if they are enabled
        if self.is_arm_enable('left'):
            if not self.left_tracking_state:
                logger.info("Enabling compliance control for left arm")
                adjusted_stiffness = self.get_adjusted_stiffness('left')
                self._set_cartesian_impedance('left', adjusted_stiffness, self.damping)
                self.left_tracking_state = True
                # update recorded tcp
                self.recorded_left_robot_tcp = self._get_current_robot_states().leftRobotTCP
                self.filtered_left_tcp = self.recorded_left_robot_tcp.copy()
            else:
                logger.info("Disabling compliance control for left arm")
                self._reset_cartesian_impedance('left')
                self.left_tracking_state = False
                # update recorded tcp
                # self.recorded_left_robot_tcp = self._get_current_robot_states().leftRobotTCP
                # self.filtered_left_tcp = self.recorded_left_robot_tcp.copy()
                
        if self.is_arm_enable('right'):
            if not self.right_tracking_state:
                logger.info("Enabling compliance control for right arm")
                adjusted_stiffness = self.get_adjusted_stiffness('right')
                self._set_cartesian_impedance('right', adjusted_stiffness, self.damping)
                self.right_tracking_state = True
                # update recorded tcp
                self.recorded_right_robot_tcp = self._get_current_robot_states().rightRobotTCP
                self.filtered_right_tcp = self.recorded_right_robot_tcp.copy()
            else:
                logger.info("Disabling compliance control for right arm")
                self._reset_cartesian_impedance('right')
                self.right_tracking_state = False
                # update recorded tcp
                # self.recorded_right_robot_tcp = self._get_current_robot_states().rightRobotTCP
                # self.filtered_right_tcp = self.recorded_right_robot_tcp.copy()
            
    def handle_key_hold(self, key: str, is_pressed: bool):
        """Process key hold events for hold-to-activate mode"""
        # Only process Pause key
        if key != 'pause':
            return
            
        # Update key state
        if is_pressed:
            # Key is pressed down
            if not self.key_held:
                self.key_held = True
                
                # Enable compliance control for both arms if they are enabled
                if self.is_arm_enable('left') and not self.left_tracking_state:
                    logger.info("Enabling compliance control for left arm")
                    adjusted_stiffness = self.get_adjusted_stiffness('left')
                    self._set_cartesian_impedance('left', adjusted_stiffness, self.damping)
                    self.left_tracking_state = True
                    # Update recorded TCP
                    self.recorded_left_robot_tcp = self._get_current_robot_states().leftRobotTCP
                    self.filtered_left_tcp = self.recorded_left_robot_tcp.copy()
                    
                if self.is_arm_enable('right') and not self.right_tracking_state:
                    logger.info("Enabling compliance control for right arm")
                    adjusted_stiffness = self.get_adjusted_stiffness('right')
                    self._set_cartesian_impedance('right', adjusted_stiffness, self.damping)
                    self.right_tracking_state = True
                    # Update recorded TCP
                    self.recorded_right_robot_tcp = self._get_current_robot_states().rightRobotTCP
                    self.filtered_right_tcp = self.recorded_right_robot_tcp.copy()
        else:
            # Key is released
            if self.key_held:
                self.key_held = False
                
                # Disable compliance control for both arms if they are enabled
                if self.is_arm_enable('left') and self.left_tracking_state:
                    logger.info("Disabling compliance control for left arm")
                    self._reset_cartesian_impedance('left')
                    self.left_tracking_state = False
                    # Update recorded TCP
                    # self.recorded_left_robot_tcp = self._get_current_robot_states().leftRobotTCP
                    # self.filtered_left_tcp = self.recorded_left_robot_tcp.copy()
                    
                if self.is_arm_enable('right') and self.right_tracking_state:
                    logger.info("Disabling compliance control for right arm")
                    self._reset_cartesian_impedance('right')
                    self.right_tracking_state = False
                    # Update recorded TCP
                    # self.recorded_right_robot_tcp = self._get_current_robot_states().rightRobotTCP
                    # self.filtered_right_tcp = self.recorded_right_robot_tcp.copy()

    def process_cmd(self):
        while not self._stop:
            start_time = time.time()
            
            # Legacy method, no longer used
            # key = self.read_single_keypress()
            # if key:
            #     self.handle_key_press(key)
            
            # Get current robot states
            try:
                robot_states = self._get_current_robot_states()
                
                # Process left arm movement
                if not self.use_external_control and self.is_arm_enable('left'):
                    if self.left_tracking_state:
                        current_tcp = robot_states.leftRobotTCP
                        self.filtered_left_tcp = self.apply_low_pass_filter(current_tcp, self.filtered_left_tcp)
                        left_tcp = self.combine_tcp_by_mode('left', self.filtered_left_tcp, self.recorded_left_robot_tcp)
                        self._send_move_tcp('left', left_tcp)
                    # else:
                    #     self._send_move_tcp('left', self.recorded_left_robot_tcp)
                
                # Process right arm movement
                if not self.use_external_control and self.is_arm_enable('right'):
                    if self.right_tracking_state:
                        current_tcp = robot_states.rightRobotTCP
                        self.filtered_right_tcp = self.apply_low_pass_filter(current_tcp, self.filtered_right_tcp)
                        right_tcp = self.combine_tcp_by_mode('right', self.filtered_right_tcp, self.recorded_right_robot_tcp)
                        self._send_move_tcp('right', right_tcp)
                    # else:
                    #     self._send_move_tcp('right', self.recorded_right_robot_tcp)
            
            except Exception as e:
                logger.exception(e)
            
            # Control loop frequency
            end_time = time.time()
            precise_sleep(self.control_cycle_time - (end_time - start_time))

    def __del__(self):
        """Clean up resources when the object is destroyed"""
        if self.keyboard_listener:
            self.keyboard_listener.stop()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Kinematic Teaching Controller')
    parser.add_argument('--robot_ip', type=str, default='192.168.2.242', help='Robot server IP address')
    parser.add_argument('--robot_port', type=int, default=8092, help='Robot server port')
    parser.add_argument('--host_ip', type=str, default='192.168.2.242', help='Host IP address')
    parser.add_argument('--port', type=int, default=8102, help='Host port')
    parser.add_argument('--fps', type=int, default=100, help='Control loop frequency')
    parser.add_argument('--mode', type=str, default='right_arm_6DOF', 
                        choices=['left_arm_6DOF', 'left_arm_3D_translation', 'left_arm_3D_translation_Y_rotation', 
                                'right_arm_6DOF', 'right_arm_3D_translation', 'dual_arm_6DOF', 'dual_arm_3D_translation'],
                        help='Robot control mode')
    parser.add_argument('--use_external_control', action='store_true', default=False,
                        help='Use external control for robot arms (default: False)')
    parser.add_argument('--filter_alpha', type=float, default=1.0,
                        help='Low-pass filter alpha parameter (0-1, lower means more smoothing)')
    parser.add_argument('--hold_to_activate', action='store_true', default=False,
                        help='Hold key to activate compliance control, release to deactivate (default: False)')
    args = parser.parse_args()
    
    # Default impedance parameters
    stiffness_ratio = [0.1, 0.1, 0.1, 0.005, 0.005, 0.005]
    damping = [0.6, 0.6, 0.6, 0.6, 0.6, 0.6]
    
    logger.info(f"Starting Kineteach Controller, Robot Server: {args.robot_ip}:{args.robot_port}")
    logger.info(f"Control Mode: {args.mode}, FPS: {args.fps}, Filter Alpha: {args.filter_alpha}")
    logger.info(f"Hold-to-activate mode: {args.hold_to_activate}")
    
    controller = KineteachController(
        robot_server_ip=args.robot_ip,
        robot_server_port=args.robot_port,
        host_ip=args.host_ip,
        port=args.port,
        fps=args.fps,
        robot_control_mode=args.mode,
        use_external_control=args.use_external_control,
        stiffness_ratio=stiffness_ratio,
        damping=damping,
        filter_alpha=args.filter_alpha,
        hold_to_activate=args.hold_to_activate
    )
    
    try:
        if args.hold_to_activate:
            logger.info("Hold Pause key to enable compliance control, release to disable")
        else:
            logger.info("Press Pause key to toggle robot compliance control state")
        logger.info("Global keyboard listener is active - no terminal focus required")
        controller.run()
    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
    except Exception as e:
        logger.exception("Program encountered an error")
    finally:
        logger.info("Program terminated")