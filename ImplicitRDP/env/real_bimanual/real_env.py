import threading
from multiprocessing import Process, Queue, Event
import queue

import cv2
import time
import torch
import numpy as np
import requests
from omegaconf import DictConfig
from copy import deepcopy
from typing import Union, List, Dict, Optional
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from message_filters import ApproximateTimeSynchronizer, Subscriber
import rclpy
from rclpy.executors import MultiThreadedExecutor

from loguru import logger
from ImplicitRDP.real_world.real_world_transforms import RealWorldTransforms
from ImplicitRDP.real_world.device_mapping.device_mapping_utils import get_topic_and_type
from ImplicitRDP.real_world.device_mapping.device_mapping_server import DeviceToTopic
from ImplicitRDP.real_world.ros_data_converter import ROS2DataConverter
from ImplicitRDP.common.data_models import SensorMessage, BimanualRobotStates, ActionType
from ImplicitRDP.common.time_utils import convert_ros_time_to_float
from ImplicitRDP.common.ring_buffer import RingBuffer
from ImplicitRDP.real_world.post_process_utils import DataPostProcessingManager
from ImplicitRDP.common.space_utils import (pose_6d_to_pose_7d, pose_6d_to_4x4matrix)

import pyinstrument

def stack_last_n_obs(all_obs, n_steps: int) -> Union[np.ndarray, torch.Tensor]:
    assert(len(all_obs) > 0)
    all_obs = list(all_obs)
    if isinstance(all_obs[0], np.ndarray):
        result = np.zeros((n_steps,) + all_obs[-1].shape,
            dtype=all_obs[-1].dtype)
        start_idx = -min(n_steps, len(all_obs))
        result[start_idx:] = np.array(all_obs[start_idx:])
        if n_steps > len(all_obs):
            # pad
            result[:start_idx] = result[start_idx]
    elif isinstance(all_obs[0], torch.Tensor):
        result = torch.zeros((n_steps,) + all_obs[-1].shape,
            dtype=all_obs[-1].dtype)
        start_idx = -min(n_steps, len(all_obs))
        result[start_idx:] = torch.stack(all_obs[start_idx:])
        if n_steps > len(all_obs):
            # pad
            result[:start_idx] = result[start_idx]
    elif isinstance(all_obs[0], bool):
        result = np.zeros((n_steps,), dtype=bool)
        start_idx = -min(n_steps, len(all_obs))
        result[start_idx:] = np.array(all_obs[start_idx:])
        if n_steps > len(all_obs):
            # pad
            result[:start_idx] = result[start_idx]
    else:
        raise RuntimeError(f'Unsupported obs type {type(all_obs[0])}')
    return result


class RealRobotEnvironment(Node):
    start_gripper_interval_control: bool = False
    gripper_interval_count: int = 0
    last_gripper_width_target: List[float] = [0.1, 0.1]
    def __init__(self,
                 robot_server_ip: str,
                 robot_server_port: int,
                 transforms: RealWorldTransforms,
                 device_mapping_server_ip: str,
                 device_mapping_server_port: int,
                 data_processing_params: DictConfig,
                 k_max_trans: float = 10000,
                 k_max_rot: float = 1500,
                 max_fps: int = 30,
                 # gripper control parameters
                 use_force_control_for_gripper: bool = True,
                 max_gripper_width: float = 0.05,
                 min_gripper_width: float = 0.,
                 grasp_force: float = 5.0,
                 enable_gripper_interval_control: bool = False,
                 gripper_control_time_interval: float = 60,
                 gripper_control_width_precision: float = 0.02,
                 gripper_width_threshold: float = 0.04,
                 enable_gripper_width_clipping: bool = True,
                 time_check: bool = False,
                 debug: bool = False):
        super().__init__('real_env')
        self.robot_server_ip = robot_server_ip
        self.robot_server_port = robot_server_port
        self.transforms = transforms
        self.k_max_trans = k_max_trans
        self.k_max_rot = k_max_rot
        self.max_fps = max_fps

        # gripper control parameters
        self.use_force_control_for_gripper = use_force_control_for_gripper
        self.max_gripper_width = max_gripper_width
        self.min_gripper_width = min_gripper_width
        self.grasp_force = grasp_force
        self.enable_gripper_interval_control = enable_gripper_interval_control
        self.gripper_control_time_interval = gripper_control_time_interval
        self.gripper_control_width_precision = gripper_control_width_precision
        self.gripper_width_threshold = gripper_width_threshold
        self.enable_gripper_width_clipping = enable_gripper_width_clipping

        self.data_processing_manager = DataPostProcessingManager(transforms,
                                                                 **data_processing_params)
        self.debug = debug
        self.subscribers = []
        self.obs_and_sensor_msg_buffer = RingBuffer(size=1024, fps=max_fps)

        self.mutex = threading.Lock()

        logger.debug("Initializing RealEnv node...")
        # Get device to topic mapping
        response = requests.get(
            f"http://{device_mapping_server_ip}:{device_mapping_server_port}/get_mapping")
        self.device_to_topic_mapping = DeviceToTopic.model_validate(response.json())

        subs_name_type = get_topic_and_type(self.device_to_topic_mapping)
        depth_camera_point_cloud_topic_names: List[Optional[str]] = [None, None, None]  # external, left wrist, right wrist
        depth_camera_rgb_topic_names: List[Optional[str]] = [None, None, None]  # external, left wrist, right wrist
        tactile_camera_rgb_topic_names: List[Optional[str]] = [None, None, None, None]  # left gripper1, left gripper2, right gripper1, right gripper2
        tactile_camera_marker_topic_names: List[Optional[str]] = [None, None, None, None]  # left gripper1, left gripper2, right gripper1, right gripper2

        for topic, msg_type in subs_name_type:
            if "depth/points" in topic:
                if "external_camera" in topic:
                    depth_camera_point_cloud_topic_names[0] = topic
                elif "left_wrist_camera" in topic:
                    depth_camera_point_cloud_topic_names[1] = topic
                elif "right_wrist_camera" in topic:
                    depth_camera_point_cloud_topic_names[2] = topic
            elif "color/image_raw" in topic:
                if "gripper_camera" in topic:
                    if "left_gripper_camera_1" in topic:
                        tactile_camera_rgb_topic_names[0] = topic
                    elif "left_gripper_camera_2" in topic:
                        tactile_camera_rgb_topic_names[1] = topic
                    elif "right_gripper_camera_1" in topic:
                        tactile_camera_rgb_topic_names[2] = topic
                    elif "right_gripper_camera_2" in topic:
                        tactile_camera_rgb_topic_names[3] = topic
                else:
                    if "external_camera" in topic:
                        depth_camera_rgb_topic_names[0] = topic
                    elif "left_wrist_camera" in topic:
                        depth_camera_rgb_topic_names[1] = topic
                    elif "right_wrist_camera" in topic:
                        depth_camera_rgb_topic_names[2] = topic
            elif "marker_offset/information" in topic:
                if "left_gripper_camera_1" in topic:
                    tactile_camera_marker_topic_names[0] = topic
                elif "left_gripper_camera_2" in topic:
                    tactile_camera_marker_topic_names[1] = topic
                elif "right_gripper_camera_1" in topic:
                    tactile_camera_marker_topic_names[2] = topic
                elif "right_gripper_camera_2" in topic:
                    tactile_camera_marker_topic_names[3] = topic

        self.time_check = time_check
        self.timestamps = {name: [] for name, _ in get_topic_and_type(self.device_to_topic_mapping)}
        # for calculating FPS
        self.prev_time = time.time()
        self.frame_count = 0

        if self.debug:
            logger.debug(f"Depth camera point cloud topic names: {depth_camera_point_cloud_topic_names}")
            logger.debug(f"Depth camera rgb topic names: {depth_camera_rgb_topic_names}")
            logger.debug(f"Tactile camera rgb topic names: {tactile_camera_rgb_topic_names}")
            logger.debug(f"Tactile camera marker topic names: {tactile_camera_marker_topic_names}")

        self.data_converter = ROS2DataConverter(self.transforms,
                                                depth_camera_point_cloud_topic_names,
                                                depth_camera_rgb_topic_names,
                                                tactile_camera_rgb_topic_names,
                                                tactile_camera_marker_topic_names,
                                                debug=self.debug)

        for name, msg_type in subs_name_type:
            self.subscribers.append(Subscriber(self, msg_type, name))
            logger.debug(f"Subscribed to topic: {name} with type: {msg_type}")

        # ApproximateTimeSynchronizer is used to synchronize multiple topics
        self.ts = ApproximateTimeSynchronizer(self.subscribers, queue_size=250, slop=1,
                                              allow_headerless=False)

        self.ts.registerCallback(self.callback)

        # Create a session with robot server
        self.session = requests.session()

    def send_command(self, endpoint: str, data: dict = None):
        url = f"http://{self.robot_server_ip}:{self.robot_server_port}{endpoint}"
        if 'get' in endpoint:
            response = self.session.get(url)
        else:
            response = self.session.post(url, json=data)
        response.raise_for_status()  # Raise an error for bad responses
        return response.json()

    # @pyinstrument.profile()
    def callback(self, *msgs):
        topic_dict = dict()
        for i, msg in enumerate(msgs):
            topic_name = self.subscribers[i].topic
            topic_dict[topic_name] = msg

        if self.time_check:
            # check the time differences across topics and interval between time stamps
            for i, msg in enumerate(msgs):
                topic_name = self.subscribers[i].topic
                self.timestamps[topic_name].append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)

        if self.debug:
        # if True:
            # calculate the lastest timestamp in the topic_dict
            latest_timestamp = max([msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9 for msg in msgs])
            # convert current time (ROS time) to python time
            current_timestamp = convert_ros_time_to_float(self.get_clock().now())
            # find out the latency compared to current time
            latency = current_timestamp - latest_timestamp
            # the latency is approximately 5ms - 10ms (~5000 points per pcd)
            # the latency is approximately 10ms - 26ms (without point cloud)
            logger.debug(f"Latency for time synchronizer: {latency:.4f} seconds")

        # this part takes about 10ms - 15ms for now (~5000 points per pcd)
        # this part takes about 3ms (with RGB only) for now
        sensor_msg: SensorMessage = self.data_converter.convert_all_data(topic_dict)

        # convert sensor msg to obs dict
        # this part takes about 2ms (without point cloud)
        obs_dict = self.data_processing_manager.convert_sensor_msg_to_obs_dict(sensor_msg)

        self.obs_and_sensor_msg_buffer.push((obs_dict, sensor_msg))

        # calculate fps
        self.frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - self.prev_time
        if elapsed_time >= 1.0:
            frame_rate = self.frame_count / elapsed_time
            logger.debug(f"Env update rate: {frame_rate:.2f} FPS")
            self.prev_time = current_time
            self.frame_count = 0
            if self.time_check:
                self.check_sync()
                self.check_timestamp()

    def check_sync(self):
        # Check and log timestamp differences across topics
        all_times = list(self.timestamps.values())
        if not all(all_times):
            return

        # Calculate time differences for each frame across topics
        for i in range(len(all_times[0])):
            max_diff = 0
            for j in range(len(all_times)):
                for k in range(j + 1, len(all_times)):
                    if i < len(all_times[j]) and i < len(all_times[k]):
                        time_diff = abs(all_times[j][i] - all_times[k][i])
                        max_diff = max(max_diff, time_diff)
            logger.info(f"Frame {i}: Maximum time difference across topics: {max_diff:.6f} seconds")

    def check_timestamp(self):
        # check the interval between different time stampss
        all_times = list(self.timestamps.values())
        if not all(all_times):
            return

        time_stamps = []
        for i in range(len(all_times[0])):
            timestamps_for_frame = []
            for j in range(len(all_times)):
                if i < len(all_times[j]):
                    timestamp = all_times[j][i]
                    timestamps_for_frame.append(timestamp)

            if timestamps_for_frame:
                mean_time_stamp = sum(timestamps_for_frame) / len(timestamps_for_frame)
                time_stamps.append(mean_time_stamp)
                logger.info(f"Frame {i}: Mean timestamp: {mean_time_stamp:.6f} seconds")


    def reset(self) -> None:
        self.start_gripper_interval_control = False
        self.obs_and_sensor_msg_buffer.reset()

    # @pyinstrument.profile()
    def get_obs(self,
                obs_steps: int = 2,
                temporal_downsample_ratio: int = 2,
                return_sensor_msg: bool = False
                ) -> Dict[str, np.ndarray]:
        """
        Get observations with temporal downsampling support.

        Args:
            obs_steps: The number of observations to stack.
            temporal_downsample_ratio: The ratio for temporal downsampling.
                For example, if ratio=2, it will sample every other observation.
        Returns:
            A dictionary containing stacked observations
        """
        # Get last n*ratio observations to ensure we have enough samples after downsampling
        last_n_sensor_msg_and_obs_list, _ = self.obs_and_sensor_msg_buffer.peek_last_n(
            obs_steps * temporal_downsample_ratio)  # newest to oldest

        result = dict()
        # Filter out None observations
        last_n_sensor_msg_and_obs_list = [sensor_msg_and_obs for sensor_msg_and_obs in last_n_sensor_msg_and_obs_list if sensor_msg_and_obs is not None]
        if len(last_n_sensor_msg_and_obs_list) == 0:
            return result

        # Apply temporal downsampling
        # If ratio=2, it will take every other observation: [0, 2, 4, ...]
        # If ratio=3, it will take every third observation: [0, 3, 6, ...]
        downsampled_sensor_msg_and_obs_list = last_n_sensor_msg_and_obs_list[::temporal_downsample_ratio]
        # Take only the last n_obs_steps observations after downsampling
        downsampled_sensor_msg_and_obs_list = downsampled_sensor_msg_and_obs_list[:obs_steps]

        # reverse the order to oldest to newest
        downsampled_sensor_msg_and_obs_list = downsampled_sensor_msg_and_obs_list[::-1]

        downsampled_obs_list, downsampled_sensor_msg_list = zip(*downsampled_sensor_msg_and_obs_list)

        # Stack observations for each key
        for key in downsampled_obs_list[0].keys():
            result[key] = stack_last_n_obs(
                [obs[key] for obs in downsampled_obs_list], obs_steps)

        # convert current time (ROS time) to python time
        current_timestamp = convert_ros_time_to_float(self.get_clock().now())
        # find out the latency compared to current time
        latency = current_timestamp - downsampled_obs_list[-1]['timestamp'][0]
        # the overall latency is approximately 20ms - 70ms (max 110ms) (~5000 points per pcd)
        logger.debug(f"Overall latency for get_obs() : {latency:.4f} seconds")

        if return_sensor_msg:
            return result, downsampled_sensor_msg_list

        return result

    def is_arm_enable(self, action_type: ActionType, robot_type: str) -> bool:
        """
        Check if the robot arm is enabled.
        :param robot_type: 'left' or 'right'
        :return: True if the arm is enabled, False otherwise
        """
        if robot_type == 'left':
            return 'left' in str(action_type) or 'dual' in str(action_type)
        elif robot_type == 'right':
            return 'right' in str(action_type) or 'dual' in str(action_type)
        else:
            raise ValueError("robot_type must be 'left' or 'right'")

    def send_gripper_command_direct(self, left_gripper_width_target: float, right_gripper_width_target: float, action_type: ActionType):
        """
        Send gripper command (width) directly to robot
        """
        if self.is_arm_enable(action_type, "left"):
            self.send_command('/move_gripper/left', {
                'width': left_gripper_width_target,
                'velocity': 10.0,
                'force_limit': self.grasp_force
            })
            self.last_gripper_width_target[0] = left_gripper_width_target
        if self.is_arm_enable(action_type, "right"):
            self.send_command('/move_gripper/right', {
                'width': right_gripper_width_target,
                'velocity': 10.0,
                'force_limit': self.grasp_force
            })
            self.last_gripper_width_target[1] = right_gripper_width_target

    def send_gripper_command(self, left_gripper_width_target: float, right_gripper_width_target: float, action_type: ActionType):
        if self.enable_gripper_interval_control and self.start_gripper_interval_control:
            self.gripper_interval_count += 1
            if self.gripper_interval_count % self.gripper_control_time_interval == 0:
                self.gripper_interval_count = 0

            if self.gripper_interval_count != 0:
                return

        if self.enable_gripper_width_clipping:
            if self.is_arm_enable(action_type, "left"):
                if left_gripper_width_target < self.gripper_width_threshold:
                    left_gripper_width_target = self.min_gripper_width
                    self.start_gripper_interval_control = True
            if self.is_arm_enable(action_type, "right"):
                if right_gripper_width_target < self.gripper_width_threshold:
                    right_gripper_width_target = self.min_gripper_width
                    self.start_gripper_interval_control = True
        else:
            self.start_gripper_interval_control = True

        robot_states = BimanualRobotStates.model_validate(self.send_command('/get_current_robot_states'))

        grasp_force = self.grasp_force
        gripper_control_width_precision = self.gripper_control_width_precision

        if self.is_arm_enable(action_type, "left"):
            left_current_width = robot_states.leftGripperState[0]
            if abs(self.last_gripper_width_target[0] - left_gripper_width_target) >= gripper_control_width_precision:
                if self.use_force_control_for_gripper and self.last_gripper_width_target[0] > left_gripper_width_target:
                    # try to close gripper with pure force control
                    logger.debug(f"left gripper moving from {left_current_width} to target: {left_gripper_width_target} "
                                 f"with force {grasp_force}")
                    self.send_command('/move_gripper_force/left', {
                        'force_limit': grasp_force
                    })
                else:
                    # open gripper with position control
                    logger.debug(f"left gripper moving from {left_current_width} to target: {left_gripper_width_target}")
                    self.send_command('/move_gripper/left', {
                        'width': left_gripper_width_target,
                        'velocity': 10.0,
                        'force_limit': grasp_force
                    })
                self.last_gripper_width_target[0] = left_gripper_width_target

        if self.is_arm_enable(action_type, "right"):
            right_current_width = robot_states.rightGripperState[0]
            if abs(self.last_gripper_width_target[1] - right_gripper_width_target) >= gripper_control_width_precision:
                if self.use_force_control_for_gripper and self.last_gripper_width_target[1] > right_gripper_width_target:
                    # try to close gripper with pure force control
                    logger.debug(f"right gripper moving from {right_current_width} to target: {right_gripper_width_target} "
                                 f"with force {grasp_force}")
                    self.send_command('/move_gripper_force/right', {
                        'force_limit': grasp_force
                    })
                else:
                    # open gripper with position control
                    logger.debug(f"right gripper moving from {right_current_width} to target: {right_gripper_width_target}")
                    self.send_command('/move_gripper/right', {
                        'width': right_gripper_width_target,
                        'velocity': 10.0,
                        'force_limit': grasp_force
                    })
                self.last_gripper_width_target[1] = right_gripper_width_target

    def execute_action(self, action: np.ndarray, action_type: ActionType, use_relative_action: bool = False) -> None:
        """
        Send action (in robot coordinate system) to robot
        :param action: np.ndarray, shape (16,) (left+right) (x, y, z, r, p, y, gripper_width)
        """
        if action_type == ActionType.right_arm_6DOF_virtual_target_stiffness:
            raise NotImplementedError(
                "right_arm_6DOF_virtual_target_stiffness is not supported for execution. "
                "Please use other action types."
            )
        
        left_action = action[:len(action) //2]
        right_action = action[len(action) //2:]

        # calculate target gripper width
        if use_relative_action:
            raise NotImplementedError
        else:
            if "gripper" in str(action_type):
                left_gripper_width_target = float(left_action[-2])
                right_gripper_width_target = float(right_action[-2])
                self.send_gripper_command(left_gripper_width_target, right_gripper_width_target, action_type)

        if use_relative_action:
            raise NotImplementedError
        else:
            left_tcp_target_6d_in_robot = left_action[:6]
            right_tcp_target_6d_in_robot = right_action[:6]

        if self.is_arm_enable(action_type, "left"):
            left_tcp_target_7d_in_robot = pose_6d_to_pose_7d(left_tcp_target_6d_in_robot)
            self.send_command('/move_tcp/left', {'target_tcp': left_tcp_target_7d_in_robot.tolist()})
        if self.is_arm_enable(action_type, "right"):
            right_tcp_target_7d_in_robot = pose_6d_to_pose_7d(right_tcp_target_6d_in_robot)
            self.send_command('/move_tcp/right', {'target_tcp': right_tcp_target_7d_in_robot.tolist()})


def env_process_worker(transforms, env_params, command_queue, response_queue, stop_event):
    """Worker function that runs the RealRobotEnvironment in a separate process"""
    try:
        # Initialize ROS2 in the new process
        rclpy.init(args=None)
        
        # Create transforms and environment
        env = RealRobotEnvironment(transforms=transforms, **env_params)
        
        # Create executor and add environment node
        executor = MultiThreadedExecutor()
        executor.add_node(env)
        
        # Start executor in a separate thread
        executor_thread = threading.Thread(target=executor.spin, daemon=True)
        executor_thread.start()
        
        logger.info("Environment process started successfully")
        
        # Main command processing loop
        while not stop_event.is_set():
            try:
                # Check for commands with timeout
                try:
                    command = command_queue.get(timeout=0.0001)
                except queue.Empty:
                    continue
                    
                method_name, args, kwargs = command
                
                if method_name == 'get_obs':
                    result = env.get_obs(*args, **kwargs)
                    response_queue.put(('success', result))
                elif method_name == 'execute_action':
                    env.execute_action(*args, **kwargs)
                    response_queue.put(('success', None))
                elif method_name == 'reset':
                    env.reset()
                    response_queue.put(('success', None))
                elif method_name == 'send_gripper_command_direct':
                    env.send_gripper_command_direct(*args, **kwargs)
                    response_queue.put(('success', None))
                elif method_name == 'send_command':
                    result = env.send_command(*args, **kwargs)
                    response_queue.put(('success', result))
                else:
                    response_queue.put(('error', f"Unknown method: {method_name}"))
                    
            except Exception as e:
                logger.error(f"Error in environment process: {e}")
                response_queue.put(('error', str(e)))
                
    except Exception as e:
        logger.error(f"Failed to initialize environment process: {e}")
        response_queue.put(('error', str(e)))
    finally:
        try:
            if 'env' in locals():
                env.destroy_node()
            if 'executor' in locals():
                executor.shutdown()
            rclpy.shutdown()
        except:
            pass