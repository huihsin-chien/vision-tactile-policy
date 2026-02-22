import threading
import time
import os.path as osp
import numpy as np
import torch
import pickle
import tqdm
from loguru import logger
from typing import Dict, Tuple, Union, Optional
import rclpy
import transforms3d as t3d
import py_cli_interaction
from rclpy.executors import MultiThreadedExecutor
from omegaconf import DictConfig, ListConfig
from ImplicitRDP.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from ImplicitRDP.policy.diffusion_transformer_image_policy import DiffusionTransformerImagePolicy
from ImplicitRDP.common.pytorch_util import dict_apply
from ImplicitRDP.common.precise_sleep import precise_sleep
from ImplicitRDP.env.real_bimanual.real_env import RealRobotEnvironment
from ImplicitRDP.real_world.real_inference_util import (
    get_real_obs_dict)
from ImplicitRDP.real_world.real_world_transforms import RealWorldTransforms
from ImplicitRDP.common.space_utils import ortho6d_to_rotation_matrix_batch
from ImplicitRDP.common.ensemble import EnsembleBuffer
from ImplicitRDP.common.action_utils import (
    interpolate_actions_with_ratio,
    relative_actions_to_absolute_actions,
    absolute_actions_to_relative_actions,
    matrix_actions_to_rpy_actions,
    rpy_actions_to_matrix_actions,
    get_inter_gripper_actions
)
from ImplicitRDP.common.data_models import ActionType
import requests

# for visualization debug
import rerun as rr

import os
import psutil
from copy import deepcopy

# add this to prevent assigning too may threads when using numpy
os.environ["OPENBLAS_NUM_THREADS"] = "12"
os.environ["MKL_NUM_THREADS"] = "12"
os.environ["NUMEXPR_NUM_THREADS"] = "12"
os.environ["OMP_NUM_THREADS"] = "12"

import cv2
# add this to prevent assigning too may threads when using open-cv
cv2.setNumThreads(12)

# Get the total number of CPU cores
total_cores = psutil.cpu_count()
# Define the number of cores you want to bind to
num_cores_to_bind = 24
# Calculate the indices of the first ten cores
# Ensure the number of cores to bind does not exceed the total number of cores
cores_to_bind = set(range(min(num_cores_to_bind, total_cores)))
# Set CPU affinity for the current process to the first ten cores
os.sched_setaffinity(0, cores_to_bind)

class RealRunner:
    def __init__(self,
                 output_dir: str,
                 transform_params: DictConfig,
                 env_params: DictConfig,
                 shape_meta: DictConfig,
                 tcp_ensemble_buffer_params: DictConfig,
                 gripper_ensemble_buffer_params: DictConfig,
                 action_type: str,
                 latent_tcp_ensemble_buffer_params: DictConfig = DictConfig({"ensemble_mode": 'new'}),
                 latent_gripper_ensemble_buffer_params: DictConfig = DictConfig({"ensemble_mode": 'new'}),
                 use_latent_action_with_rnn_decoder: bool = False,
                 use_reactive_transformer: bool = False,
                 use_relative_action: bool = False,
                 use_relative_tcp_obs_for_relative_action: bool = True,
                 use_rpy_for_rotation: bool = False,
                 action_interpolation_ratio: int = 1,
                 eval_episodes=10,
                 max_duration_time: float = 30,
                 tcp_action_update_interval: int = 6,
                 gripper_action_update_interval: int = 10,
                 tcp_pos_clip_range: ListConfig = ListConfig([[0.6, -0.4, 0.03], [1.0, 0.45, 0.4]]),
                 tcp_rot_clip_range: ListConfig = ListConfig([[-np.pi, 0., np.pi], [-np.pi, 0., np.pi]]),
                 tqdm_interval_sec = 5.0,
                 control_fps: float = 10,
                 inference_fps: float = 5,
                 latency_step: int = 0,
                 gripper_latency_step: Optional[int] = None,
                 n_obs_steps: int = 2,
                 obs_temporal_downsample_ratio: int = 2,
                 dataset_obs_temporal_downsample_ratio: int = 1,
                 downsample_extended_obs: bool = True,
                 controller_server_ip: Optional[str] = None,
                 controller_server_port: Optional[int] = None,
                 enable_video_recording: bool = False,
                 vcamera_server_ip: Optional[Union[str, ListConfig]] = None,
                 vcamera_server_port: Optional[Union[int, ListConfig]] = None,
                 task_name=None,
                 enable_rerun_visualization: bool = False,
                 ):
        self.task_name = task_name
        self.transforms = RealWorldTransforms(option=transform_params)
        self.shape_meta = dict(shape_meta)
        self.eval_episodes = eval_episodes

        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys

        extended_rgb_keys = list()
        extended_lowdim_keys = list()
        extended_obs_shape_meta = shape_meta.get('extended_obs', dict())
        for key, attr in extended_obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                extended_rgb_keys.append(key)
            elif type == 'low_dim':
                extended_lowdim_keys.append(key)
        self.extended_rgb_keys = extended_rgb_keys
        self.extended_lowdim_keys = extended_lowdim_keys

        self.action_type: ActionType = ActionType[action_type]

        rclpy.init(args=None)
        self.env = RealRobotEnvironment(transforms=self.transforms, **env_params)
        # set gripper to max width
        self.env.send_gripper_command_direct(self.env.max_gripper_width, self.env.max_gripper_width, self.action_type)
        time.sleep(2)

        self.max_duration_time = max_duration_time
        self.tcp_action_update_interval = tcp_action_update_interval
        self.gripper_action_update_interval = gripper_action_update_interval
        self.tcp_pos_clip_range = tcp_pos_clip_range
        self.tcp_rot_clip_range = tcp_rot_clip_range
        self.tqdm_interval_sec = tqdm_interval_sec
        self.control_fps = control_fps
        self.control_interval_time = 1.0 / control_fps
        self.inference_fps = inference_fps
        self.inference_interval_time = 1.0 / inference_fps
        assert self.control_fps % self.inference_fps == 0, "Control FPS should be divisible by inference FPS."
        self.latency_step = latency_step
        self.gripper_latency_step = gripper_latency_step if gripper_latency_step is not None else latency_step
        self.n_obs_steps = n_obs_steps
        self.obs_temporal_downsample_ratio = obs_temporal_downsample_ratio
        self.dataset_obs_temporal_downsample_ratio = dataset_obs_temporal_downsample_ratio
        self.downsample_extended_obs = downsample_extended_obs
        self.use_latent_action_with_rnn_decoder = use_latent_action_with_rnn_decoder
        self.use_reactive_transformer = use_reactive_transformer
        if self.use_latent_action_with_rnn_decoder or self.use_reactive_transformer:
            assert not (self.use_latent_action_with_rnn_decoder and self.use_reactive_transformer), "Cannot use both latent action with RNN decoder and reactive transformer."
            assert latent_tcp_ensemble_buffer_params.ensemble_mode == 'new', "Only support new ensemble mode for latent action or reactive transformer."
            assert latent_gripper_ensemble_buffer_params.ensemble_mode == 'new', "Only support new ensemble mode for latent action or reactive transformer."
            self.tcp_ensemble_buffer = EnsembleBuffer(**latent_tcp_ensemble_buffer_params)
            self.gripper_ensemble_buffer = EnsembleBuffer(**latent_gripper_ensemble_buffer_params)
        else:
            self.tcp_ensemble_buffer = EnsembleBuffer(**tcp_ensemble_buffer_params)
            self.gripper_ensemble_buffer = EnsembleBuffer(**gripper_ensemble_buffer_params)

        self.use_relative_action = use_relative_action
        self.use_relative_tcp_obs_for_relative_action = use_relative_tcp_obs_for_relative_action
        self.use_rpy_for_rotation = use_rpy_for_rotation
        self.action_interpolation_ratio = action_interpolation_ratio

        # controller server
        self.controller_server_ip = controller_server_ip
        self.controller_server_port = controller_server_port

        # video recording
        self.enable_video_recording = enable_video_recording
        if enable_video_recording:
            assert isinstance(vcamera_server_ip, str) and isinstance(vcamera_server_port, int) or \
                     isinstance(vcamera_server_ip, ListConfig) and isinstance(vcamera_server_port, ListConfig), \
                "vcamera_server_ip and vcamera_server_port should be a string or ListConfig."
        if isinstance(vcamera_server_ip, str):
            vcamera_server_ip_list = [vcamera_server_ip]
            vcamera_server_port_list = [vcamera_server_port]
        elif isinstance(vcamera_server_ip, ListConfig):
            vcamera_server_ip_list = list(vcamera_server_ip)
            vcamera_server_port_list = list(vcamera_server_port)
        self.vcamera_server_ip_list = vcamera_server_ip_list
        self.vcamera_server_port_list = vcamera_server_port_list
        self.video_dir = osp.join(output_dir, 'videos')

        self.stop_event = threading.Event()
        self.session = requests.Session()

        # Rerun visualization settings
        self.enable_rerun_visualization = enable_rerun_visualization
        if self.enable_rerun_visualization:
            self._init_rerun_logging()

    @staticmethod
    def spin_executor(executor):
        executor.spin()

    def _init_rerun_logging(self):
        """Initialize rerun logging for visualization."""
        try:
            rr.init("ImplicitRDP_obs_visualization", spawn=True)
            logger.info("Rerun visualization initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize rerun: {e}")
            self.enable_rerun_visualization = False

    def _visualize_dict_with_rerun(self, obs_dict: Dict, prefix: str = "obs"):
        """Visualize observation data using rerun.

        Args:
            obs_dict: Dictionary containing observation data
            prefix: Prefix for the rerun entity path
        """
        try:
            timestamp = time.time()
            rr.set_time(timeline="ImplicitRDP_extended_obs_visualization", timestamp=timestamp)

            for key, value in obs_dict.items():
                if value is None:
                    continue
                
                entity_path = f"{prefix}/{key}"

                # Handle different types of observations
                if isinstance(value, np.ndarray):
                    assert value.shape[0] == 1, "Only support single step observation for rerun visualization."
                    value = value[0]  # Get the first step
                    if len(value.shape) == 3 and value.shape[-1] == 3:  # RGB images
                        rr.log(entity_path, rr.Image(value))
                    elif len(value.shape) == 1 and len(value) <= 15:
                        for i, v in enumerate(value):
                            rr.log(f"{entity_path}/{i}", rr.Scalars(float(v)))
                    else:
                        # General tensor logging
                        rr.log(entity_path, rr.Tensor(value))

                    # Also log some statistics for numerical data
                    if value.dtype.kind in 'biufc':  # numeric types
                        stats_path = f"{entity_path}_stats"
                        rr.log(f"{stats_path}/mean", rr.Scalars(float(np.mean(value))))
                        rr.log(f"{stats_path}/norm", rr.Scalars(float(np.linalg.norm(value))))

        except Exception as e:
            logger.error(f"Error in rerun visualization: {e}")

    def pre_process_obs(self, obs_dict: Dict) -> Tuple[Dict, Dict]:
        obs_dict = deepcopy(obs_dict)

        for key in self.lowdim_keys:
            if "wrt" not in key:
                if len(self.shape_meta['obs'][key]['shape']) == 1:
                    obs_dict[key] = obs_dict[key][:, :self.shape_meta['obs'][key]['shape'][0]]
                elif len(self.shape_meta['obs'][key]['shape']) == 2:
                    obs_dict[key] = obs_dict[key][:, :self.shape_meta['obs'][key]['shape'][0], :self.shape_meta['obs'][key]['shape'][1]]
                else:
                    raise ValueError(f"Only support 1d or 2d obs")

        # inter-gripper relative action
        obs_dict.update(get_inter_gripper_actions(obs_dict, self.lowdim_keys, self.transforms))
        for key in self.lowdim_keys:
            if len(self.shape_meta['obs'][key]['shape']) == 1:
                obs_dict[key] = obs_dict[key][:, :self.shape_meta['obs'][key]['shape'][0]]
            elif len(self.shape_meta['obs'][key]['shape']) == 2:
                obs_dict[key] = obs_dict[key][:, :self.shape_meta['obs'][key]['shape'][0], :self.shape_meta['obs'][key]['shape'][1]]
            else:
                raise ValueError(f"Only support 1d or 2d obs")

        absolute_obs_dict = dict()
        for key in self.lowdim_keys:
            absolute_obs_dict[key] = obs_dict[key].copy()

        # convert absolute action to relative action
        if self.use_relative_action and self.use_relative_tcp_obs_for_relative_action:
            for key in self.lowdim_keys:
                if 'robot_tcp_pose' in key and 'wrt' not in key:
                    base_absolute_action = obs_dict[key][-1].copy()
                    obs_dict[key] = absolute_actions_to_relative_actions(obs_dict[key], base_absolute_action=base_absolute_action)

        if self.use_rpy_for_rotation:
            for key in self.lowdim_keys:
                if 'robot_tcp_pose' in key:
                    obs_dict[key] = matrix_actions_to_rpy_actions(obs_dict[key])

        return obs_dict, absolute_obs_dict

    def pre_process_extended_obs(self, extended_obs_dict: Dict, base_action_last_step: Optional[int] = None) -> Tuple[Dict, Dict]:
        extended_obs_dict = deepcopy(extended_obs_dict)

        absolute_extended_obs_dict = dict()
        for key in self.extended_lowdim_keys:
            if len(self.shape_meta['extended_obs'][key]['shape']) == 1:
                extended_obs_dict[key] = extended_obs_dict[key][:, :self.shape_meta['extended_obs'][key]['shape'][0]]
            elif len(self.shape_meta['extended_obs'][key]['shape']) == 2:
                extended_obs_dict[key] = extended_obs_dict[key][:, :self.shape_meta['extended_obs'][key]['shape'][0], :self.shape_meta['extended_obs'][key]['shape'][1]]
            else:
                raise ValueError(f"Only support 1d or 2d extended obs")
            absolute_extended_obs_dict[key] = extended_obs_dict[key].copy()

        # convert absolute action to relative action
        if self.use_relative_action:
            for key in self.extended_lowdim_keys:
                if 'robot_tcp_pose' in key and 'wrt' not in key:
                    base_absolute_action = extended_obs_dict[key][-base_action_last_step].copy()
                    extended_obs_dict[key] = absolute_actions_to_relative_actions(extended_obs_dict[key], base_absolute_action=base_absolute_action)

        if self.use_rpy_for_rotation:
            for key in self.extended_lowdim_keys:
                if 'robot_tcp_pose' in key:
                    extended_obs_dict[key] = matrix_actions_to_rpy_actions(extended_obs_dict[key])

        return extended_obs_dict, absolute_extended_obs_dict

    def post_process_action(self, action: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Post-process the action before sending to the robot based on ActionType.
        """
        assert len(action.shape) == 2  # (action_steps, d_a)
        if self.env.data_processing_manager.use_6d_rotation:
            if self.action_type == ActionType.left_arm_3D_translation_gripper_width or \
                self.action_type == ActionType.dual_arm_3D_translation_gripper_width:
                # convert to 6D pose
                left_trans_batch = action[:, :3]  # (action_steps, 3)
                # we use default euler angles as 0
                left_euler_batch = np.zeros_like(left_trans_batch)
                left_action_6d = np.concatenate([left_trans_batch, left_euler_batch], axis=1)  # (action_steps, 6)
                if self.action_type == ActionType.dual_arm_3D_translation_gripper_width:
                    right_trans_batch = action[:, 3:6]  # (action_steps, 3)
                    right_euler_batch = np.zeros_like(right_trans_batch)
                    right_action_6d = np.concatenate([right_trans_batch, right_euler_batch], axis=1)
                else:
                    right_action_6d = None
            elif self.action_type == ActionType.left_arm_6DOF_gripper_width or \
                self.action_type == ActionType.left_arm_6DOF_gripper_width_emb:
                # convert to 6D pose
                left_rot_mat_batch = ortho6d_to_rotation_matrix_batch(action[:, 3:9])  # (action_steps, 3, 3)
                left_euler_batch = np.array([t3d.euler.mat2euler(rot_mat) for rot_mat in left_rot_mat_batch])  # (action_steps, 3)
                left_trans_batch = action[:, :3]  # (action_steps, 3)
                left_action_6d = np.concatenate([left_trans_batch, left_euler_batch], axis=1)  # (action_steps, 6)
                right_action_6d = None
            elif self.action_type == ActionType.right_arm_3D_translation:
                # convert to 6D pose
                right_trans_batch = action[:, :3]
                # we use default euler angles as 0
                right_euler_batch = np.zeros_like(right_trans_batch)
                right_action_6d = np.concatenate([right_trans_batch, right_euler_batch], axis=1)
                left_action_6d = None
            elif self.action_type == ActionType.right_arm_6DOF or \
                self.action_type == ActionType.right_arm_6DOF_wrench:
                # convert to 6D pose
                right_rot_mat_batch = ortho6d_to_rotation_matrix_batch(action[:, 3:9])
                right_euler_batch = np.array([t3d.euler.mat2euler(rot_mat) for rot_mat in right_rot_mat_batch])
                right_trans_batch = action[:, :3]
                right_action_6d = np.concatenate([right_trans_batch, right_euler_batch], axis=1)
                left_action_6d = None
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        # clip action (x, y, z)
        clip_idx_base = 0
        if left_action_6d is not None:
            left_action_6d[:, :3] = np.clip(left_action_6d[:, :3], np.array(self.tcp_pos_clip_range[clip_idx_base]), np.array(self.tcp_pos_clip_range[clip_idx_base+1]))
            clip_idx_base += 2
        if right_action_6d is not None:
            right_action_6d[:, :3] = np.clip(right_action_6d[:, :3], np.array(self.tcp_pos_clip_range[clip_idx_base]), np.array(self.tcp_pos_clip_range[clip_idx_base+1]))
        # clip action (r, p, y)
        clip_idx_base = 0
        if left_action_6d is not None:
            left_action_6d[:, 3:] = np.clip(left_action_6d[:, 3:], np.array(self.tcp_rot_clip_range[clip_idx_base]), np.array(self.tcp_rot_clip_range[clip_idx_base+1]))
            clip_idx_base += 2
        if right_action_6d is not None:
            right_action_6d[:, 3:] = np.clip(right_action_6d[:, 3:], np.array(self.tcp_rot_clip_range[clip_idx_base]), np.array(self.tcp_rot_clip_range[clip_idx_base+1]))
        # add gripper action
        if self.action_type == ActionType.left_arm_3D_translation_gripper_width:
            left_action = np.concatenate([left_action_6d, action[:, 3][:, np.newaxis],
                                          np.zeros((action.shape[0], 1))], axis=1)
            right_action = None
        elif self.action_type == ActionType.dual_arm_3D_translation_gripper_width:
            left_action = np.concatenate([left_action_6d, action[:, 6][:, np.newaxis],
                                          np.zeros((action.shape[0], 1))], axis=1)
            right_action = np.concatenate([right_action_6d, action[:, 7][:, np.newaxis],
                                           np.zeros((action.shape[0], 1))], axis=1)
        elif self.action_type == ActionType.left_arm_6DOF_gripper_width or \
            self.action_type == ActionType.left_arm_6DOF_gripper_width_emb:
            left_action = np.concatenate([left_action_6d, action[:, 9][:, np.newaxis],
                                          np.zeros((action.shape[0], 1))], axis=1)
            right_action = None
        elif self.action_type == ActionType.right_arm_6DOF or \
                self.action_type == ActionType.right_arm_6DOF_wrench or \
                self.action_type == ActionType.right_arm_3D_translation:
            right_action = right_action_6d
            left_action = None
        else:
            raise NotImplementedError

        if left_action is None:
            left_action = right_action.copy()
        if right_action is None:
            right_action = left_action.copy()
        action_all = np.concatenate([left_action, right_action], axis=-1)
        return action_all
    
    def is_kinematic_teaching_controlled(self):
        """
        Check if the environment is controlled by kinematic teaching controller.
        """
        if self.controller_server_ip is None or self.controller_server_port is None:
            return False
        try:
            response = self.session.get(f'http://{self.controller_server_ip}:{self.controller_server_port}/get_controller_type')
            controller_type = response.json()['controller_type']
            return controller_type == "kineteach"
        except requests.RequestException as e:
            logger.error(f"Error checking kinematic teaching control status: {e}")
            return False
    
    def enable_external_control(self):
        """
        Enable external control.
        """
        try:
            response = self.session.post(f'http://{self.controller_server_ip}:{self.controller_server_port}/enable_external_control')
            return response.json()['message']
        except requests.RequestException as e:
            logger.error(f"Error enabling external control: {e}")
            return None

    def disable_external_control(self):
        """
        Disable external control.
        """
        try:
            response = self.session.post(f'http://{self.controller_server_ip}:{self.controller_server_port}/disable_external_control')
            return response.json()['message']
        except requests.RequestException as e:
            logger.error(f"Error disabling external control: {e}")
            return None

    def action_command_thread(self, policy: Union[DiffusionUnetImagePolicy, DiffusionTransformerImagePolicy], stop_event):
        extended_obs = None
        prev_time = time.time()
        frame_count = 0
        last_tcp_extended_obs_step = np.inf
        last_gripper_extended_obs_step = np.inf
        
        execute_start_time = None
        while not stop_event.is_set():
            start_time = time.time()
            # get step action from ensemble buffer
            tcp_step_action = self.tcp_ensemble_buffer.get_action()
            gripper_step_action = self.gripper_ensemble_buffer.get_action()
            if tcp_step_action is None or gripper_step_action is None:  # no action in the buffer => no movement.
                cur_time = time.time()
                precise_sleep(max(0., self.control_interval_time - (cur_time - start_time)))
                logger.warning(f"No action in the buffer, sleep for {self.control_interval_time}s")
                continue
            else:
                self.action_step_count += 1
                if execute_start_time is None:
                    execute_start_time = time.time()
                desired_action_step = int((time.time() - execute_start_time) / self.control_interval_time) + 1
                while self.action_step_count < desired_action_step:
                    tcp_step_action = self.tcp_ensemble_buffer.get_action()
                    gripper_step_action = self.gripper_ensemble_buffer.get_action()
                    self.action_step_count += 1

            if self.action_type == ActionType.left_arm_3D_translation_gripper_width or \
                    self.action_type == ActionType.right_arm_3D_translation:  # (x, y, z, gripper_width)
                tcp_len = 3
            elif self.action_type == ActionType.left_arm_6DOF_gripper_width or \
                    self.action_type == ActionType.left_arm_6DOF_gripper_width_emb or \
                    self.action_type == ActionType.right_arm_6DOF or \
                    self.action_type == ActionType.right_arm_6DOF_wrench: # (x, y, z, rx1, rx2, rx3, ry1, ry2, ry3)
                tcp_len = 9
            elif self.action_type == ActionType.dual_arm_3D_translation_gripper_width: # (x_l, y_l, z_l, x_r, y_r, z_r, gripper_width_l, gripper_width_r)
                tcp_len = 6
            else:
                raise NotImplementedError

            if self.use_latent_action_with_rnn_decoder or self.use_reactive_transformer:
                tcp_extended_obs_step = int(tcp_step_action[-1])
                gripper_extended_obs_step = int(gripper_step_action[-1])
                tcp_step_action = tcp_step_action[:-1]
                gripper_step_action = gripper_step_action[:-1]

                # 1. get base absolute action
                if self.use_relative_action:
                    action_dim = 0
                    if 'left_robot_tcp_pose' in self.shape_meta['obs']:
                        action_dim += self.shape_meta['obs']['left_robot_tcp_pose']['shape'][0]
                    if 'right_robot_tcp_pose' in self.shape_meta['obs']:
                        action_dim += self.shape_meta['obs']['right_robot_tcp_pose']['shape'][0]
                    tcp_base_absolute_action = tcp_step_action[-action_dim:]
                    gripper_base_absolute_action = gripper_step_action[-action_dim:]
                    tcp_step_action = tcp_step_action[:-action_dim]
                    gripper_step_action = gripper_step_action[:-action_dim]

                tcp_step_latent_action = torch.from_numpy(tcp_step_action.astype(np.float32)).unsqueeze(0)
                gripper_step_latent_action = torch.from_numpy(gripper_step_action.astype(np.float32)).unsqueeze(0)

                # 2. get extended obs
                padding_extended_obs = None

                longer_extended_obs_step = max(tcp_extended_obs_step, gripper_extended_obs_step)
                obs_temporal_downsample_ratio = self.obs_temporal_downsample_ratio if self.downsample_extended_obs else 1
                if extended_obs is None or longer_extended_obs_step > len(next(iter(extended_obs.values()))) + 1:
                    result = self.env.get_obs(longer_extended_obs_step,
                                              temporal_downsample_ratio=obs_temporal_downsample_ratio)
                else:
                    # important: ensure adjacent obs are consistent
                    result = self.env.get_obs(1,
                                              temporal_downsample_ratio=obs_temporal_downsample_ratio)
                
                # optimize performance
                filtered_result = dict()
                for key in result.keys():
                    if key in self.shape_meta['extended_obs'].keys():
                        filtered_result[key] = result[key]
                result = filtered_result
                
                if extended_obs is None or longer_extended_obs_step > len(next(iter(extended_obs.values()))) + 1:
                    extended_obs = result
                else:
                    padding_extended_obs = result

                if padding_extended_obs is not None:
                    # Visualize padding_extended_obs if rerun visualization is enabled
                    if self.enable_rerun_visualization:
                        self._visualize_dict_with_rerun(padding_extended_obs, "padding_extended_obs")
                    
                    for key in padding_extended_obs.keys():
                        extended_obs[key] = np.concatenate([extended_obs[key], padding_extended_obs[key]], axis=0)
                        extended_obs[key] = extended_obs[key][-longer_extended_obs_step:]

                np_extended_obs_dict = deepcopy(extended_obs)
                np_extended_obs_dict = get_real_obs_dict(
                    env_obs=np_extended_obs_dict, shape_meta=self.shape_meta, is_extended_obs=True)
                if self.use_latent_action_with_rnn_decoder:
                    np_extended_obs_dict, _ = self.pre_process_extended_obs(np_extended_obs_dict)
                    extended_obs_dict = dict_apply(np_extended_obs_dict, lambda x: torch.from_numpy(x).unsqueeze(0))
                if self.use_reactive_transformer:
                    np_tcp_extended_obs_dict, _ = self.pre_process_extended_obs(np_extended_obs_dict, tcp_extended_obs_step-self.n_obs_steps*self.dataset_obs_temporal_downsample_ratio+1)
                    tcp_extended_obs_dict = dict_apply(np_tcp_extended_obs_dict, lambda x: torch.from_numpy(x).unsqueeze(0))
                    np_gripper_extended_obs_dict, _ = self.pre_process_extended_obs(np_extended_obs_dict, gripper_extended_obs_step-self.n_obs_steps*self.dataset_obs_temporal_downsample_ratio+1)
                    gripper_extended_obs_dict = dict_apply(np_gripper_extended_obs_dict, lambda x: torch.from_numpy(x).unsqueeze(0))

                # 3. get decoded action
                dataset_obs_temporal_downsample_ratio = self.dataset_obs_temporal_downsample_ratio
                fast_policy_time = time.time()
                with torch.no_grad():
                    if self.use_latent_action_with_rnn_decoder:
                        tcp_step_action = policy.predict_from_latent_action(tcp_step_latent_action, extended_obs_dict, tcp_extended_obs_step, dataset_obs_temporal_downsample_ratio)['action'][0].detach().cpu().numpy()
                        gripper_step_action = policy.predict_from_latent_action(gripper_step_latent_action, extended_obs_dict, gripper_extended_obs_step, dataset_obs_temporal_downsample_ratio)['action'][0].detach().cpu().numpy()
                    if self.use_reactive_transformer:
                        step_action_list = policy.predict_from_obs_tokens_and_noisy_trajectory(
                            [tcp_step_latent_action, gripper_step_latent_action],
                            [tcp_extended_obs_dict, gripper_extended_obs_dict],
                            [tcp_extended_obs_step, gripper_extended_obs_step],
                            dataset_obs_temporal_downsample_ratio)
                        tcp_step_action = step_action_list[0]['action'][0].detach().cpu().numpy()
                        gripper_step_action = step_action_list[1]['action'][0].detach().cpu().numpy()
                logger.debug(f"Fast policy inference time: {time.time() - fast_policy_time:.3f}s")

                if self.use_rpy_for_rotation:
                    tcp_step_action = rpy_actions_to_matrix_actions(tcp_step_action, self.action_type)
                    gripper_step_action = rpy_actions_to_matrix_actions(gripper_step_action, self.action_type)

                if self.use_relative_action:
                    tcp_step_action = relative_actions_to_absolute_actions(tcp_step_action, tcp_base_absolute_action)
                    gripper_step_action = relative_actions_to_absolute_actions(gripper_step_action, gripper_base_absolute_action)

                tcp_step_action = tcp_step_action[-1]
                gripper_step_action = gripper_step_action[-1]

                tcp_step_action = tcp_step_action[:tcp_len]
                gripper_step_action = gripper_step_action[tcp_len:]

            combined_action = np.concatenate([tcp_step_action, gripper_step_action], axis=-1)
            step_action = self.post_process_action(combined_action[np.newaxis, :])
            step_action = step_action.squeeze(0)

            # Visualize action if rerun visualization is enabled
            if self.enable_rerun_visualization:
                action_dict = {
                    'tcp_action': step_action[None, :tcp_len],
                    'gripper_action': step_action[None, tcp_len:],
                }
                self._visualize_dict_with_rerun(action_dict, "action")

            # Send action to the robot using ActionType
            self.env.execute_action(step_action, self.action_type, use_relative_action=False)
            
            cur_time = time.time()
            precise_sleep(max(0., self.control_interval_time - (cur_time - start_time)))
            if cur_time - start_time > self.control_interval_time:
                logger.warning(f"Fast control step time: {cur_time - start_time:.3f}s, which exceeds the control interval time {self.control_interval_time}s.")

            frame_count += 1
            elapsed_time = cur_time - prev_time
            frame_rate = frame_count / elapsed_time
            logger.debug(f"Real control rate: {frame_rate:.2f} FPS")
            if elapsed_time >= 1.0:
                prev_time = cur_time
                frame_count = 0


    def start_record_video(self, video_path):
        for vcamera_server_ip, vcamera_server_port in zip(self.vcamera_server_ip_list, self.vcamera_server_port_list):
            response = self.session.post(f'http://{vcamera_server_ip}:{vcamera_server_port}/start_recording/{video_path}')
            if response.status_code == 200:
                logger.info(f"Start recording video to {video_path}")
            else:
                logger.error(f"Failed to start recording video to {video_path}")

    def stop_record_video(self):
        for vcamera_server_ip, vcamera_server_port in zip(self.vcamera_server_ip_list, self.vcamera_server_port_list):
            response = self.session.post(f'http://{vcamera_server_ip}:{vcamera_server_port}/stop_recording')
            if response.status_code == 200:
                logger.info(f"Stop recording video")
            else:
                logger.error(f"Failed to stop recording video")

    def run(self, policy: Union[DiffusionUnetImagePolicy, DiffusionTransformerImagePolicy]):
        if self.use_latent_action_with_rnn_decoder:
            assert policy.at.use_rnn_decoder, "Policy should use rnn decoder for latent action."
        else:
            assert not hasattr(policy, 'at') or not policy.at.use_rnn_decoder, "Policy should not use rnn decoder for action."

        if self.use_reactive_transformer:
            assert policy.rnn_obs_encoder is not None, "Policy should use RNNObsEncoder for reactive transformer."
        else:
            assert not hasattr(policy, 'rnn_obs_encoder') or policy.rnn_obs_encoder is None, "Policy should not use RNNObsEncoder when not using reactive transformer."

        device = policy.device

        executor = MultiThreadedExecutor()
        executor.add_node(self.env)

        try:
            spin_thread = threading.Thread(target=self.spin_executor, args=(executor,), daemon=True)
            spin_thread.start()

            time.sleep(2)
            for episode_idx in tqdm.tqdm(range(0, self.eval_episodes),
                                         desc=f"Eval for {self.task_name}",
                                         leave=False, mininterval=self.tqdm_interval_sec):
                logger.info(f"Start evaluation episode {episode_idx}")
                # ask user whether the environment resetting is done
                reset_flag = py_cli_interaction.parse_cli_bool('Has the environment reset finished?', default_value=True)
                if not reset_flag:
                    logger.warning("Skip this episode.")
                    continue

                logger.info("Start episode rollout.")
                # start rollout
                self.env.reset()
                # set gripper to max width
                self.env.send_gripper_command_direct(self.env.max_gripper_width, self.env.max_gripper_width, self.action_type)
                time.sleep(1)

                policy.reset()

                self.tcp_ensemble_buffer.clear()
                self.gripper_ensemble_buffer.clear()
                logger.debug("Reset environment and policy.")

                if self.enable_video_recording:
                    video_path = os.path.join(self.video_dir, f'episode_{episode_idx}.mp4')
                    self.start_record_video(video_path)
                    logger.info(f"Start recording video to {video_path}")

                self.stop_event.clear()
                time.sleep(0.5)
                # start a new thread for action command
                action_thread = threading.Thread(target=self.action_command_thread, args=(policy, self.stop_event,),
                                                 daemon=True)
                action_thread.start()

                self.action_step_count = 0
                step_count = 0
                steps_per_inference = int(self.control_fps / self.inference_fps)
                start_timestamp = time.time()
                last_timestamp = start_timestamp

                if self.is_kinematic_teaching_controlled():
                    self.enable_external_control()

                try:
                    while True:
                        # profiler = Profiler()
                        # profiler.start()
                        start_time = time.time()
                        # get obs
                        obs = self.env.get_obs(
                            obs_steps=self.n_obs_steps,
                            temporal_downsample_ratio=self.obs_temporal_downsample_ratio)
                        # obs = dict()

                        if len(obs) == 0:
                            logger.warning("No observation received! Skip this step.")
                            cur_time = time.time()
                            precise_sleep(max(0., self.inference_interval_time - (cur_time - start_time)))
                            step_count += steps_per_inference
                            continue

                        # create obs dict
                        np_obs_dict = dict(obs)
                        # get transformed real obs dict
                        np_obs_dict = get_real_obs_dict(
                            env_obs=np_obs_dict, shape_meta=self.shape_meta)
                        np_obs_dict, np_absolute_obs_dict = self.pre_process_obs(np_obs_dict)

                        # device transfer
                        obs_dict = dict_apply(np_obs_dict,
                                              lambda x: torch.from_numpy(x).unsqueeze(0).to(
                                                  device=device))
                        base_absolute_action = np.concatenate([
                            np_absolute_obs_dict['left_robot_tcp_pose'][
                                -1] if 'left_robot_tcp_pose' in np_absolute_obs_dict else np.array([]),
                            np_absolute_obs_dict['right_robot_tcp_pose'][
                                -1] if 'right_robot_tcp_pose' in np_absolute_obs_dict else np.array([])
                        ], axis=-1)

                        slow_policy_time = time.time()
                        # run policy
                        with torch.no_grad():
                            if self.use_reactive_transformer:
                                obs_tokens = policy.predict_obs_tokens(obs_dict, return_expanded_obs_tokens=True)
                                noisy_trajectory = policy.sample_noisy_trajectory(return_expanded_noisy_trajectory=True)
                                action_all = torch.cat([obs_tokens, noisy_trajectory], dim=-1)
                                action_all = action_all.squeeze(0).detach().to('cpu').numpy()
                            else:
                                if self.use_latent_action_with_rnn_decoder:
                                    action_dict = policy.predict_action(obs_dict,
                                                                        dataset_obs_temporal_downsample_ratio=self.dataset_obs_temporal_downsample_ratio,
                                                                        return_latent_action=True)
                                else:
                                    action_dict = policy.predict_action(obs_dict)

                                # device_transfer
                                np_action_dict = dict_apply(action_dict,
                                                            lambda x: x.detach().to('cpu').numpy())

                                action_all = np_action_dict['action'].squeeze(0)
                        logger.debug(f"Slow policy inference time: {time.time() - slow_policy_time:.3f}s")

                        if not (self.use_latent_action_with_rnn_decoder or self.use_reactive_transformer):
                            if self.use_rpy_for_rotation:
                                action_all = rpy_actions_to_matrix_actions(action_all, self.action_type)

                        if self.use_latent_action_with_rnn_decoder or self.use_reactive_transformer:
                            # add first absolute action to get absolute action
                            if self.use_relative_action:
                                action_all = np.concatenate([
                                    action_all,
                                    base_absolute_action[np.newaxis, :].repeat(action_all.shape[0], axis=0)
                                ], axis=-1)
                            # add action step to get corresponding observation
                            action_all = np.concatenate([
                                action_all,
                                np.arange(self.n_obs_steps * self.dataset_obs_temporal_downsample_ratio, action_all.shape[0] + self.n_obs_steps * self.dataset_obs_temporal_downsample_ratio)[:, np.newaxis]
                            ], axis=-1)
                        else:
                            if self.use_relative_action:
                                action_all = relative_actions_to_absolute_actions(action_all, base_absolute_action)

                        if self.action_interpolation_ratio > 1:
                            if self.use_latent_action_with_rnn_decoder or self.use_reactive_transformer:
                                action_all = action_all.repeat(self.action_interpolation_ratio, axis=0)
                            else:
                                action_all = interpolate_actions_with_ratio(action_all, self.action_interpolation_ratio)

                        # TODO: only takes the first n_action_steps and add to the ensemble buffer
                        if step_count % self.tcp_action_update_interval == 0:
                            if self.use_latent_action_with_rnn_decoder or self.use_reactive_transformer:
                                tcp_action = action_all[self.latency_step:, ...]
                            else:
                                if self.action_type == ActionType.left_arm_3D_translation_gripper_width or \
                                        self.action_type == ActionType.right_arm_3D_translation:
                                    tcp_action = action_all[self.latency_step:, :3]
                                elif self.action_type == ActionType.left_arm_6DOF_gripper_width or \
                                        self.action_type == ActionType.left_arm_6DOF_gripper_width_emb or \
                                        self.action_type == ActionType.right_arm_6DOF or \
                                        self.action_type == ActionType.right_arm_6DOF_wrench:
                                    tcp_action = action_all[self.latency_step:, :9]
                                elif self.action_type == ActionType.dual_arm_3D_translation_gripper_width:
                                    tcp_action = action_all[self.latency_step:, :6]
                                else:
                                    raise NotImplementedError
                            # add to ensemble buffer
                            logger.debug(f"Step: {step_count}, Add TCP action to ensemble buffer: {tcp_action}")
                            self.tcp_ensemble_buffer.add_action(tcp_action, step_count)

                        if step_count % self.gripper_action_update_interval == 0:
                            if self.use_latent_action_with_rnn_decoder or self.use_reactive_transformer:
                                gripper_action = action_all[self.gripper_latency_step:, ...]
                            else:
                                if self.action_type == ActionType.left_arm_3D_translation_gripper_width or \
                                        self.action_type == ActionType.right_arm_3D_translation:
                                    gripper_action = action_all[self.gripper_latency_step:, 3:]
                                elif self.action_type == ActionType.left_arm_6DOF_gripper_width or \
                                        self.action_type == ActionType.left_arm_6DOF_gripper_width_emb or \
                                        self.action_type == ActionType.right_arm_6DOF or \
                                        self.action_type == ActionType.right_arm_6DOF_wrench:
                                    gripper_action = action_all[self.gripper_latency_step:, 9:]
                                elif self.action_type == ActionType.dual_arm_3D_translation_gripper_width:
                                    gripper_action = action_all[self.gripper_latency_step:, 6:]
                                else:
                                    raise NotImplementedError
                            # add to ensemble buffer
                            logger.debug(f"Step: {step_count}, Add gripper action to ensemble buffer: {gripper_action}")
                            self.gripper_ensemble_buffer.add_action(gripper_action, step_count)

                        cur_time = time.time()
                        precise_sleep(max(0., self.inference_interval_time - (cur_time - start_time)))
                        if cur_time - start_timestamp >= self.max_duration_time:
                            logger.info(f"Episode {episode_idx} reaches max duration time {self.max_duration_time} seconds.")
                            break
                        step_count += steps_per_inference
                        # profiler.stop()
                        # profiler.print()

                except KeyboardInterrupt:
                    logger.warning("KeyboardInterrupt! Terminate the episode now!")
                finally:
                    self.stop_event.set()
                    action_thread.join()
                    if self.enable_video_recording:
                        self.stop_record_video()

                    if self.is_kinematic_teaching_controlled():
                        self.disable_external_control()

            # TODO: support success count
            spin_thread.join()
        finally:
            self.env.destroy_node()
