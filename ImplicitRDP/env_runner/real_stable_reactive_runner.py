import threading
import time
import os.path as osp
import numpy as np
import torch
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
    absolute_actions_to_relative_actions_batch_torch,
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


class RealStableReactiveRunner:
    def __init__(self,
                 output_dir: str,
                 transform_params: DictConfig,
                 env_params: DictConfig,
                 shape_meta: DictConfig,
                 tcp_ensemble_buffer_params: DictConfig,
                 gripper_ensemble_buffer_params: DictConfig,
                 action_type: str,
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
                 tqdm_interval_sec=5.0,
                 control_fps: float = 10,
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
                 **kwargs
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
        # for reactive runner, we use the same interval time for control and inference
        self.inference_interval_time = 1.0 / control_fps
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
            assert tcp_ensemble_buffer_params.ensemble_mode == 'new', "Only support new ensemble mode for reactive transformer."
            assert gripper_ensemble_buffer_params.ensemble_mode == 'new', "Only support new ensemble mode for reactive transformer."
            self.tcp_ensemble_buffer = EnsembleBuffer(**tcp_ensemble_buffer_params)
            self.gripper_ensemble_buffer = EnsembleBuffer(**gripper_ensemble_buffer_params)
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
            self.init_rerun_logging()

    @staticmethod
    def spin_executor(executor):
        executor.spin()

    def init_rerun_logging(self):
        """Initialize rerun logging for visualization."""
        try:
            rr.init("ImplicitRDP_obs_visualization", spawn=True)
            logger.info("Rerun visualization initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize rerun: {e}")
            self.enable_rerun_visualization = False

    def visualize_dict_with_rerun(self, dictionary: Dict, prefix: str = "obs"):
        """Visualize observation data using rerun.

        Args:
            dictionary: Dictionary containing observation or action data
            prefix: Prefix for the rerun entity path
        """
        try:
            timestamp = time.time()
            rr.set_time(timeline="ImplicitRDP_obs_visualization", timestamp=timestamp)

            for key, value in dictionary.items():
                if value is None:
                    continue

                entity_path = f"{prefix}/{key}"

                # Handle different types of data
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
                    obs_dict[key] = absolute_actions_to_relative_actions(obs_dict[key],
                                                                         base_absolute_action=base_absolute_action)
        
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
                    extended_obs_dict[key] = absolute_actions_to_relative_actions(extended_obs_dict[key],
                                                                                  base_absolute_action=base_absolute_action)

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
                left_euler_batch = np.array(
                    [t3d.euler.mat2euler(rot_mat) for rot_mat in left_rot_mat_batch])  # (action_steps, 3)
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
            left_action_6d[:, :3] = np.clip(left_action_6d[:, :3], np.array(self.tcp_pos_clip_range[clip_idx_base]),
                                            np.array(self.tcp_pos_clip_range[clip_idx_base + 1]))
            clip_idx_base += 2
        if right_action_6d is not None:
            right_action_6d[:, :3] = np.clip(right_action_6d[:, :3], np.array(self.tcp_pos_clip_range[clip_idx_base]),
                                             np.array(self.tcp_pos_clip_range[clip_idx_base + 1]))
        # clip action (r, p, y)
        clip_idx_base = 0
        if left_action_6d is not None:
            left_action_6d[:, 3:] = np.clip(left_action_6d[:, 3:], np.array(self.tcp_rot_clip_range[clip_idx_base]),
                                            np.array(self.tcp_rot_clip_range[clip_idx_base + 1]))
            clip_idx_base += 2
        if right_action_6d is not None:
            right_action_6d[:, 3:] = np.clip(right_action_6d[:, 3:], np.array(self.tcp_rot_clip_range[clip_idx_base]),
                                             np.array(self.tcp_rot_clip_range[clip_idx_base + 1]))
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
    
    def action_command_thread(self, stop_event):
        prev_time = time.time()
        frame_count = 0
        while not stop_event.is_set():
            start_time = time.time()
            # get step action from ensemble buffer
            tcp_step_action = self.tcp_ensemble_buffer.get_action()
            gripper_step_action = self.gripper_ensemble_buffer.get_action()
            if tcp_step_action is None or gripper_step_action is None:  # no action in the buffer => no movement.
                cur_time = time.time()
                precise_sleep(max(0., self.control_interval_time - (cur_time - start_time)))
                logger.debug(f"Step: {self.action_step_count}, control_interval_time: {self.control_interval_time}, "
                             f"cur_time-start_time: {cur_time - start_time}")
                self.action_step_count += 1
                continue

            if self.action_type == ActionType.left_arm_3D_translation_gripper_width or \
                    self.action_type == ActionType.right_arm_3D_translation: # (x, y, z, gripper_width)
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

            combined_action = np.concatenate([tcp_step_action, gripper_step_action], axis=-1)
            step_action = self.post_process_action(combined_action[np.newaxis, :])
            step_action = step_action.squeeze(0)

            # Visualize action if rerun visualization is enabled
            if self.enable_rerun_visualization:
                action_dict = {
                    'tcp_action': step_action[None, :tcp_len],
                    'gripper_action': step_action[None, tcp_len:],
                }
                self.visualize_dict_with_rerun(action_dict, prefix="action")

            # Send action to the robot using ActionType
            print(f"Executing action at step {self.action_step_count}: {step_action}")
            self.env.execute_action(step_action, self.action_type, use_relative_action=False)

            cur_time = time.time()
            precise_sleep(max(0., self.control_interval_time - (cur_time - start_time)))
            self.action_step_count += 1

            frame_count += 1
            elapsed_time = cur_time - prev_time
            frame_rate = frame_count / elapsed_time
            logger.debug(f"Real control rate: {frame_rate:.2f} FPS")
            if elapsed_time >= 1.0:
                prev_time = cur_time
                frame_count = 0

    def start_record_video(self, video_path):
        for vcamera_server_ip, vcamera_server_port in zip(self.vcamera_server_ip_list, self.vcamera_server_port_list):
            response = self.session.post(
                f'http://{vcamera_server_ip}:{vcamera_server_port}/start_recording/{video_path}')
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

    def run(self,
            policy: Union[DiffusionUnetImagePolicy, DiffusionTransformerImagePolicy]):
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
                reset_flag = py_cli_interaction.parse_cli_bool('Has the environment reset finished?',
                                                               default_value=True)
                if not reset_flag:
                    logger.warning("Skip this episode.")
                    continue

                logger.info("Start episode rollout.")
                # start rollout
                self.env.reset()
                # set gripper to max width
                self.env.send_gripper_command_direct(self.env.max_gripper_width, self.env.max_gripper_width,
                                                     self.action_type)
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
                action_thread = threading.Thread(target=self.action_command_thread, args=(self.stop_event,),
                                                 daemon=True)
                action_thread.start()

                self.action_step_count = 0
                step_count = 0
                steps_per_inference = 1
                start_timestamp = time.time()

                # For reactive transformer, we need to maintain the following variables
                tcp_obs_tokens = None
                tcp_noisy_trajectory = None
                tcp_latent_action = None
                tcp_base_absolute_action = None
                tcp_action = None
                gripper_obs_tokens = None
                gripper_noisy_trajectory = None
                gripper_latent_action = None
                gripper_base_absolute_action = None
                gripper_action = None
                extended_obs = None
                dataset_obs_temporal_downsample_ratio = self.dataset_obs_temporal_downsample_ratio

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
                        if self.use_latent_action_with_rnn_decoder or self.use_reactive_transformer:
                            update_tcp = (step_count % self.tcp_action_update_interval == 0)
                            update_gripper = (step_count % self.gripper_action_update_interval == 0)
                            if update_tcp or update_gripper:
                                if self.use_latent_action_with_rnn_decoder:
                                    with torch.no_grad():
                                        latent_action = policy.predict_action(obs_dict,
                                                                            dataset_obs_temporal_downsample_ratio=self.dataset_obs_temporal_downsample_ratio,
                                                                            return_latent_action=True)['action'][:, 0, :]
                                    if update_tcp:
                                        tcp_latent_action = latent_action
                                        tcp_base_absolute_action = base_absolute_action
                                    if update_gripper:
                                        gripper_latent_action = latent_action
                                        gripper_base_absolute_action = base_absolute_action
                                if self.use_reactive_transformer:
                                    with torch.no_grad():
                                        obs_tokens = policy.predict_obs_tokens(obs_dict)
                                        noisy_trajectory = policy.sample_noisy_trajectory()
                                    if update_tcp:
                                        tcp_obs_tokens = obs_tokens
                                        tcp_noisy_trajectory = noisy_trajectory
                                        tcp_base_absolute_action = base_absolute_action
                                    if update_gripper:
                                        gripper_obs_tokens = obs_tokens
                                        gripper_noisy_trajectory = noisy_trajectory
                                        gripper_base_absolute_action = base_absolute_action
                            tcp_extended_obs_last_step = self.latency_step + step_count % self.tcp_action_update_interval + self.n_obs_steps * self.dataset_obs_temporal_downsample_ratio
                            gripper_extended_obs_last_step = self.gripper_latency_step + step_count % self.gripper_action_update_interval + self.n_obs_steps * self.dataset_obs_temporal_downsample_ratio

                            if self.use_reactive_transformer:
                                obs_tokens_list = [tcp_obs_tokens, gripper_obs_tokens]
                                noisy_trajectory_list = [tcp_noisy_trajectory, gripper_noisy_trajectory]
                                obs_tokens_and_noisy_trajectory_list = [torch.cat([obs_tokens, noisy_trajectory], dim=-1) for obs_tokens, noisy_trajectory in zip(obs_tokens_list, noisy_trajectory_list)]
                                extended_obs_last_step_list = [tcp_extended_obs_last_step, gripper_extended_obs_last_step]

                            longer_extended_obs_last_step = max(tcp_extended_obs_last_step, gripper_extended_obs_last_step)
                            obs_temporal_downsample_ratio = self.obs_temporal_downsample_ratio if self.downsample_extended_obs else 1
                            if extended_obs is None:
                                extended_obs = self.env.get_obs(longer_extended_obs_last_step, temporal_downsample_ratio=obs_temporal_downsample_ratio)
                            else:
                                new_obs = self.env.get_obs(1, temporal_downsample_ratio=obs_temporal_downsample_ratio)
                                if self.enable_rerun_visualization:
                                    self.visualize_dict_with_rerun(new_obs, prefix="padding_extended_obs")
                                for key in extended_obs:
                                    extended_obs[key] = np.concatenate([extended_obs[key], new_obs[key]], axis=0)
                                    extended_obs[key] = extended_obs[key][-longer_extended_obs_last_step:]
                            # optimize performance
                            filtered_extended_obs = dict()
                            for key in extended_obs:
                                if key in self.shape_meta['extended_obs'].keys():
                                    filtered_extended_obs[key] = extended_obs[key]
                            extended_obs = filtered_extended_obs
                            
                            np_extended_obs_dict = deepcopy(extended_obs)
                            np_extended_obs_dict = get_real_obs_dict(
                                env_obs=np_extended_obs_dict, shape_meta=self.shape_meta, is_extended_obs=True)
                            if self.use_latent_action_with_rnn_decoder:
                                np_extended_obs_dict, _ = self.pre_process_extended_obs(np_extended_obs_dict)
                                extended_obs_dict = dict_apply(np_extended_obs_dict, lambda x: torch.from_numpy(x).unsqueeze(0))
                            if self.use_reactive_transformer:
                                np_tcp_extended_obs_dict, _ = self.pre_process_extended_obs(np_extended_obs_dict, tcp_extended_obs_last_step-self.n_obs_steps*self.dataset_obs_temporal_downsample_ratio+1)
                                tcp_extended_obs_dict = dict_apply(np_tcp_extended_obs_dict, lambda x: torch.from_numpy(x).unsqueeze(0).to(policy.device))
                                np_gripper_extended_obs_dict, _ = self.pre_process_extended_obs(np_extended_obs_dict, gripper_extended_obs_last_step-self.n_obs_steps*self.dataset_obs_temporal_downsample_ratio+1)
                                gripper_extended_obs_dict = dict_apply(np_gripper_extended_obs_dict, lambda x: torch.from_numpy(x).unsqueeze(0).to(policy.device))
                                extended_obs_dict_list = [tcp_extended_obs_dict, gripper_extended_obs_dict]

                            # call policy
                            if self.use_latent_action_with_rnn_decoder:
                                tcp_step_action = policy.predict_from_latent_action(tcp_latent_action, extended_obs_dict, tcp_extended_obs_last_step, dataset_obs_temporal_downsample_ratio)
                                gripper_step_action = policy.predict_from_latent_action(gripper_latent_action, extended_obs_dict, gripper_extended_obs_last_step, dataset_obs_temporal_downsample_ratio)
                                result = [tcp_step_action, gripper_step_action]
                            if self.use_reactive_transformer:
                                action_dicts = policy.predict_from_obs_tokens_and_noisy_trajectory(
                                    obs_tokens_and_noisy_trajectory_list,
                                    extended_obs_dict_list,
                                    extended_obs_last_step_list,
                                    dataset_obs_temporal_downsample_ratio
                                )
                            logger.debug(f"Policy inference time: {time.time() - slow_policy_time:.3f}s")

                            # update action
                            tcp_action = action_dicts[0]['action']
                            gripper_action = action_dicts[1]['action']
                            tcp_action = tcp_action[0]
                            tcp_action = tcp_action.detach().cpu().numpy()
                            gripper_action = gripper_action[0]
                            gripper_action = gripper_action.detach().cpu().numpy()
                            if self.use_rpy_for_rotation:
                                tcp_action = rpy_actions_to_matrix_actions(tcp_action, self.action_type)
                                gripper_action = rpy_actions_to_matrix_actions(gripper_action, self.action_type)
                            if self.use_relative_action:
                                tcp_action = relative_actions_to_absolute_actions(tcp_action, tcp_base_absolute_action)
                                gripper_action = relative_actions_to_absolute_actions(gripper_action, gripper_base_absolute_action)
                            
                            tcp_action = tcp_action[-1:, :]
                            self.tcp_ensemble_buffer.add_action(tcp_action, step_count)
                            
                            gripper_action = gripper_action[-1:, :]
                            self.gripper_ensemble_buffer.add_action(gripper_action, step_count)
                        else:
                            action_dict = policy.predict_action(obs_dict)

                            # device_transfer
                            np_action_dict = dict_apply(action_dict,
                                                        lambda x: x.detach().to('cpu').numpy())

                            action_all = np_action_dict['action'].squeeze(0)
                            logger.debug(f"Policy inference time: {time.time() - slow_policy_time:.3f}s")

                            if self.use_rpy_for_rotation:
                                action_all = rpy_actions_to_matrix_actions(action_all, self.action_type)
                            
                            if self.use_relative_action:
                                action_all = relative_actions_to_absolute_actions(action_all, base_absolute_action)

                            if self.action_interpolation_ratio > 1:
                                action_all = interpolate_actions_with_ratio(action_all, self.action_interpolation_ratio)

                            # TODO: only takes the first n_action_steps and add to the ensemble buffer
                            if step_count % self.tcp_action_update_interval == 0:
                                
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
                        if cur_time - start_time > self.inference_interval_time:
                            logger.warning(f"Slow inference step time: {cur_time - start_time:.3f}s, which exceeds the inference interval time {self.inference_interval_time}s.")
                        if cur_time - start_timestamp >= self.max_duration_time:
                            logger.info(
                                f"Episode {episode_idx} reaches max duration time {self.max_duration_time} seconds.")
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
