from pydantic import BaseModel, Field
import numpy as np
from enum import Enum, auto
from typing import List, final

class HandMes(BaseModel):
    wristPos: List[float]  # (x, y, z)
    wristQuat: List[float]  # (w, qx, qy, qz)
    triggerState:float
    # # for left controller (B, A, joystick, trigger, side_trigger)
    # # for right controller (Y, X, joystick, trigger, side_trigger)
    buttonState: List[bool]

class UnityMes(BaseModel):
    timestamp: float
    leftHand: HandMes
    rightHand: HandMes

class Arrow(BaseModel):
    start: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    end: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])

class TactileSensorMessage(BaseModel):
    device_id: str
    arrows: List[Arrow]
    scale: List[float] = Field(default_factory=lambda: [0.01, 0.005, 0.005])  # meters (sphere radius, arrow x scale, arrow y scale)

class ForceSensorMessage(BaseModel):
    device_id: str
    arrow: Arrow
    scale: List[float] = Field(default_factory=lambda: [0.01, 0.005, 0.005])  # meters (sphere radius, arrow x scale, arrow y scale)

class BimanualRobotStates(BaseModel):
    leftRobotTCP: List[float] = [0.0] * 7  # (7) (x, y, z, qw, qx, qy, qz)
    rightRobotTCP: List[float] = [0.0] * 7  # (7) (x, y, z, qw, qx, qy, qz)
    leftRobotTCPVel: List[float] = [0.0] * 6  # (6) (vx, vy, vz, wx, wy, wz)
    rightRobotTCPVel: List[float] = [0.0] * 6  # (6) (vx, vy, vz, wx, wy, wz)
    leftRobotTCPWrench: List[float] = [0.0] * 6  # (6) (fx, fy, fz, mx, my, mz)
    rightRobotTCPWrench: List[float] = [0.0] * 6  # (6) (fx, fy, fz, mx, my, mz)
    leftGripperState: List[float] = [0.0] * 2  # (2) (width, force)
    rightGripperState: List[float] = [0.0] * 2  # (2) (width, force)

class MoveGripperRequest(BaseModel):
    width: float = 0.05
    velocity: float = 10.0
    force_limit: float = 5.0

class TargetTCPRequest(BaseModel):
    target_tcp: List[float]  # (7) (x, y, z, qw, qx, qy, qz)

class ActionPrimitiveRequest(BaseModel):
    primitive_name: str
    input_params: dict = {}

class NominalStiffness(BaseModel):
    stiffness: List[float] = [10000.0, 10000.0, 10000.0, 1500.0, 1500.0, 1500.0]  # (6) (kx, ky, kz, krx, kry, krz)

class CartesianImpedanceRequest(BaseModel):
    stiffness: List[float] = [10000.0, 10000.0, 10000.0, 1500.0, 1500.0, 1500.0]  # (6) (kx, ky, kz, krx, kry, krz)
    damping: List[float] = [0.7] * 6  # (6) (dx, dy, dz, drx, dry, drz)
    source: str = "user" # "user" or "policy"

class SensorMessage(BaseModel):
    # TODO: adaptable for different dimensions, considering abolishing the 2-D version
    timestamp: float
    leftRobotTCP: np.ndarray = Field(default_factory=lambda: np.zeros((6, ), dtype=np.float32))  # (6) (x, y, z, r, p, y)
    rightRobotTCP: np.ndarray = Field(default_factory=lambda: np.zeros((6, ), dtype=np.float32))  # (6) (x, y, z, r, p, y)
    leftRobotTCPVel: np.ndarray = Field(default_factory=lambda: np.zeros((6, ), dtype=np.float32))  # (6) (vx, vy, vz, wx, wy, wz)
    rightRobotTCPVel: np.ndarray = Field(default_factory=lambda: np.zeros((6, ), dtype=np.float32))  # (6) (vx, vy, vz, wx, wy, wz)
    leftRobotTCPWrench: np.ndarray = Field(default_factory=lambda: np.zeros((6, ), dtype=np.float32))  # (6) (fx, fy, fz, mx, my, mz)
    rightRobotTCPWrench: np.ndarray = Field(default_factory=lambda: np.zeros((6, ), dtype=np.float32))  # (6) (fx, fy, fz, mx, my, mz)
    leftRobotGripperState: np.ndarray = Field(default_factory=lambda: np.zeros((2, ), dtype=np.float32))  # (2) gripper (width, force)
    rightRobotGripperState: np.ndarray = Field(default_factory=lambda: np.zeros((2, ), dtype=np.float32))  # (2) gripper (width, force)
    externalCameraPointCloud: np.ndarray = Field(default_factory=lambda: np.zeros((10, 6), dtype=np.float16)) # (N, 6) (x, y, z, r, g, b)
    externalCameraRGB: np.ndarray = Field(default_factory=lambda: np.zeros((48, 64, 3), dtype=np.uint8))  # (H, W, 3) (r, g, b)
    leftWristCameraPointCloud: np.ndarray = Field(default_factory=lambda: np.zeros((10, 6), dtype=np.float16))  # (N, 6) (x, y, z, r, g, b)
    leftWristCameraRGB: np.ndarray = Field(default_factory=lambda: np.zeros((48, 64, 3), dtype=np.uint8))  # (H, W, 3) (r, g, b)
    rightWristCameraPointCloud: np.ndarray = Field(default_factory=lambda: np.zeros((10, 6), dtype=np.float16))  # (N, 6) (x, y, z, r, g, b)
    rightWristCameraRGB: np.ndarray = Field(default_factory=lambda: np.zeros((48, 64, 3), dtype=np.uint8))  # (H, W, 3) (r, g, b)
    leftGripperCameraRGB1: np.ndarray = Field(
        default_factory=lambda: np.zeros((24, 32, 3), dtype=np.uint8))  # (H, W, 3) (r, g, b)
    leftGripperCameraMarker1: np.ndarray = Field(
        default_factory=lambda: np.zeros((63, 3), dtype=np.float32))  # (num_markers, 3)(x, y, z)
    leftGripperCameraMarkerOffset1: np.ndarray = Field(
        default_factory=lambda: np.zeros((63, 3), dtype=np.float32))  # (num_markers, 3)(x, y, z)
    leftGripperCameraRGB2: np.ndarray = Field(
        default_factory=lambda: np.zeros((24, 32, 3), dtype=np.uint8))  # (H, W, 3) (r, g, b)
    leftGripperCameraMarker2: np.ndarray = Field(
        default_factory=lambda: np.zeros((63, 3), dtype=np.float32))  # (num_markers, 3)(x, y, z)
    leftGripperCameraMarkerOffset2: np.ndarray = Field(
        default_factory=lambda: np.zeros((63, 3), dtype=np.float32))  # (num_markers, 2)(x, y,z)
    rightGripperCameraRGB1: np.ndarray = Field(default_factory=lambda: np.zeros((24, 32, 3), dtype=np.uint8))  # (H, W, 3) (r, g, b)
    rightGripperCameraMarker1: np.ndarray = Field(
        default_factory=lambda: np.zeros((63, 3), dtype=np.float32))  # (num_markers, 3)(x, y, z)
    rightGripperCameraMarkerOffset1: np.ndarray = Field(
        default_factory=lambda: np.zeros((63, 3), dtype=np.float32))  # (num_markers, 3)(x, y, z)
    rightGripperCameraRGB2: np.ndarray = Field(default_factory=lambda: np.zeros((24, 32, 3), dtype=np.uint8))  # (H, W, 3) (r, g, b)
    rightGripperCameraMarker2: np.ndarray = Field(
        default_factory=lambda: np.zeros((63, 3), dtype=np.float32))  # (num_markers, 3)(x, y, z)
    rightGripperCameraMarkerOffset2: np.ndarray = Field(
        default_factory=lambda: np.zeros((63, 3), dtype=np.float32))  # (num_markers, 3)(x, y, z)

    class Config:
        arbitrary_types_allowed = True

class SensorMessageList(BaseModel):
    sensorMessages: List[SensorMessage]

class SensorMode(Enum):
    single_arm_one_realsense_no_tactile = auto()
    single_arm_two_realsense_no_tactile = auto()
    single_arm_two_realsense_two_tactile = auto()
    dual_arm_two_realsense_four_tactile = auto()

class RobotControlMode(Enum):
    left_arm_6DOF = auto()
    left_arm_3D_translation = auto()
    left_arm_3D_translation_Y_rotation = auto()
    right_arm_6DOF = auto()
    right_arm_3D_translation = auto()
    dual_arm_3D_translation = auto()

class ActionType(Enum):
    left_arm_6DOF_gripper_width = auto()
    left_arm_3D_translation_gripper_width = auto()
    left_arm_6DOF_gripper_width_emb = auto()
    right_arm_6DOF = auto()
    right_arm_6DOF_virtual_target_stiffness = auto()
    right_arm_6DOF_wrench = auto()
    right_arm_3D_translation = auto()
    right_arm_3D_translation_virtual_target_stiffness = auto()
    dual_arm_3D_translation_gripper_width = auto()