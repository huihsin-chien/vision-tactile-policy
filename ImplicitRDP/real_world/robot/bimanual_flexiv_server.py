import threading
from typing import List, Dict, Optional
import time
import uvicorn
from fastapi import FastAPI, HTTPException
from loguru import logger

from ImplicitRDP.real_world.robot.single_flexiv_controller import FlexivController
from ImplicitRDP.common.data_models import (BimanualRobotStates, MoveGripperRequest,
                                                          TargetTCPRequest, ActionPrimitiveRequest,
                                                          NominalStiffness, CartesianImpedanceRequest)

class BimanualFlexivServer():
    """
    Bimanual Flexiv Server Class
    Can be configured to use only left arm, only right arm, or both arms
    """
    # TODO: use UDP to respond
    def __init__(self,
                 left_robot_serial_number,
                 right_robot_serial_number,
                 left_gripper_name: Optional[str] = None,
                 right_gripper_name: Optional[str] = None,
                 host_ip="192.168.2.187",
                 port: int = 8092,
                 use_planner: bool = False,
                 robot_control_mode: str = 'left_arm_6DOF',
                 ) -> None:
        self.host_ip = host_ip
        self.port = port
        self.enable_left_arm = True if 'left' in robot_control_mode or 'dual' in robot_control_mode else False
        self.enable_right_arm = True if 'right' in robot_control_mode or 'dual' in robot_control_mode else False
        
        # Check if at least one arm is enabled
        if not (self.enable_left_arm or self.enable_right_arm):
            raise ValueError("At least one robot arm (left or right) must be enabled")
            
        # Initialize only enabled robot arms
        self.left_robot: Optional[FlexivController] = None
        self.right_robot: Optional[FlexivController] = None

        self.left_robot_impedance_control_status: Optional[str] = None
        self.right_robot_impedance_control_status: Optional[str] = None
        
        if self.enable_left_arm:
            logger.info(f"Initializing left robot with serial number {left_robot_serial_number}")
            self.left_robot = FlexivController(robot_serial_number=left_robot_serial_number,
                                               gripper_name=left_gripper_name)
            self.left_robot.robot.SwitchMode(self.left_robot.mode.NRT_CARTESIAN_MOTION_FORCE)
            # open the gripper
            self.left_robot.gripper.Move(0.1, 10, 0)
        
        if self.enable_right_arm:
            logger.info(f"Initializing right robot with serial number {right_robot_serial_number}")
            self.right_robot = FlexivController(robot_serial_number=right_robot_serial_number,
                                                gripper_name=right_gripper_name)
            self.right_robot.robot.SwitchMode(self.right_robot.mode.NRT_CARTESIAN_MOTION_FORCE)
            # open the gripper
            self.right_robot.gripper.Move(0.1, 10, 0)

        if use_planner and self.enable_left_arm and self.enable_right_arm:
            # TODO: support bimanual planner
            raise NotImplementedError
        else:
            self.planner = None

        self.app = FastAPI()
        # Start the receiving command thread
        self.setup_routes()
        
    def _get_default_robot_state(self):
        """Return default values for a disabled robot arm"""
        return type('obj', (object,), {
            'tcp_pose': [0.0] * 7,  # Default TCP pose (x, y, z, qx, qy, qz, qw)
            'tcp_vel': [0.0] * 6,   # Default TCP velocity (linear and angular)
            'ext_wrench_in_tcp': [0.0] * 6,  # Default wrench (force and torque)
        })
        
    def _get_default_gripper_state(self):
        """Return default values for a disabled gripper"""
        return type('obj', (object,), {
            'width': 0.0,
            'force': 0.0
        })

    def setup_routes(self):
        @self.app.post('/clear_fault')
        async def clear_fault() -> List[str]:
            fault_msgs = []
            
            if self.enable_left_arm and self.left_robot.robot.fault():
                logger.warning("Fault occurred on left robot server, trying to clear ...")
                thread_left = threading.Thread(target=self.left_robot.clear_fault)
                thread_left.start()
                thread_left.join()
                fault_msgs.append("Left robot fault cleared")
                
            if self.enable_right_arm and self.right_robot.robot.fault():
                logger.warning("Fault occurred on right robot server, trying to clear ...")
                thread_right = threading.Thread(target=self.right_robot.clear_fault)
                thread_right.start()
                thread_right.join()
                fault_msgs.append("Right robot fault cleared")
                
            return fault_msgs

        @self.app.get('/get_current_robot_states')
        async def get_current_robot_states() -> BimanualRobotStates:
            # Get states for enabled robots or use defaults for disabled ones
            if self.enable_left_arm:
                left_robot_state = self.left_robot.get_current_robot_states()
                left_robot_gripper_state = self.left_robot.get_current_gripper_states()
            else:
                left_robot_state = self._get_default_robot_state()
                left_robot_gripper_state = self._get_default_gripper_state()
                
            if self.enable_right_arm:
                right_robot_state = self.right_robot.get_current_robot_states()
                right_robot_gripper_state = self.right_robot.get_current_gripper_states()
            else:
                right_robot_state = self._get_default_robot_state()
                right_robot_gripper_state = self._get_default_gripper_state()
                
            return BimanualRobotStates(leftRobotTCP=left_robot_state.tcp_pose,
                                       rightRobotTCP=right_robot_state.tcp_pose,
                                       leftRobotTCPVel=left_robot_state.tcp_vel,
                                       rightRobotTCPVel=right_robot_state.tcp_vel,
                                       leftRobotTCPWrench=left_robot_state.ext_wrench_in_tcp,
                                       rightRobotTCPWrench=right_robot_state.ext_wrench_in_tcp,
                                       leftGripperState=[left_robot_gripper_state.width,
                                                            left_robot_gripper_state.force],
                                       rightGripperState=[right_robot_gripper_state.width,
                                                                 right_robot_gripper_state.force])

        @self.app.post('/move_gripper/{robot_side}')
        async def move_gripper(robot_side: str, request: MoveGripperRequest) -> Dict[str, str]:
            if robot_side not in ['left', 'right']:
                raise HTTPException(status_code=400, detail="Invalid robot side. Use 'left' or 'right'.")

            # Check if the requested robot is enabled
            if robot_side == 'left' and not self.enable_left_arm:
                raise HTTPException(status_code=400, detail="Left robot is not enabled.")
                
            if robot_side == 'right' and not self.enable_right_arm:
                raise HTTPException(status_code=400, detail="Right robot is not enabled.")

            robot_gripper = self.left_robot.gripper if robot_side == 'left' else self.right_robot.gripper
            robot_gripper.Move(request.width, request.velocity, request.force_limit)
            return {
                "message": f"{robot_side.capitalize()} gripper moving to width {request.width} "
                           f"with velocity {request.velocity} and force limit {request.force_limit}"}

        @self.app.post('/move_gripper_force/{robot_side}')
        async def move_gripper_force(robot_side: str, request: MoveGripperRequest) -> Dict[str, str]:
            if robot_side not in ['left', 'right']:
                raise HTTPException(status_code=400, detail="Invalid robot side. Use 'left' or 'right'.")

            # Check if the requested robot is enabled
            if robot_side == 'left' and not self.enable_left_arm:
                raise HTTPException(status_code=400, detail="Left robot is not enabled.")
                
            if robot_side == 'right' and not self.enable_right_arm:
                raise HTTPException(status_code=400, detail="Right robot is not enabled.")

            robot_gripper = self.left_robot.gripper if robot_side == 'left' else self.right_robot.gripper
            # use force control mode to grasp
            robot_gripper.Grasp(request.force_limit)
            return {
                "message": f"{robot_side.capitalize()} gripper grasp with force limit {request.force_limit}"}

        @self.app.post('/stop_gripper/{robot_side}')
        async def stop_gripper(robot_side: str) -> Dict[str, str]:
            if robot_side not in ['left', 'right']:
                raise HTTPException(status_code=400, detail="Invalid robot side. Use 'left' or 'right'.")

            # Check if the requested robot is enabled
            if robot_side == 'left' and not self.enable_left_arm:
                raise HTTPException(status_code=400, detail="Left robot is not enabled.")
                
            if robot_side == 'right' and not self.enable_right_arm:
                raise HTTPException(status_code=400, detail="Right robot is not enabled.")

            robot_gripper = self.left_robot.gripper if robot_side == 'left' else self.right_robot.gripper
            robot_gripper.Stop()
            return {"message": f"{robot_side.capitalize()} gripper stopping"}

        @self.app.post('/move_tcp/{robot_side}')
        async def move_tcp(robot_side: str, request: TargetTCPRequest) -> Dict[str, str]:
            if robot_side not in ['left', 'right']:
                raise HTTPException(status_code=400, detail="Invalid robot side. Use 'left' or 'right'.")

            # Check if the requested robot is enabled
            if robot_side == 'left' and not self.enable_left_arm:
                raise HTTPException(status_code=400, detail="Left robot is not enabled.")
                
            if robot_side == 'right' and not self.enable_right_arm:
                raise HTTPException(status_code=400, detail="Right robot is not enabled.")

            robot = self.left_robot if robot_side == 'left' else self.right_robot
            robot.tcp_move(request.target_tcp)
            return {"message": f"{robot_side.capitalize()} robot moving to target tcp {request.target_tcp}"}

        @self.app.post('/execute_primitive/{robot_side}')
        async def execute_primitive(robot_side: str, request: ActionPrimitiveRequest) -> Dict[str, str]:
            if robot_side not in ['left', 'right']:
                raise HTTPException(status_code=400, detail="Invalid robot side. Use 'left' or 'right'.")

            # Check if the requested robot is enabled
            if robot_side == 'left' and not self.enable_left_arm:
                raise HTTPException(status_code=400, detail="Left robot is not enabled.")
                
            if robot_side == 'right' and not self.enable_right_arm:
                raise HTTPException(status_code=400, detail="Right robot is not enabled.")

            robot = self.left_robot if robot_side == 'left' else self.right_robot
            robot.execute_primitive(request.primitive_name, request.input_params)
            return {"message": f"{robot_side.capitalize()} robot executing primitive {request}"}

        @self.app.get('/get_nominal_stiffness/{robot_side}')
        async def get_nominal_stiffness(robot_side: str) -> NominalStiffness:
            if robot_side not in ['left', 'right']:
                raise HTTPException(status_code=400, detail="Invalid robot side. Use 'left' or 'right'.")
            
            robot = self.left_robot if robot_side == 'left' else self.right_robot
            return NominalStiffness(stiffness=robot.get_nominal_stiffness())

        @self.app.post('/set_cartesian_impedance/{robot_side}')
        async def set_cartesian_impedance(robot_side: str, request: CartesianImpedanceRequest) -> Dict[str, str]:
            if robot_side not in ['left', 'right']:
                raise HTTPException(status_code=400, detail="Invalid robot side. Use 'left' or 'right'.")
            
            if robot_side == 'left':
                if request.source == "policy" and self.left_robot_impedance_control_status == "user":
                    raise HTTPException(status_code=400, detail="Left robot's impedance is under user's control")
                self.left_robot_impedance_control_status = request.source
            elif robot_side == 'right':
                if request.source == "policy" and self.right_robot_impedance_control_status == "user":
                    raise HTTPException(status_code=400, detail="Right robot's impedance is under user's control")
                self.right_robot_impedance_control_status = request.source
            
            robot = self.left_robot if robot_side == 'left' else self.right_robot
            robot.set_cartesian_impedance(request.stiffness, request.damping)
            return {"message": f"{robot_side.capitalize()} robot cartesian impedance set to "
                    f"stiffness: {request.stiffness}, damping: {request.damping}"}

        @self.app.post('/reset_cartesian_impedance/{robot_side}')
        async def reset_cartesian_impedance(robot_side: str) -> Dict[str, str]:
            if robot_side not in ['left', 'right']:
                raise HTTPException(status_code=400, detail="Invalid robot side. Use 'left' or 'right'.")
            
            robot = self.left_robot if robot_side == 'left' else self.right_robot
            robot.reset_cartesian_impedance()

            if robot_side == 'left':
                self.left_robot_impedance_control_status = None
            elif robot_side == 'right':
                self.right_robot_impedance_control_status = None

            return {"message": f"{robot_side.capitalize()} robot cartesian impedance reset to default"}

        @self.app.get('/get_current_tcp/{robot_side}')
        async def get_current_tcp(robot_side: str) -> List[float]:
            if robot_side not in ['left', 'right']:
                raise HTTPException(status_code=400, detail="Invalid robot side. Use 'left' or 'right'.")

            # Check if the requested robot is enabled
            if robot_side == 'left' and not self.enable_left_arm:
                raise HTTPException(status_code=400, detail="Left robot is not enabled.")
                
            if robot_side == 'right' and not self.enable_right_arm:
                raise HTTPException(status_code=400, detail="Right robot is not enabled.")

            robot = self.left_robot if robot_side == 'left' else self.right_robot
            return robot.get_current_tcp()

        @self.app.post('/birobot_go_home')
        async def birobot_go_home() -> Dict[str, str]:
            # Require both robots to be enabled for bimanual home operation
            if not (self.enable_left_arm and self.enable_right_arm):
                raise HTTPException(status_code=400, 
                                  detail="Both robots must be enabled to use bimanual go home function")
                
            if self.planner is None:
                return {"message": "Planner is not available"}
                
            self.left_robot.robot.SwitchMode(self.left_robot.mode.NRT_JOINT_POSITION)
            self.right_robot.robot.SwitchMode(self.right_robot.mode.NRT_JOINT_POSITION)

            current_q = self.left_robot.get_current_q() + self.right_robot.get_current_q()
            waypoints = self.planner.getGoHomeTraj(current_q)

            for js in waypoints:
                print(js)
                self.left_robot.move(js[:7])
                self.right_robot.move(js[7:])
                time.sleep(0.01)

            self.left_robot.robot.SwitchMode(self.left_robot.mode.NRT_CARTESIAN_MOTION_FORCE)
            self.right_robot.robot.SwitchMode(self.right_robot.mode.NRT_CARTESIAN_MOTION_FORCE)
            return {"message": "Bimanual robots have gone home"}

    def run(self):
        enabled_arms = []
        if self.enable_left_arm:
            enabled_arms.append("left")
        if self.enable_right_arm:
            enabled_arms.append("right")
        
        logger.info(f"Start Robot Fast-API Server at {self.host_ip}:{self.port}, enabled arms: {', '.join(enabled_arms)}")
        uvicorn.run(self.app, host=self.host_ip, port=self.port, log_level="critical")

def main():
    from hydra import initialize, compose
    from hydra.utils import instantiate

    with initialize(config_path='../../../config', version_base="1.3"):
        # config is relative to a module
        cfg = compose(config_name="bimanual_two_realsense_one_gelslim")

    robot_server = instantiate(cfg.robot_server)
    robot_server.run()


if __name__ == "__main__":
    main()