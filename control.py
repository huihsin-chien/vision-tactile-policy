import threading
import multiprocessing
import psutil
import time
import os

import rclpy
from ImplicitRDP.real_world.real_world_transforms import RealWorldTransforms
from ImplicitRDP.real_world.teleoperation.teleop_server import TeleopServer
from ImplicitRDP.real_world.kinematic_teaching.kineteach_controller import KineteachController
from ImplicitRDP.real_world.publisher.bimanual_robot_publisher import BimanualRobotPublisher
from ImplicitRDP.real_world.robot.bimanual_flexiv_server import BimanualFlexivServer
import hydra
from omegaconf import DictConfig
from loguru import logger

# add this to prevent assigning too many threads when using numpy
os.environ["OPENBLAS_NUM_THREADS"] = "12"
os.environ["MKL_NUM_THREADS"] = "12"
os.environ["NUMEXPR_NUM_THREADS"] = "12"
os.environ["OMP_NUM_THREADS"] = "12"

import cv2
# add this to prevent assigning too many threads when using open-cv
cv2.setNumThreads(12)

# Set the CPU affinity to the last 3 cores to avoid conflict with the main process
# get the total number of cores
total_cores = psutil.cpu_count()
# assign the last 4 cores to the server
num_cores_to_bind = 8
# calculate the start core
cores_to_bind = set(range(total_cores - num_cores_to_bind, total_cores))
# set the CPU affinity
os.sched_setaffinity(0, cores_to_bind)

def create_robot_publisher_node(cfg: DictConfig, transforms: RealWorldTransforms):
    rclpy.init(args=None)
    robot_publisher_node = BimanualRobotPublisher(transforms=transforms,
                                                  **cfg.task.publisher.robot_publisher)
    try:
        rclpy.spin(robot_publisher_node)
    except KeyboardInterrupt:
        robot_publisher_node.destroy_node()
        # rclpy.shutdown()

@hydra.main(
    config_path="ImplicitRDP/config", config_name="real_world_env", version_base="1.3"
)
def main(cfg: DictConfig):
    # create robot server
    if cfg.task.teleop_server is not None:
        control_mode = cfg.task.teleop_server.robot_control_mode
    elif cfg.task.kineteach_controller is not None:
        control_mode = cfg.task.kineteach_controller.robot_control_mode
    else:
        raise ValueError("Neither teleop_server nor kineteach_controller is configured. No control interface will be started.")
    
    robot_server = BimanualFlexivServer(robot_control_mode=control_mode,
                                        **cfg.task.robot_server)
    robot_server_thread = threading.Thread(target=robot_server.run, daemon=True)
    # start the robot server
    robot_server_thread.start()
    # wait for the robot server to start
    time.sleep(1)

    # create transforms with the configuration
    transforms = RealWorldTransforms(option=cfg.task.transforms)
    
    # Choose control interface based on configuration (teleop_server or kineteach_controller)
    control_process = None
    
    if cfg.task.teleop_server is not None:
        # Create and start teleop_server
        teleop_server = TeleopServer(robot_server_ip=cfg.task.robot_server.host_ip,
                                     robot_server_port=cfg.task.robot_server.port,
                                     transforms=transforms,
                                     **cfg.task.teleop_server)
        control_process = multiprocessing.Process(target=teleop_server.run)
        logger.info("Using TeleopServer for control")
    elif cfg.task.kineteach_controller is not None:
        # Create and start kineteach_controller
        kineteach_controller = KineteachController(robot_server_ip=cfg.task.robot_server.host_ip,
                                                   robot_server_port=cfg.task.robot_server.port,
                                                   **cfg.task.kineteach_controller)
        control_process = multiprocessing.Process(target=kineteach_controller.run)
        logger.info("Using KineteachController for control")
    else:
        raise ValueError("Neither teleop_server nor kineteach_controller is configured. No control interface will be started.")

    publisher_process = multiprocessing.Process(target=create_robot_publisher_node, args=(cfg, transforms))
    try:
        publisher_process.start()
        control_process.start()
        publisher_process.join()
        robot_server_thread.join()
    except KeyboardInterrupt:
        publisher_process.terminate()
        control_process.terminate()
    finally:
        # Wait for the process and thread to finish
        control_process.join()
        logger.info("Control process finished")
        publisher_process.join()
        logger.info("Publisher process finished")


if __name__ == "__main__":
    main()