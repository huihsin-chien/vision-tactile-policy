## Customized Deployment Guide

### Add Customized Tasks
There are two kinds of task configs: **shared configs** and **policy-related configs**.
#### Shared Configs
Shared configs define the robot's degrees of freedom, as well as the sensors used and their parameters.
You can take [ImplicitRDP/config/task/real_flip_one_usb_camera_kineteach_10fps.yaml](../ImplicitRDP/config/task/real_flip_one_usb_camera_kineteach_10fps.yaml)
as an example.

Note that the `robot_control_mode` represents the degrees of freedom of the robot action during the kinematic teaching.
You can add new mode in [ImplicitRDP/common/data_models.py](../ImplicitRDP/common/data_models.py) and
modify [ImplicitRDP/real_world/kinematic_teaching/kineteach_controller.py](../ImplicitRDP/real_world/kinematic_teaching/kineteach_controller.py) accordingly.

#### Policy-related Configs
Policy-related configs define the obs and action shape, env runner parameters, and dataset parameters.

### Add Customized Force Sensors
1. **Implement the Sensor Publisher.**
   Because Flexiv Rizon 4s is equipped with a built-in 6-axis force/torque sensor at the end effector, we have implemented the publisher in [ImplicitRDP/real_world/publisher/bimanual_robot_publisher.py](../ImplicitRDP/real_world/publisher/bimanual_robot_publisher.py).
   If you want to use a separate force sensor, you can add an individual force sensor publisher to publish the wrench to the ROS topic.
2. **Modify the Device Mapping Server.**
   Our experiments use many different settings of sensor hardwares, so we use `Device Mapping Server` as an online database for other processes to query the current hardware settings.
   It provides the mapping from the sensor to the ROS2 topic name,
   which is requested by the `Data Recorder` and `Runner`. Please add the new sensor mapping in [ImplicitRDP/real_world/device_mapping/device_mapping_server.py](../ImplicitRDP/real_world/device_mapping/device_mapping_server.py).
    
3. **Modify Services.**
   Modify the following services to be compatible with the new sensors:
   - Data Recorder: [ImplicitRDP/real_world/teleoperation/data_recorder.py](../ImplicitRDP/real_world/teleoperation/data_recorder.py)
   - Runner:
      - Real Runner: [ImplicitRDP/env_runner/real_runner.py](../ImplicitRDP/env_runner/real_runner.py)
      - Real Stable Reactive Runner: [ImplicitRDP/env_runner/real_stable_reactive_runner.py](../ImplicitRDP/env_runner/real_stable_reactive_runner.py)

### Add Customized Robots
1. **Implement the Robot Server.**
   Note that we change the stiffness of the robot for kinematic teaching, so you need to adopt a robot which supports stiffness adjustment.
   Refer to [ImplicitRDP/real_world/robot/bimanual_flexiv_server.py](../ImplicitRDP/real_world/robot/bimanual_flexiv_server.py)
   and implement a robot server for your robot with the same API.
   This server is requested by the `Kineteach Controller` and `Real Env`.
2. **Implement the Robot Publisher.**
   Refer to [ImplicitRDP/real_world/publisher/bimanual_robot_publisher.py](../ImplicitRDP/real_world/publisher/bimanual_robot_publisher.py)
   and implement a similar publisher for your robot.
3. **(Optional) Modify Services.**
   The following services may need to be modified to be compatible with the new robot:
   - Kineteach Controller: [ImplicitRDP/real_world/kinematic_teaching/kineteach_controller.py](../ImplicitRDP/real_world/kinematic_teaching/kineteach_controller.py)
   - Real Env: [ImplicitRDP/env/real_bimanual/real_env.py](../ImplicitRDP/env/real_bimanual/real_env.py)
