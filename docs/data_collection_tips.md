## Data Collection Tips
### 1. Pay Attention to Force Feedback during Kinematic Teaching
Since ImplicitRDP relies on force information for real-time responses, the collected data have to include actions which apply appropriate force to the object.
We provide a web server for monitoring the force in [ImplicitRDP/real_world/publisher/bimanual_robot_publisher.py](../ImplicitRDP/real_world/publisher/bimanual_robot_publisher.py).
You can access it at `http://0.0.0.0:8000` to monitor the force during kinematic teaching.
