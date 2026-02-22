#!/bin/bash

# Box Flipping v2

# DP (Box Flipping) v2
# python eval_real_robot_flexiv.py \
#      --config-name train_diffusion_unet_real_image_workspace \
#      task=real_flip_image_dp_10fps \
#      +task.env_runner.output_dir=/home/wendi/Desktop/ImplicitRDP/data/checkpoints/real_flip_image_dp_10fps \
#      ckpt_path=/home/wendi/Desktop/ImplicitRDP/data/checkpoints/real_flip_image_dp_10fps/checkpoints/latest.ckpt

# RDP w. Force (Box Flipping) v2
# python eval_real_robot_flexiv.py \
#      --config-name train_latent_diffusion_unet_real_image_workspace \
#      task=real_flip_image_wrench_ldp_10fps \
#      at=at_flip \
#      at_load_dir=/home/wendi/Desktop/ImplicitRDP/data/checkpoints/real_flip_image_wrench_at_10fps/checkpoints/latest.ckpt \
#      +task.env_runner.output_dir=/home/wendi/Desktop/ImplicitRDP/data/checkpoints/real_flip_image_wrench_ldp_10fps \
#      ckpt_path=/home/wendi/Desktop/ImplicitRDP/data/checkpoints/real_flip_image_wrench_ldp_10fps/checkpoints/latest.ckpt

# DPT w. Force (Box Flipping) v2
# python eval_real_robot_flexiv.py \
#    --config-name=train_diffusion_transformer_real_image_workspace \
#    policy.noise_scheduler.prediction_type=v_prediction \
#    task=real_flip_image_wrench_dpt_10fps \
#    +task.env_runner.output_dir=/home/wendi/Desktop/ImplicitRDP/data/checkpoints/real_flip_image_wrench_dpt_10fps \
#    ckpt_path=/home/wendi/Desktop/ImplicitRDP/data/checkpoints/real_flip_image_wrench_dpt_10fps/checkpoints/latest.ckpt

# DPT w. Force (Box Flipping) v2, VRR
# python eval_real_robot_flexiv.py \
#    --config-name=train_diffusion_transformer_real_image_workspace \
#    policy.noise_scheduler.prediction_type=v_prediction \
#    task=real_flip_image_wrench_dpt_vrr_10fps \
#    task.env_runner.action_type=right_arm_6DOF \
#    +task.env_runner.output_dir=/home/wendi/Desktop/ImplicitRDP/data/checkpoints/real_flip_image_wrench_dpt_vrr_10fps \
#    ckpt_path=/home/wendi/Desktop/ImplicitRDP/data/checkpoints/real_flip_image_wrench_dpt_vrr_10fps/checkpoints/latest.ckpt

# ImplicitRDP w. Force (Box Flipping) v2
python eval_real_robot_flexiv.py \
   --config-name train_reactive_diffusion_transformer_real_image_workspace \
   policy.noise_scheduler.prediction_type=v_prediction \
   task=real_flip_image_wrench_implicitrdp_10fps \
   task.env_runner.action_type=right_arm_6DOF \
   +task.env_runner.output_dir=/home/wendi/Desktop/ImplicitRDP/data/checkpoints/real_flip_image_wrench_implicitrdp_10fps \
   ckpt_path=/home/wendi/Desktop/ImplicitRDP/data/checkpoints/real_flip_image_wrench_implicitrdp_10fps/checkpoints/latest.ckpt

# ImplicitRDP w. Force (Box Flipping) v2, no auxiliary task
# python eval_real_robot_flexiv.py \
#    --config-name train_reactive_diffusion_transformer_real_image_workspace \
#    policy.noise_scheduler.prediction_type=v_prediction \
#    task=real_flip_image_wrench_implicitrdp_noaux_10fps \
#    +task.env_runner.output_dir=/home/wendi/Desktop/ImplicitRDP/data/checkpoints/real_flip_image_wrench_implicitrdp_noaux_10fps \
#    ckpt_path=/home/wendi/Desktop/ImplicitRDP/data/checkpoints/real_flip_image_wrench_implicitrdp_noaux_10fps/checkpoints/latest.ckpt

# ImplicitRDP w. Force (Box Flipping) v2, force prediction
# python eval_real_robot_flexiv.py \
#    --config-name train_reactive_diffusion_transformer_real_image_workspace \
#    policy.noise_scheduler.prediction_type=v_prediction \
#    task=real_flip_image_wrench_implicitrdp_fp_10fps \
#    +task.env_runner.output_dir=/home/wendi/Desktop/ImplicitRDP/data/checkpoints/real_flip_image_wrench_implicitrdp_fp_10fps \
#    ckpt_path=/home/wendi/Desktop/ImplicitRDP/data/checkpoints/real_flip_image_wrench_implicitrdp_fp_10fps/checkpoints/latest.ckpt

# ImplicitRDP w. Force (Box Flipping) v2, epsilon
# python eval_real_robot_flexiv.py \
#    --config-name train_reactive_diffusion_transformer_real_image_workspace \
#    policy.noise_scheduler.prediction_type=epsilon \
#    task=real_flip_image_wrench_implicitrdp_10fps \
#    task.env_runner.action_type=right_arm_6DOF \
#    +task.env_runner.output_dir=/home/wendi/Desktop/ImplicitRDP/data/checkpoints/real_flip_image_wrench_implicitrdp_epsilon_10fps \
#    ckpt_path=/home/wendi/Desktop/ImplicitRDP/data/checkpoints/real_flip_image_wrench_implicitrdp_epsilon_10fps/checkpoints/latest.ckpt

# ImplicitRDP w. Force (Box Flipping) v2, sample
# python eval_real_robot_flexiv.py \
#    --config-name train_reactive_diffusion_transformer_real_image_workspace \
#    policy.noise_scheduler.prediction_type=sample \
#    task=real_flip_image_wrench_implicitrdp_10fps \
#    task.env_runner.action_type=right_arm_6DOF \
#    +task.env_runner.output_dir=/home/wendi/Desktop/ImplicitRDP/data/checkpoints/real_flip_image_wrench_implicitrdp_sample_10fps \
#    ckpt_path=/home/wendi/Desktop/ImplicitRDP/data/checkpoints/real_flip_image_wrench_implicitrdp_sample_10fps/checkpoints/latest.ckpt

# ImplicitRDP w. Force (Box Flipping) v2, 6d rotation
# python eval_real_robot_flexiv.py \
#    --config-name train_reactive_diffusion_transformer_real_image_workspace \
#    policy.noise_scheduler.prediction_type=v_prediction \
#    task=real_flip_image_wrench_implicitrdp_6d_10fps \
#    task.env_runner.action_type=right_arm_6DOF \
#    +task.env_runner.output_dir=/home/wendi/Desktop/ImplicitRDP/data/checkpoints/real_flip_image_wrench_implicitrdp_6d_10fps \
#    ckpt_path=/home/wendi/Desktop/ImplicitRDP/data/checkpoints/real_flip_image_wrench_implicitrdp_6d_10fps/checkpoints/latest.ckpt

# ---------------------------------------------------

# Switch Toggling v2

# DP (Switch Toggling) v2
# python eval_real_robot_flexiv.py \
#      --config-name train_diffusion_unet_real_image_workspace \
#      task=real_toggle_image_dp_10fps \
#      +task.env_runner.output_dir=/home/wendi/Desktop/ImplicitRDP/data/checkpoints/real_toggle_image_dp_10fps \
#      ckpt_path=/home/wendi/Desktop/ImplicitRDP/data/checkpoints/real_toggle_image_dp_10fps/checkpoints/latest.ckpt

# RDP w. Force (Switch Toggling) v2
# python eval_real_robot_flexiv.py \
#      --config-name train_latent_diffusion_unet_real_image_workspace \
#      task=real_toggle_image_wrench_ldp_10fps \
#      at=at_toggle \
#      at_load_dir=/home/wendi/Desktop/ImplicitRDP/data/checkpoints/real_toggle_image_wrench_at_10fps/checkpoints/latest.ckpt \
#      +task.env_runner.output_dir=/home/wendi/Desktop/ImplicitRDP/data/checkpoints/real_toggle_image_wrench_ldp_10fps \
#      ckpt_path=/home/wendi/Desktop/ImplicitRDP/data/checkpoints/real_toggle_image_wrench_ldp_10fps/checkpoints/latest.ckpt

# DPT w. Force (Switch Toggling) v2
# python eval_real_robot_flexiv.py \
#    --config-name=train_diffusion_transformer_real_image_workspace \
#    policy.noise_scheduler.prediction_type=v_prediction \
#    task=real_toggle_image_wrench_dpt_10fps \
#    +task.env_runner.output_dir=/home/wendi/Desktop/ImplicitRDP/data/checkpoints/real_toggle_image_wrench_dpt_10fps \
#    ckpt_path=/home/wendi/Desktop/ImplicitRDP/data/checkpoints/real_toggle_image_wrench_dpt_10fps/checkpoints/latest.ckpt

# DPT w. Force (Switch Toggling) v2, VRR
# python eval_real_robot_flexiv.py \
#    --config-name=train_diffusion_transformer_real_image_workspace \
#    policy.noise_scheduler.prediction_type=v_prediction \
#    task=real_toggle_image_wrench_dpt_vrr_10fps \
#    task.env_runner.action_type=right_arm_6DOF \
#    +task.env_runner.output_dir=/home/wendi/Desktop/ImplicitRDP/data/checkpoints/real_toggle_image_wrench_dpt_vrr_10fps \
#    ckpt_path=/home/wendi/Desktop/ImplicitRDP/data/checkpoints/real_toggle_image_wrench_dpt_vrr_10fps/checkpoints/latest.ckpt

# ImplicitRDP w. Force (Switch Toggling) v2
# python eval_real_robot_flexiv.py \
#    --config-name train_reactive_diffusion_transformer_real_image_workspace \
#    policy.noise_scheduler.prediction_type=v_prediction \
#    task=real_toggle_image_wrench_implicitrdp_10fps \
#    task.env_runner.action_type=right_arm_6DOF \
#    +task.env_runner.output_dir=/home/wendi/Desktop/ImplicitRDP/data/checkpoints/real_toggle_image_wrench_implicitrdp_10fps \
#    ckpt_path=/home/wendi/Desktop/ImplicitRDP/data/checkpoints/real_toggle_image_wrench_implicitrdp_10fps/checkpoints/latest.ckpt

# ImplicitRDP w. Force (Switch Toggling) v2, no auxiliary task
# python eval_real_robot_flexiv.py \
#    --config-name train_reactive_diffusion_transformer_real_image_workspace \
#    policy.noise_scheduler.prediction_type=v_prediction \
#    task=real_toggle_image_wrench_implicitrdp_noaux_10fps \
#    +task.env_runner.output_dir=/home/wendi/Desktop/ImplicitRDP/data/checkpoints/real_toggle_image_wrench_implicitrdp_noaux_10fps \
#    ckpt_path=/home/wendi/Desktop/ImplicitRDP/data/checkpoints/real_toggle_image_wrench_implicitrdp_noaux_10fps/checkpoints/latest.ckpt

# ImplicitRDP w. Force (Switch Toggling) v2, force prediction
# python eval_real_robot_flexiv.py \
#    --config-name train_reactive_diffusion_transformer_real_image_workspace \
#    policy.noise_scheduler.prediction_type=v_prediction \
#    task=real_toggle_image_wrench_implicitrdp_fp_10fps \
#    +task.env_runner.output_dir=/home/wendi/Desktop/ImplicitRDP/data/checkpoints/real_toggle_image_wrench_implicitrdp_fp_10fps \
#    ckpt_path=/home/wendi/Desktop/ImplicitRDP/data/checkpoints/real_toggle_image_wrench_implicitrdp_fp_10fps/checkpoints/latest.ckpt

# ImplicitRDP w. Force (Switch Toggling) v2, epsilon
# python eval_real_robot_flexiv.py \
#    --config-name train_reactive_diffusion_transformer_real_image_workspace \
#    policy.noise_scheduler.prediction_type=epsilon \
#    task=real_toggle_image_wrench_implicitrdp_10fps \
#    task.env_runner.action_type=right_arm_6DOF \
#    +task.env_runner.output_dir=/home/wendi/Desktop/ImplicitRDP/data/checkpoints/real_toggle_image_wrench_implicitrdp_epsilon_10fps \
#    ckpt_path=/home/wendi/Desktop/ImplicitRDP/data/checkpoints/real_toggle_image_wrench_implicitrdp_epsilon_10fps/checkpoints/latest.ckpt

# ImplicitRDP w. Force (Switch Toggling) v2, sample
# python eval_real_robot_flexiv.py \
#    --config-name train_reactive_diffusion_transformer_real_image_workspace \
#    policy.noise_scheduler.prediction_type=sample \
#    task=real_toggle_image_wrench_implicitrdp_10fps \
#    task.env_runner.action_type=right_arm_6DOF \
#    +task.env_runner.output_dir=/home/wendi/Desktop/ImplicitRDP/data/checkpoints/real_toggle_image_wrench_implicitrdp_sample_10fps \
#    ckpt_path=/home/wendi/Desktop/ImplicitRDP/data/checkpoints/real_toggle_image_wrench_implicitrdp_sample_10fps/checkpoints/latest.ckpt

# ImplicitRDP w. Force (Switch Toggling) v2, 6d rotation
# python eval_real_robot_flexiv.py \
#    --config-name train_reactive_diffusion_transformer_real_image_workspace \
#    policy.noise_scheduler.prediction_type=v_prediction \
#    task=real_toggle_image_wrench_implicitrdp_6d_10fps \
#    task.env_runner.action_type=right_arm_6DOF \
#    +task.env_runner.output_dir=/home/wendi/Desktop/ImplicitRDP/data/checkpoints/real_toggle_image_wrench_implicitrdp_6d_10fps \
#    ckpt_path=/home/wendi/Desktop/ImplicitRDP/data/checkpoints/real_toggle_image_wrench_implicitrdp_6d_10fps/checkpoints/latest.ckpt