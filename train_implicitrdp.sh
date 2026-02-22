#!/bin/bash

#--------------------------------
# Config
GPU_ID=0

TASK="flip"
DATASET_VERSION="v2"
# ACTION_TYPE="right_arm_6DOF" # No Auxiliary Task
# ACTION_TYPE="right_arm_6DOF_wrench" # FP
ACTION_TYPE="right_arm_6DOF_virtual_target_stiffness" # VRR
DATASET_PATH="/home/wendi/Desktop/ImplicitRDP/data/${TASK}_${DATASET_VERSION}_zarr"
TASK_CONFIG="real_${TASK}_image_wrench_implicitrdp_10fps"

USE_AMP_FOR_POLICY=False
AMP_DTYPE_FOR_POLICY="bf16"
TRAINING_DEBUG=False
POLICY_LR=1.0e-4
LOGGING_MODE="online"
#--------------------------------

# Stage 0: Post-process data
echo "Stage 0: post-processing data..."
python post_process_data.py --tag ${TASK}_${DATASET_VERSION} --action_type ${ACTION_TYPE} --skip_static_frames

# Stage 1: Train Policy
echo "Stage 1: training Policy..."
POLICY_TASK_NAME=${TASK}_${DATASET_VERSION}_policy_bigger_acp
CUDA_VISIBLE_DEVICES=${GPU_ID} accelerate launch train.py \
    --config-name=train_reactive_diffusion_transformer_real_image_workspace \
    task=${TASK_CONFIG} \
    task.dataset_path=${DATASET_PATH} \
    task.name=${POLICY_TASK_NAME} \
    policy.noise_scheduler.prediction_type=v_prediction \
    policy.use_amp_for_inference=${USE_AMP_FOR_POLICY} \
    policy.amp_dtype=${AMP_DTYPE_FOR_POLICY} \
    training.use_amp=${USE_AMP_FOR_POLICY} \
    training.amp_dtype=${AMP_DTYPE_FOR_POLICY} \
    training.debug=${TRAINING_DEBUG} \
    optimizer.lr=${POLICY_LR} \
    logging.mode=${LOGGING_MODE}

exit 0