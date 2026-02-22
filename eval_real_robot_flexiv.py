# %%
import pathlib
import torch
import dill
import hydra
from omegaconf import OmegaConf
from ImplicitRDP.workspace.base_workspace import BaseWorkspace
from ImplicitRDP.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from ImplicitRDP.policy.diffusion_transformer_image_policy import DiffusionTransformerImagePolicy

import os
import psutil

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

OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'ImplicitRDP', 'config')),
    config_name="train_diffusion_unet_real_image_workspace"
)
def main(cfg):
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    
    # load checkpoint
    ckpt_path = cfg.ckpt_path
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    workspace.load_payload(payload, exclude_keys=["optimizer", "scaler"], include_keys=None, strict=False)

    # hacks for method-specific setup.
    if 'unet' in cfg.name:
        # diffusion unet model
        policy: DiffusionUnetImagePolicy
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        if 'latent' in cfg.name:
            policy.at.set_normalizer(policy.normalizer)

        device = torch.device('cuda')
        policy.eval().to(device)

        # set inference params
        policy.num_inference_steps = 8  # DDIM inference iterations
    elif 'transformer' in cfg.name:
        # diffusion transformer model
        policy: DiffusionTransformerImagePolicy
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        device = torch.device('cuda')
        policy.eval().to(device)

        # set inference params
        policy.num_inference_steps = 5  # DDIM inference iterations
    else:
        raise NotImplementedError

    # run eval
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner)
    env_runner.run(policy)


# %%
if __name__ == '__main__':
    main()
