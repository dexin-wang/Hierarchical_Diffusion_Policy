"""
Usage:
Training:
python train_hdp.py --config-name=train_diffusion_lowdim_workspace
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import wandb
import hydra
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig
import pathlib
from hiera_diffusion_policy.workspace.base_workspace import BaseWorkspace

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'hiera_diffusion_policy','config'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)
    
    train_models = []
    if cfg.use_subgoal:
        train_models.append('guider')
    if cfg.eta > 0:
        train_models.append('critic')
    train_models.append('actor')

    for model in train_models:
        cfg.train_model = model
        if model == 'actor':
            if cfg.use_subgoal:
                cfg.guider_path = HydraConfig.get().runtime.output_dir
            if cfg.eta > 0:
                cfg.critic_path = HydraConfig.get().runtime.output_dir

        cls = hydra.utils.get_class(cfg._target_)
        workspace: BaseWorkspace = cls(cfg)
        workspace.run()

if __name__ == "__main__":
    main()
