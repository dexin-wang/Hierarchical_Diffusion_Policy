if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import numpy as np
import random
import wandb
import tqdm
import shutil

from hiera_diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from hiera_diffusion_policy.workspace.base_workspace import BaseWorkspace
from hiera_diffusion_policy.policy.dp3 import DP3
from hiera_diffusion_policy.model.diffusion.ema_model import EMAModel
from hiera_diffusion_policy.env_runner.base_pcd_runner import BasePcdRunner
from hiera_diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from hiera_diffusion_policy.common.json_logger import JsonLogger
from hiera_diffusion_policy.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainWorkspaceDP3(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # ************ configure model ************
        self.model:DP3 = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DP3 = None
        try:
            self.ema_model = copy.deepcopy(self.model)
        except: # minkowski engine could not be copied. recreate it
            self.ema_model = hydra.utils.instantiate(cfg.policy)

        # configure training state
        self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=self.model.parameters())

        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg) 

        # configure dataset
        # ************ 数据集 ************
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()   # 归一化
        self.model.set_normalizer(normalizer)
        self.ema_model.set_normalizer(normalizer)

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        # configure lr scheduler
        # ************ 设置 lr scheduler ************
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * cfg.training.num_epochs),
            last_epoch=self.global_step-1
        )

        ema: EMAModel = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        # configure env runner
        # ************ 设置任务的仿真环境 ************
        env_runner: BasePcdRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir,
            replay_buffer=dataset.replay_buffer)

        # configure logging
        # ************ 设置wandb ************
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update({"output_dir": self.output_dir,})

        # configure checkpoint
        # ************ 保存权重的路径生成器，动态保存topk个权重 ************
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        # ************ 迁移到指定设备 ************
        device = torch.device(cfg.training.device)
        self.model.to(device)
        self.ema_model.to(device)
        optimizer_to(self.optimizer, device)


        # save batch for sampling
        train_sampling_batch = None
        log_path = os.path.join(self.output_dir, 'logs.json.txt')

        # ********** 训练 **********
        # training loop
        with JsonLogger(log_path) as json_logger:
            # ************ 遍历epoch开始训练 ************
            for local_epoch_idx in range(cfg.training.num_epochs):
                if self.epoch >= cfg.training.num_epochs:
                    break
                step_log = dict()
                # ************ train for this epoch ************
                train_losses = list()

                with tqdm.tqdm(train_dataloader, desc=f"Training - epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # optimize
                        raw_loss = self.model.compute_loss(batch)
                        self.optimizer.zero_grad()
                        raw_loss.backward()
                        self.optimizer.step()
                        lr_scheduler.step()
                        loss = raw_loss.item()

                        ema.step(self.model)

                        # logging
                        tepoch.set_postfix(loss=loss, refresh=False)
                        train_losses.append(loss)

                        step_log = {
                            'train_loss': loss,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1
                
                # at the end of each epoch, replace train_loss with epoch average
                step_log['train_loss'] = np.mean(train_losses)
                

                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                # ************ 在仿真环境中测试 ************
                if (self.epoch % cfg.training.rollout_every) == 0:
                    runner_log = env_runner.run(self.model)
                    step_log.update(runner_log)

                # ************ 在测试集中测试loss ************
                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation - epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss = self.model.compute_loss(batch)
                                val_losses.append(loss)
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss

                # run diffusion sampling on a training batch
                # ************ 在一个训练batch上测试完整逆扩散过程的损失 ************
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        batch = train_sampling_batch    # Tensor, no norm

                        pred_action = self.model.predict_action(batch)['action_pred']
                        mse_action = torch.nn.functional.mse_loss(pred_action, batch['action'])
                        step_log['train_mse_error_action'] = mse_action.item()
                        
                        # release RAM
                        del batch
                        del pred_action, mse_action
                
                # ************ checkpoint ************
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    # checkpointing
                    # 保存最新的权重
                    if cfg.checkpoint.save_last_ckpt:   # <--
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()
                
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1