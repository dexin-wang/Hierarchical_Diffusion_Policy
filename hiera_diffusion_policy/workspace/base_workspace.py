from typing import Optional
import os
import pathlib
import hydra
import copy
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import dill
import torch
import threading


class BaseWorkspace:
    include_keys = tuple()
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir: Optional[str]=None):
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None

    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir
    
    def run(self):
        """
        Create any resource shouldn't be serialized as local variables
        """
        pass

    def save_checkpoint(self, path=None, tag='latest', 
            exclude_keys=None,
            include_keys=None,
            use_thread=True):
        """
        保存model, ema_model, optimizer的参数，以及global_step和epoch
        """
        if path is None:    # <--
            path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        else:
            path = pathlib.Path(path)
        '''
        self.exclude_keys = ()
        self.include_keys = ['global_step', 'epoch']
        '''
        if exclude_keys is None:    # <--
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ('_output_dir',)

        path.parent.mkdir(parents=False, exist_ok=True)
        payload = {
            'cfg': self.cfg,
            'state_dicts': dict(),
            'pickles': dict()
        }
        '''
        实例化对象的__dict__中存储了一些类中__init__的一些属性。
        如果子类的__init__中执行里父类的__init__，子类的__dict__会同时包含父类和子类__init__中的属性。
        '''
        for key, value in self.__dict__.items():
            if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'):
                # modules, optimizers and samplers etc
                # model, ema_model, optimizer
                if key not in exclude_keys:
                    if use_thread:  # <--
                        payload['state_dicts'][key] = _copy_to_cpu(value.state_dict())
                    else:
                        payload['state_dicts'][key] = value.state_dict()
            elif key in include_keys:
                payload['pickles'][key] = dill.dumps(value)
        if use_thread:  # <--
            self._saving_thread = threading.Thread(
                target=lambda : torch.save(payload, path.open('wb'), pickle_module=dill))
            self._saving_thread.start()
        else:
            torch.save(payload, path.open('wb'), pickle_module=dill)
        return str(path.absolute())
    
    def get_checkpoint_path(self, tag='latest'):
        return pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload['pickles'].keys()

        for key, value in payload['state_dicts'].items():
            if key not in exclude_keys:
                try:
                    self.__dict__[key].load_state_dict(value, **kwargs)
                except:
                    print(key, 'load failed!')
        for key in include_keys:
            if key in payload['pickles']:
                self.__dict__[key] = dill.loads(payload['pickles'][key])
    
    def load_checkpoint(self, path=None, tag='latest',
            exclude_keys=None, 
            include_keys=None, 
            **kwargs):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        payload = torch.load(path.open('rb'), pickle_module=dill, **kwargs)
        self.load_payload(payload, 
            exclude_keys=exclude_keys, 
            include_keys=include_keys)
        return payload
    
    def load_checkpoint_guider(self, guider_path):
        print('loading Guider:', guider_path)
        payload_guider = torch.load(guider_path.open('rb'), pickle_module=dill)
            
        for key, value in payload_guider['state_dicts'].items():
            if key == 'model' or key == 'ema_model':
                model_dict =  self.__dict__[key].state_dict()
                guider_params = {k:v for k, v in value.items() if k.startswith('guider')}
                model_dict.update(guider_params)
                self.__dict__[key].load_state_dict(model_dict)
    
    def load_checkpoint_AC(self, AC_path):
        print('loading AC:', AC_path)
        payload_AC = torch.load(AC_path.open('rb'), pickle_module=dill)
        
        for key, value in payload_AC['state_dicts'].items():
            if key == 'model' or key == 'ema_model':
                model_dict =  self.__dict__[key].state_dict()
                ac_params = {k:v for k, v in value.items() if k.startswith('actor') or k.startswith('critic')}
                model_dict.update(ac_params)
                self.__dict__[key].load_state_dict(model_dict)
    
    def load_checkpoint_actor(self, actor_path):
        print('loading actor:', actor_path)
        payload_actor = torch.load(actor_path.open('rb'), pickle_module=dill)
        
        for key, value in payload_actor['state_dicts'].items():
            if key == 'model' or key == 'ema_model':
                model_dict =  self.__dict__[key].state_dict()
                actor_params = {k:v for k, v in value.items() if k.startswith('actor')}
                model_dict.update(actor_params)
                self.__dict__[key].load_state_dict(model_dict)

    def load_checkpoint_critic(self, critic_path):
        print('loading critic:', critic_path)
        payload_critic = torch.load(critic_path.open('rb'), pickle_module=dill)
        
        for key, value in payload_critic['state_dicts'].items():
            if key == 'model' or key == 'ema_model':
                model_dict =  self.__dict__[key].state_dict()
                critic_params = {k:v for k, v in value.items() if k.startswith('critic')}
                model_dict.update(critic_params)
                self.__dict__[key].load_state_dict(model_dict)

    
    @classmethod
    def create_from_checkpoint(cls, path, 
            exclude_keys=None, 
            include_keys=None,
            **kwargs):
        payload = torch.load(open(path, 'rb'), pickle_module=dill)
        instance = cls(payload['cfg'])
        instance.load_payload(
            payload=payload, 
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs)
        return instance

    def save_snapshot(self, tag='latest'):
        """
        Quick loading and saving for reserach, saves full state of the workspace.

        However, loading a snapshot assumes the code stays exactly the same.
        Use save_checkpoint for long-term storage.
        """
        path = pathlib.Path(self.output_dir).joinpath('snapshots', f'{tag}.pkl')
        path.parent.mkdir(parents=False, exist_ok=True)
        torch.save(self, path.open('wb'), pickle_module=dill)
        return str(path.absolute())
    
    @classmethod
    def create_from_snapshot(cls, path):
        return torch.load(open(path, 'rb'), pickle_module=dill)


def _copy_to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to('cpu')
    elif isinstance(x, dict):
        result = dict()
        for k, v in x.items():
            result[k] = _copy_to_cpu(v)
        return result
    elif isinstance(x, list):
        return [_copy_to_cpu(k) for k in x]
    else:
        return copy.deepcopy(x)
