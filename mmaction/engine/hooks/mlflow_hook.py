from mmengine.hooks import Hook
from mmengine.runner import  Runner

from mmaction.registry import HOOKS

from pathlib import Path
import mlflow
from typing import Any
import os
import glob
import torch
from collections import defaultdict

import importlib.util as imu

NOT_LOG = [
    'default_hooks','custom_hooks','default_scope','env_cfg','file_client_args',
    'launcher','log_level','train_cfg','train_dataset_cfg','train_dataloader',
    'val_cfg','val_dataset_cfg','val_dataloader','val_evaluator',
    'vis_backends','visualizer','log_processor','tags'
    ]

def flatten_dict(d:dict, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list) and all(isinstance(i, dict) for i in v):
            for i, item in enumerate(v):
                list_key = f"{new_key}_{i}"
                items.extend(flatten_dict(item, list_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


@HOOKS.register_module()
class MLflowHook(Hook):

    def __init__(self,
                 log_interval: int = 1,
                 ) -> None:
        self.log_interval = log_interval
        self.val_metric_sums: dict[str,list] = defaultdict(list)
        self.best_val_acc: dict[str,float]= defaultdict(float)

    def before_run(self, runner:Runner) -> None:
        
        cfg = Path(runner.log_dir + '/vis_data' + '/config.py')
        mlflow.log_artifact(str(cfg))

        spec = imu.spec_from_file_location("config_module", cfg)
        config = imu.module_from_spec(spec)
        spec.loader.exec_module(config)

        params:dict[str,Any] = {}
        for key, value in config.__dict__.items():
            if (key.startswith("__") and not callable(value) ) or (key in NOT_LOG):
                continue

            if isinstance(value, dict):
                params.update(flatten_dict(value,key))
            elif isinstance(value, list) and all(isinstance(i, dict) for i in value):
                for i, item in enumerate(value):
                    params.update(flatten_dict(item, f"{key}_{i}") )
            else:
                params[key] = value

        mlflow.log_params(params)

    def before_train(self, runner:Runner) -> None:
        params:dict[str,Any] = {}
        params.update(dict(train_batch_size=runner.train_dataloader.batch_size))
        params.update(dict(val_batch_size=runner.val_dataloader.batch_size))

        if runner.train_dataloader.batch_sampler:
            num_samples_per_epoch = runner.train_dataloader.batch_sampler.sampler.num_samples
            params.update(dict(samples_per_epoch=num_samples_per_epoch))

        num_samples_train = len(runner.train_dataloader.dataset)
        params.update(dict(num_samples_train=num_samples_train))

        num_samples_val = len(runner.val_dataloader.dataset)
        params.update(dict(num_samples_val=num_samples_val))

        mlflow.log_params(params)

        dataset_path = Path(runner.train_dataloader.dataset.ann_file).parent
        mlflow.log_artifacts(str(dataset_path),artifact_path='dataset')
        
    def before_train_epoch(self, runner:Runner) -> None:
        self.metric_sum: dict[str, float] = defaultdict(float)
        self.num_batches = 0

    def after_train_iter(self, runner:Runner, batch_idx: int, data_batch: dict, outputs: dict) -> None:
        self.num_batches += 1

        for key,value in outputs.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            if not isinstance(value, (int, float)):
                continue
            
            self.metric_sum[key] += value

        
    def after_train_epoch(self, runner:Runner) -> None:
        if not runner.epoch % self.log_interval == 0:
            return
        
        for key, total_value in self.metric_sum.items():
            avg_value = total_value / self.num_batches
            self.metric_sum[key] = avg_value

        mlflow.log_metrics(self.metric_sum,step=runner.epoch)

        lrs = runner.optim_wrapper.get_lr()['lr']
        for i,lr in enumerate(lrs):
            mlflow.log_metric(f'lr_{i}',lr,step=runner.epoch)

    def after_val_epoch(self, runner:Runner, metrics: dict) -> None:
        for key, value in metrics.items():
            self.val_metric_sums[key].append(value)

            if value >= self.best_val_acc[key]:
                self.best_val_acc[key] = value
                mlflow.log_metric(f'best_{key}',value,step=runner.epoch)
         
        mlflow.log_metrics(metrics=metrics,step=runner.epoch)



    
    def after_test_epoch(self, runner:Runner, metrics: dict) -> None:
        mlflow.log_metrics(metrics=metrics,step=runner.epoch)


    def after_run(self, runner:Runner) -> None:
        if not runner.train_loop:
            mlflow.end_run()
            return
        
        work_dir = runner.work_dir
        patt = os.path.join(work_dir, 'best_*.pth')
        best_weights = glob.glob(patt)[0]
        
        mlflow.log_artifact(best_weights)
        last_epoch = os.path.join(work_dir,f'epoch_{runner.max_epochs}.pth')
        mlflow.log_artifact(last_epoch)

        val_avg_metrics = {f'average_{key}': sum(value) / len(value) for key, value in self.val_metric_sums.items()}
        mlflow.log_metrics(val_avg_metrics)