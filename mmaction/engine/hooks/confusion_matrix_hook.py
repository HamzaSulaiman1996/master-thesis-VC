# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence

from mmengine.hooks import Hook
from mmengine.runner import  Runner

from mmaction.registry import HOOKS
from mmaction.structures import ActionDataSample
from mmaction.evaluation import ConfusionMatrix

import torch
from pathlib import Path
import json

import mlflow


@HOOKS.register_module()
class ConfusionMatrixHook(Hook):
    priority='LOW'
    def __init__(self,
                 class_map:str,
                 out_dir: Optional[str] = None,
                 ):
        
        assert class_map.endswith('.json')
        self.class_map = class_map
        with open(self.class_map,'r') as f:
            class_idx = json.load(f)

        self.classes = [value for _,value in class_idx.items()]
        self.num_classes = len(self.classes)
        self.out_dir = out_dir
        
    def before_run(self, runner:Runner) -> None:
        mlflow.log_artifact(self.class_map)

    def before_val_epoch(self, runner:Runner) -> None:
        self.cfmatrix = []
    
    def before_test_epoch(self, runner:Runner) -> None:
        self.cfmatrix = []

    def after_val_iter(self, runner:Runner, batch_idx: int, data_batch: dict , outputs: Sequence[ActionDataSample]) -> None:
        for data_sample in outputs:
            pred_label = data_sample.get('pred_label').cpu()
            gt_label = data_sample.get('gt_label').cpu()

            self.cfmatrix.append({
                'pred_label': pred_label,
                'gt_label': gt_label
            })
    def after_test_iter(self, runner:Runner, batch_idx: int, data_batch: dict , outputs: Sequence[ActionDataSample]) -> None:
        for data_sample in outputs:
            pred_label = data_sample.get('pred_label').cpu()
            gt_label = data_sample.get('gt_label').cpu()

            self.cfmatrix.append({
                'pred_label': pred_label,
                'gt_label': gt_label
            })

    def after_val_epoch(self, runner:Runner, metrics) -> None:
        out_path = self._compute_matrix(runner)
        mlflow.log_artifact(str(out_path),artifact_path='confusion matrix')

    def after_test_epoch(self, runner:Runner, metrics) -> None:
        out_path = self._compute_matrix(runner)
        mlflow.log_artifact(str(out_path),artifact_path='confusion matrix')


    def _compute_matrix(self,runner:Runner):
        pred_label = []
        gt_labels = []
        for result in self.cfmatrix:
            pred_label.append(result['pred_label'])
            gt_labels.append(result['gt_label'])
        confusion_matrix = ConfusionMatrix().calculate(
            torch.cat(pred_label),
            torch.cat(gt_labels),
            num_classes=self.num_classes
        )

        fig = ConfusionMatrix().plot(confusion_matrix,include_values=True,classes=self.classes)

        self.out_dir = Path(self.out_dir) if self.out_dir else Path(runner.log_dir) / f'confusion_matrix'

        self.out_dir.mkdir(parents=True,exist_ok=True)
        fname = f"epoch_{runner.epoch}.png"
        image_path = self.out_dir / fname
        fig.savefig(image_path)

        return image_path

