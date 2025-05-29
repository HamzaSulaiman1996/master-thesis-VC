from mmaction.registry import MODELS
from mmengine.model import BaseModule
from typing import Optional,Union,List,Dict
from timm.models.resnet import resnet50
import torch



@MODELS.register_module()
class Resnet50(BaseModule):
    def __init__(
            self,
            pretrained:bool = True, # pretrained weights from vit_base_patch16_224
            load_weights: Optional[str] = None, # custom pretrained weights
            freeze: bool = False,
            init_cfg: Optional[Union[Dict, List[Dict]]] = [
                dict(type='Kaiming', layer='Conv2d'),
                dict(type='Constant', layer='BatchNorm2d', val=1.)
                ]         
    ):
        
        self.freeze = freeze
        self.pretrained = pretrained
        self.load_weights = load_weights
        if self.load_weights and self.pretrained:
            raise ValueError('load_weights and pretrained cannot be true at the same time')
        
        super().__init__(init_cfg=init_cfg)

        self.backbone = resnet50(
            pretrained=pretrained,
            num_classes=0,
            features_only=True,
            out_indices=1,
            )


    def init_weights(self):
        if self.load_weights:
            self.init_cfg = dict(type='Pretrained', checkpoint=self.load_weights)
            super().init_weights()

        elif self.pretrained:
            return None
        

    def _freeze_stages(self):
        if self.freeze:
            for _,weights in self.named_parameters():
                    weights.requires_grad = False
    

    def forward(self, x: torch.Tensor):
        B, C, F, H, W = x.shape
        x = x.reshape(B * F, C, H, W)
        x = self.backbone(x)[0]
        x = x.reshape(B,F,*x.shape[1:])
        return x
    

    def train(self, mode: bool = True) -> None:
        """Convert the model into training mode while keep layers frozen."""
        super(Resnet50, self).train(mode)
        self._freeze_stages()