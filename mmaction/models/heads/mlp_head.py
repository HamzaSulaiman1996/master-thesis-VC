from torch import Tensor, nn
from mmengine.model.weight_init import normal_init

from mmaction.registry import MODELS
from .base import BaseHead


@MODELS.register_module()
class MLP(BaseHead):
    def __init__(
            self,
            in_channels: int,
            dropout_rate: float,
            num_classes:int,
            use_cls_token:bool=False,
            **kwargs,
    ):  
        self.use_cls_token = use_cls_token
        self.in_channels = in_channels
        self.dropout_rate = dropout_rate
        super().__init__(num_classes=num_classes,in_channels=in_channels,**kwargs)
        
        self.features = nn.Sequential(
            nn.Linear(self.in_channels,self.in_channels),
            nn.BatchNorm1d(self.in_channels),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.in_channels,self.num_classes),
        )

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.norm_layer = nn.LayerNorm(self.in_channels)

    def init_weights(self) -> None:
        """Initiate the parameters from scratch."""
        normal_init(self.features, std=0.02)


    def forward(self,x:Tensor,**kwargs):
        if self.use_cls_token:
            x = self.norm_layer(x[:,0])
            x = self.features(x)
            return x
        
        x = x.permute(0,2,1)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.features(x)
        return x