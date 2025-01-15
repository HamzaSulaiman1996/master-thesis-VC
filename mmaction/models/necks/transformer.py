import torch
import torch.nn as nn
from torch import Tensor
from mmengine.model import BaseModule
from mmaction.registry import MODELS
from mmengine.model.weight_init import normal_init,constant_init
import collections.abc
from typing import Tuple, Iterable,Optional,Union,Dict,List

class VideoMAEPatchEmbeddings(nn.Module):
    """
    Video to Patch Embedding. This module turns a batch of videos of shape (batch_size, num_frames, num_channels,
    height, width) into a tensor of shape (batch_size, seq_len, hidden_size) to be consumed by a Transformer encoder.

    The seq_len (the number of patches) equals (number of frames // tubelet_size) * (height // patch_size) * (width //
    patch_size).

    """

    def __init__(self,
                image_size:Tuple[int,int] | Iterable[int],
                patch_size:Tuple[int,int] | Iterable[int],
                num_channels:int,
                hidden_size:int,
                num_frames:int,
                tubelet_size:int,
                ):
        super().__init__()

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        self.image_size = image_size
        self.patch_size = patch_size
        self.tubelet_size = int(tubelet_size)
        num_patches = (
            (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0]) * (num_frames // self.tubelet_size)
        )
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.projection = nn.Conv3d(
            in_channels=num_channels,
            out_channels=hidden_size,
            kernel_size=(self.tubelet_size, patch_size[0], patch_size[1]),
            stride=(self.tubelet_size, patch_size[0], patch_size[1]),
        )

    def forward(self, pixel_values):
        batch_size, num_frames, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )
        # permute to (batch_size, num_channels, num_frames, height, width)
        pixel_values = pixel_values.permute(0, 2, 1, 3, 4)
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings


@MODELS.register_module()
class Transformer(BaseModule):
    def __init__(self,
                feat:int,
                out_feat:int,
                nhead:int,
                dropout:float=0.1,
                num_layers:int=5,
                max_seq_length: int = 200,
                add_cls_token:bool=True,
                freeze: bool = False,
                ):
        
        super(Transformer, self).__init__()
        self.freeze = freeze
        
        encoder_layer = nn.TransformerEncoderLayer(feat,nhead,dropout=dropout,dim_feedforward=out_feat,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.add_cls_token = add_cls_token

        # Positional embeddings buffer
        self.register_buffer('positional_encoding',
                            self.get_sinusoid_encoding(max_seq_length, feat))

        if self.add_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, feat))

    def get_sinusoid_encoding(self, n_position: int, embed_dims: int) -> Tensor:
        """Generate sinusoid encoding table."""
        vec = torch.arange(embed_dims, dtype=torch.float64)
        vec = (vec - vec % 2) / embed_dims
        vec = torch.pow(10000, -vec).view(1, -1)

        sinusoid_table = torch.arange(n_position).view(-1, 1) * vec
        sinusoid_table[:, 0::2].sin_()  # dim 2i
        sinusoid_table[:, 1::2].cos_()  # dim 2i+1

        sinusoid_table = sinusoid_table.to(torch.float32)

        return sinusoid_table.unsqueeze(0)  # Shape: (1, n_position, embed_dims)
    
    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                normal_init(m,0)
            elif isinstance(m, nn.Embedding):
                normal_init(m,0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m,1,0)
    
    def _freeze_stages(self):
        if self.freeze:
            for _,weights in self.named_parameters():
                    weights.requires_grad = False


    def forward(self,x:Tensor,data_samples=None):
        loss_aux = dict()
        batch_size, seq_length, _ = x.size()
        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)
            positional_encoding = self.positional_encoding[:, :seq_length+1, :].to(x.device)
        else:
            positional_encoding = self.positional_encoding[:, :seq_length, :].to(x.device)
        x = x + positional_encoding

        x = self.transformer_encoder(x)
        return x,loss_aux
    
    def train(self, mode: bool = True) -> None:
        """Convert the model into training mode while keep layers frozen."""
        super(Transformer, self).train(mode)
        self._freeze_stages()


class TransformerEncoderLayerWithAttention(nn.Module):
    def __init__(self, feat, nhead, dropout=0.1, dim_feedforward=2048):
        super(TransformerEncoderLayerWithAttention, self).__init__()
        self.self_attn = nn.MultiheadAttention(feat, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(feat, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, feat)
        self.norm1 = nn.LayerNorm(feat)
        self.norm2 = nn.LayerNorm(feat)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Store attention weights
        self.attn_weights = None

    def forward(self,src, src_mask=None, src_key_padding_mask=None,is_causal=None):
        # Self-attention with weights
        src2, attn_weights = self.self_attn(src, src, src, need_weights=True)
        self.attn_weights = attn_weights  # Save attention weights for later visualization

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

@MODELS.register_module()
class TransformerNew(BaseModule):
    def __init__(
            self,
            feat: int,
            out_feat: int,
            nhead: int,
            dropout: float = 0.1,
            num_layers: int = 5,
            max_seq_length: int = 200,
            add_cls_token: bool = True,
            freeze: bool = False,
            ):

        super(TransformerNew, self).__init__()

        # Replace TransformerEncoderLayer with the customized version
        encoder_layer = TransformerEncoderLayerWithAttention(feat, nhead, dropout=dropout, dim_feedforward=out_feat)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.add_cls_token = add_cls_token
        self.freeze = freeze

        # Positional embeddings
        self.register_buffer('positional_encoding', self.get_sinusoid_encoding(max_seq_length, feat))

        if self.add_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, feat))

    def get_sinusoid_encoding(self, n_position: int, embed_dims: int) -> Tensor:
        """Generate sinusoid encoding table."""
        vec = torch.arange(embed_dims, dtype=torch.float64)
        vec = (vec - vec % 2) / embed_dims
        vec = torch.pow(10000, -vec).view(1, -1)

        sinusoid_table = torch.arange(n_position).view(-1, 1) * vec
        sinusoid_table[:, 0::2].sin_()  # dim 2i
        sinusoid_table[:, 1::2].cos_()  # dim 2i+1

        sinusoid_table = sinusoid_table.to(torch.float32)

        return sinusoid_table.unsqueeze(0)  # Shape: (1, n_position, embed_dims)
    
    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                normal_init(m,0)
            elif isinstance(m, nn.Embedding):
                normal_init(m,0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m,1,0)

    def forward(self, x: Tensor, data_samples=None,output_attn_weights:bool=False):
        loss_aux = dict()
        batch_size, seq_length, _ = x.size()
        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            positional_encoding = self.positional_encoding[:, :seq_length + 1, :].to(x.device)
        else:
            positional_encoding = self.positional_encoding[:, :seq_length, :].to(x.device)

        x = x + positional_encoding
        x = self.transformer_encoder(x)

        # Extract attention weights from the first Transformer layer
        # first_layer = self.transformer_encoder.layers[0]
        # attn_weights = first_layer.attn_weights  # Shape: (batch_size, seq_length, seq_length)
        last_layer = self.transformer_encoder.layers[-1]
        attn_weights = last_layer.attn_weights
        if output_attn_weights:
            return x, loss_aux, attn_weights
        else:
            return x,loss_aux
        
    def _freeze_stages(self):
        if self.freeze:
            for _,weights in self.named_parameters():
                    weights.requires_grad = False
        

    def train(self, mode: bool = True) -> None:
        """Convert the model into training mode while keep layers frozen."""
        super(TransformerNew, self).train(mode)
        self._freeze_stages()



@MODELS.register_module()
class TransformerTubelet(BaseModule):
    def __init__(self,
                 feature_inchannels:int=768,
                 sample_frames:int=16,
                 tubelet_size:int=8,
                 tubelet_feature_size:int=128,
                 image_size:Tuple[int,int]=(14,14),
                 patch_size:Tuple[int,int]=(7,7),
                 transformer_heads:int = 2,
                 transformer_layers:int=1,
                 transformer_ffn:int = 256,
                 dropout:float=0.1,
                 add_cls_token:bool = True,
                 freeze:bool=False,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = [
                     dict(
                         type='TruncNormal', layer='Linear', std=0.02,
                         bias=0.),
                     dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
                     ]
                ):
        

        super().__init__(init_cfg=init_cfg)
        
        self.freeze = freeze
        self.add_cls_token = add_cls_token

        num_patches = (
            (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0]) * (sample_frames // tubelet_size)
        )

        encoder_layer = TransformerEncoderLayerWithAttention(feat=tubelet_feature_size, nhead=transformer_heads, dropout=dropout,
                                                              dim_feedforward=transformer_ffn)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        pos_embed = self.get_sinusoid_encoding(num_patches + 1, tubelet_feature_size) if self.add_cls_token else self.get_sinusoid_encoding(num_patches, tubelet_feature_size)
        self.register_buffer('pos_embed', pos_embed)
        
        self.patch = VideoMAEPatchEmbeddings(
            image_size=image_size,
            patch_size=patch_size,
            num_channels=feature_inchannels,
            hidden_size=tubelet_feature_size,
            num_frames=sample_frames,
            tubelet_size=tubelet_size,
            )

        if self.add_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, tubelet_feature_size))

    def get_sinusoid_encoding(self, n_position: int, embed_dims: int) -> Tensor:
        """Generate sinusoid encoding table."""
        vec = torch.arange(embed_dims, dtype=torch.float64)
        vec = (vec - vec % 2) / embed_dims
        vec = torch.pow(10000, -vec).view(1, -1)

        sinusoid_table = torch.arange(n_position).view(-1, 1) * vec
        sinusoid_table[:, 0::2].sin_()  # dim 2i
        sinusoid_table[:, 1::2].cos_()  # dim 2i+1

        sinusoid_table = sinusoid_table.to(torch.float32)

        return sinusoid_table.unsqueeze(0)  # Shape: (1, n_position, embed_dims)
        
    def _freeze_stages(self):
        if self.freeze:
            for _,weights in self.named_parameters():
                    weights.requires_grad = False


    def forward(self, x: Tensor, data_samples=None):
        loss_aux = dict()
        b,_,_,_,_ = x.shape
        x = self.patch(x)
        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(b, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.transformer_encoder(x)
        return x,loss_aux
    
    def train(self, mode: bool = True) -> None:
        """Convert the model into training mode while keep layers frozen."""
        super(TransformerTubelet, self).train(mode)
        self._freeze_stages()