import torch
import torch.nn as nn
from torch import Tensor
from mmengine.model import BaseModule
from mmaction.registry import MODELS
from mmengine.model.weight_init import normal_init,constant_init


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