import torch
from transformers import LongformerModel, LongformerConfig
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmaction.registry import MODELS
import torch.nn as nn
from typing import Optional

@MODELS.register_module()
class VTNLongformerModel(BaseModule):
    def __init__(self,
                 embed_dim,
                 max_position_embeddings,
                 num_attention_heads,
                 num_hidden_layers,
                 attention_mode,
                 pad_token_id,
                 attention_window,
                 intermediate_size,
                 attention_probs_dropout_prob,
                 hidden_dropout_prob,
                 init_cfg=None,
                 freeze: bool = False
                 ):
        self.embed_dim = embed_dim
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.attention_mode = attention_mode
        self.pad_token_id = pad_token_id
        self.attention_window = attention_window
        self.intermediate_size = intermediate_size
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.freeze = freeze
        


        assert len(self.attention_window)== self.num_hidden_layers
        assert (self.embed_dim%self.num_attention_heads) == 0

        super(VTNLongformerModel,self).__init__(init_cfg=init_cfg)

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.temporal_encoder = VTNLongformerModelHelper(
            embed_dim=self.embed_dim,
            max_position_embeddings=self.max_position_embeddings,
            num_attention_heads=self.num_attention_heads,
            num_hidden_layers=self.num_hidden_layers,
            attention_mode=self.num_hidden_layers,
            pad_token_id=self.pad_token_id,
            attention_window=self.attention_window,
            intermediate_size=self.intermediate_size,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            hidden_dropout_prob=self.hidden_dropout_prob)
        


    def forward(self,x,data_samples=None) -> tuple:
        loss_aux = dict()

        position_ids = None
        if isinstance(x,tuple):
            x, position_ids = x
        

        # temporal encoder (Longformer)
        B, D, E  = x.shape

        attention_mask = torch.ones((B, D), dtype=torch.long, device=x.device)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        cls_atten = torch.ones(1).expand(B, -1).to(x.device)
        attention_mask = torch.cat((attention_mask, cls_atten), dim=1)
        attention_mask[:, 0] = 2
        x, attention_mask, position_ids = pad_to_window_size_local(
            x,
            attention_mask,
            position_ids,
            self.temporal_encoder.config.attention_window[0],
            self.temporal_encoder.config.pad_token_id)
        
        token_type_ids = torch.zeros(x.size()[:-1], dtype=torch.long, device=x.device)
        token_type_ids[:, 0] = 1

        # position_ids
        if isinstance(position_ids,torch.Tensor):
            position_ids = position_ids.long()
            mask = attention_mask.ne(0).int()
            max_position_embeddings = torch.tensor(self.temporal_encoder.config.max_position_embeddings)
            position_ids = position_ids % (max_position_embeddings - 2)
            position_ids[:, 0] = max_position_embeddings - 2
            position_ids[mask == 0] = max_position_embeddings - 1




        x = self.temporal_encoder(input_ids=None,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids,
                                  position_ids=position_ids,
                                  inputs_embeds=x,
                                  output_attentions=None,
                                  output_hidden_states=None,
                                  return_dict=None)
        # MLP head
        x = x["last_hidden_state"] # type: ignore
        return x,loss_aux
    
    def _freeze_stages(self):
        if self.freeze:
            for _,weights in self.named_parameters():
                    weights.requires_grad = False


    def train(self, mode: bool = True) -> None:
        """Convert the model into training mode while keep layers frozen."""
        super(VTNLongformerModel, self).train(mode)
        self._freeze_stages()



class VTNLongformerModelHelper(LongformerModel):

    def __init__(self,
                 embed_dim=768,
                 max_position_embeddings=2 * 60 * 60,
                 num_attention_heads=12,
                 num_hidden_layers=3,
                 attention_mode='sliding_chunks',
                 pad_token_id=-1,
                 attention_window=None,
                 intermediate_size=3072,
                 attention_probs_dropout_prob=0.1,
                 hidden_dropout_prob=0.1):

        self.config = LongformerConfig()
        self.config.attention_mode = attention_mode
        self.config.intermediate_size = intermediate_size
        self.config.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.config.hidden_dropout_prob = hidden_dropout_prob
        self.config.attention_dilation = [1, ] * num_hidden_layers
        self.config.attention_window = [256, ] * num_hidden_layers if attention_window is None else attention_window
        self.config.num_hidden_layers = num_hidden_layers
        self.config.num_attention_heads = num_attention_heads
        self.config.pad_token_id = pad_token_id
        self.config.max_position_embeddings = max_position_embeddings
        self.config.hidden_size = embed_dim
        super(VTNLongformerModelHelper, self).__init__(self.config, add_pooling_layer=False)
        self.embeddings.word_embeddings = None  # to avoid distributed error of unused parameters


def pad_to_window_size_local(input_ids: torch.Tensor, attention_mask: torch.Tensor, position_ids: Optional[torch.Tensor],
                             one_sided_window_size: int, pad_token_id: int):
    '''A helper function to pad tokens and mask to work with the sliding_chunks implementation of Longformer self-attention.
    Based on _pad_to_window_size from https://github.com/huggingface/transformers:
    https://github.com/huggingface/transformers/blob/71bdc076dd4ba2f3264283d4bc8617755206dccd/src/transformers/models/longformer/modeling_longformer.py#L1516
    Input:
        input_ids = torch.Tensor(bsz x seqlen): ids of wordpieces
        attention_mask = torch.Tensor(bsz x seqlen): attention mask
        one_sided_window_size = int: window size on one side of each token
        pad_token_id = int: tokenizer.pad_token_id
    Returns
        (input_ids, attention_mask) padded to length divisible by 2 * one_sided_window_size
    '''
    w = 2 * one_sided_window_size
    seqlen = input_ids.size(1)
    padding_len = (w - seqlen % w) % w
    input_ids = F.pad(input_ids.permute(0, 2, 1), (0, padding_len), value=pad_token_id).permute(0, 2, 1)
    attention_mask = F.pad(attention_mask, (0, padding_len), value=False)  # no attention on the padding tokens
    if position_ids is not None:
        position_ids = F.pad(position_ids, (1, padding_len), value=False)  # no attention on the padding tokens
    return input_ids, attention_mask, position_ids
