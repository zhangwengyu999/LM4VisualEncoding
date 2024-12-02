import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm
from pathlib import Path
from timm.models.layers import DropPath, trunc_normal_
from .dvae import Group
from .dvae import DiscreteVAE, Encoder
from .llama import LLaMATransformer

from mamba_ssm import Mamba  # Import the Mamba module

from .build import MODELS
from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *

import numpy as np
import random


@MODELS.register_module()
class PointMamba(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth 
        self.drop_path_rate = config.drop_path_rate 
        self.cls_dim = config.cls_dim 
        self.num_heads = config.num_heads 

        self.group_size = config.group_size
        self.num_group = config.num_group
        # grouper
        # self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
        # define the encoder
        self.encoder_dims =  config.encoder_dims
        
        self.encoder = nn.Linear(3, self.trans_dim)
        # self.encoder = Encoder(encoder_channel = self.encoder_dims)
        # bridge encoder and transformer
        # self.reduce_dim = nn.Linear(self.encoder_dims,  self.trans_dim)

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        # self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        
        # dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        # self.blocks = TransformerEncoder(
        #     embed_dim = self.trans_dim,
        #     depth = self.depth,
        #     drop_path_rate = dpr,
        #     num_heads = self.num_heads
        # )
        
        # Replace TransformerEncoder with Mamba2
        # https://github.com/state-spaces/mamba
        self.blocks = nn.ModuleList([
            Mamba(
                d_model=self.trans_dim,  # Use the transformer dimension
                d_state=64,             # SSM state expansion factor
                d_conv=4,               # Local convolution width
                expand=2                # Block expansion factor
            )
            for _ in range(self.depth)  # Depth of the Mamba block
        ])

        self.norm = nn.LayerNorm(self.trans_dim)

        self.cls_head_finetune = nn.Sequential(
            nn.Linear(self.trans_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.cls_dim)
        )

        if hasattr(config, 'use_llama') and config.use_llama:
            llama_default_config = dict(config.llama_cfg)
            self.llama = LLaMATransformer(llama_default_config)
            for param in self.llama.parameters():
                param.requires_grad = False
            self.llama_dim_mapper1 = nn.Linear(config.trans_dim, 4096, bias=False)
            self.llama_dim_mapper2 = nn.Linear(4096, config.trans_dim, bias=False)


        self.build_loss_func()
        
    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()
    
    def get_loss_acc(self, pred, gt, smoothing=True):
        # import pdb; pdb.set_trace()
        gt = gt.contiguous().view(-1).long()

        if smoothing:
            eps = 0.2
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, gt.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = self.loss_ce(pred, gt.long())

        pred = pred.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))

        return loss, acc * 100


    def load_model_from_ckpt(self, bert_ckpt_path):
        ckpt = torch.load(bert_ckpt_path)
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        for k in list(base_ckpt.keys()):
            if k.startswith('transformer_q') and not k.startswith('transformer_q.cls_head'):
                base_ckpt[k[len('transformer_q.'):]] = base_ckpt[k]
            elif k.startswith('base_model'):
                base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
            del base_ckpt[k]


        incompatible = self.load_state_dict(base_ckpt, strict=False)

        if incompatible.missing_keys:
            print_log('missing_keys', logger = 'Transformer')
            print_log(
                get_missing_parameters_message(incompatible.missing_keys),
                logger = 'Transformer'
            )
        if incompatible.unexpected_keys:
            print_log('unexpected_keys', logger = 'Transformer')
            print_log(
                get_unexpected_parameters_message(incompatible.unexpected_keys),
                logger = 'Transformer'
            )

        if hasattr(self.config, 'use_llama') and self.config.use_llama:
            print_log("Loading LLaMA checkpoints", logger = 'LLaMA')
            checkpoints = sorted(Path(self.config.llama_path).glob("*.pth"))
            ckpt_path = checkpoints[0]
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            self.llama.custom_load_state_dict(checkpoint, tail=True, strict=False)
            print_log("Loading LLaMA Done", logger = 'LLaMA')

        print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger = 'Transformer')


    def forward(self, pts):
        # pts: B N 3
        # print(">>>1 ", pts.shape)
        x = self.encoder(pts)
        # print(">>>2 ", pts_encoded.shape)
        # x = self.reduce_dim(pts_encoded)
        # print(">>>3 ", x.shape)
        # Pass through Mamba2 blocks
        for block in self.blocks:
            x = block(x)
        # print(">>>4 ", x.shape)
        
        if hasattr(self.config, 'use_llama') and self.config.use_llama:
            x = self.llama_dim_mapper1(x)
            x = self.llama(x)
            x = self.llama_dim_mapper2(x)
        
        # print(">>>5 ", x.shape)
        x = self.norm(x)
        # concat_f = torch.cat([x[:,0], x[:, 1:].max(1)[0]], dim = -1)
        # print(">>>6 ", x.shape) # torch.Size([32, 1024, 384])
        global_features = torch.max(x, dim=1)[0]  # Max pooling across points
        # print(">>>7 Global Features: ", global_features.shape)  # [B, D]
        ret = self.cls_head_finetune(global_features) 
        # print(">>>7 ", ret.shape) # torch.Size([32, 1024, 40])
        return ret

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output