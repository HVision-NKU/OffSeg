import torch
import torch.nn as nn
from mmengine.model.weight_init import (constant_init, trunc_normal_,
                                        trunc_normal_init)
from mmcv.cnn import build_norm_layer

class Offset_Learning(nn.Module):
    """
    ICCV 2025: Revisiting Efficient Semantic Segmentation: Learning Offsets for Better Spatial and Class Feature Alignment
    <https://arxiv.org/abs/2508.08811>
    """
    def __init__(self, num_classes, embed_dims, init_std=0.02, norm_cfg=dict(type='LN'),):
        super(Offset_Learning, self).__init__()
        self.num_classes = num_classes
        self.cls_repr = nn.Parameter(
            torch.randn(1, num_classes, embed_dims))
        self.init_std = init_std
        self.mask_norm = build_norm_layer(
            norm_cfg, self.num_classes, postfix=1)[1]
        self.cls_offset_proj = nn.Linear(embed_dims, embed_dims, bias=False)
        self.feat_offset_proj = nn.Linear(embed_dims, embed_dims, bias=False)
        
        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.cls_repr, std=self.init_std)
        trunc_normal_init(self.cls_offset_proj, std=self.init_std)
        trunc_normal_init(self.feat_offset_proj, std=self.init_std)
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=self.init_std, bias=0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)

    def forward(self, x):
        b, c, h, w = x.shape
        cls_repr = self.cls_repr.expand(b, -1, -1)  # b, k, c
        img_feat = x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)  # b, hw, c

        # compute coupled attention
        coupled_attn = img_feat @ cls_repr.transpose(1, 2)  # b, hw, k
        coupled_attn = coupled_attn.permute(0, 2, 1)  # b, k, hw

        # class offset learning
        cls_attn = coupled_attn.softmax(dim=2)  # b, k, hw
        cls_offset = self.cls_offset_proj(cls_attn @ img_feat)  # b, k, c
        aligned_cls_repr = cls_repr + cls_offset  # b, k, c

        # feature offset learning
        pos_attn = coupled_attn.softmax(dim=1)  # b, k, hw
        feat_offset = self.feat_offset_proj(pos_attn.transpose(1, 2) @ cls_repr)  # b, hw, c
        aligned_img_feat = img_feat + feat_offset  # b, hw, c

        # compute masks
        masks = aligned_img_feat @ aligned_cls_repr.transpose(1, 2)  # b, hw, k
        masks = self.mask_norm(masks)
        masks = masks.permute(0, 2, 1).contiguous().view(b, -1, h, w)
        return masks

class Offset_Learning_Mask(nn.Module):
    """
    ICCV 2025: Revisiting Efficient Semantic Segmentation: Learning Offsets for Better Spatial and Class Feature Alignment
    <https://arxiv.org/abs/2508.08811>
    """
    def __init__(self, embed_dims, init_std=0.02, norm_cfg=dict(type='LN'),):
        super(Offset_Learning_Mask, self).__init__()
        self.init_std = init_std
        self.cls_offset_proj = nn.Linear(embed_dims, embed_dims, bias=False)
        self.feat_offset_proj = nn.Linear(embed_dims, embed_dims, bias=False)        
        self.init_weights()
    def init_weights(self):
        trunc_normal_init(self.cls_offset_proj, std=self.init_std)
        trunc_normal_init(self.feat_offset_proj, std=self.init_std)
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=self.init_std, bias=0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)
    def forward(self, mask_embed, mask_features):
        b, c, h, w = mask_features.shape
        cls_repr = mask_embed  # b, k, c
        img_feat = mask_features.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)  # b, hw, c

        # compute coupled attention
        coupled_attn = img_feat @ cls_repr.transpose(1, 2)  # b, hw, k
        coupled_attn = coupled_attn.permute(0, 2, 1)  # b, k, hw

        # class offset learning
        cls_attn = coupled_attn.softmax(dim=2)
        cls_offset = self.cls_offset_proj(cls_attn @ img_feat)
        aligned_cls_repr = cls_repr + cls_offset

        # feature offset learning
        pos_attn = coupled_attn.softmax(dim=1)
        feat_offset = self.feat_offset_proj(pos_attn.transpose(1, 2) @ cls_repr)
        aligned_img_feat = img_feat + feat_offset
        aligned_img_feat = aligned_img_feat.transpose(1, 2).contiguous().view(b, c, h, w)

        return aligned_cls_repr, aligned_img_feat
