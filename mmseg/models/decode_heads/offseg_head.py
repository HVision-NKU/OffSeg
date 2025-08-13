# ICCV 2025: Revisiting Efficient Semantic Segmentation: Learning Offsets for Better Spatial and Class Feature Alignment
# <https://arxiv.org/abs/2508.08811>
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.device import get_device

from mmseg.registry import MODELS
from ..utils import resize
from .decode_head import BaseDecodeHead
from .freqfusion import FreqFusion
from mmseg.models.decode_heads import Offset_Learning


@MODELS.register_module()
class OffSegHead(BaseDecodeHead):
    """OffSeg decode head.

    This decode head is the implementation of `Revisiting Efficient Semantic Segmentation: 
    Learning Offsets for Better Spatial and Class Feature Alignment
    <https://arxiv.org/abs/2508.08811>`.

    Args:
        in_channels (list): input channels for OffSeg.
        new_channels (list): hidden channels for OffSeg.
        num_classes (int): number of classes.
    """

    def __init__(self, 
                 in_channels,
                 new_channels,
                 num_classes,
                 **kwargs):
        super().__init__(in_channels=in_channels, 
                         num_classes=num_classes, 
                         input_transform='multiple_select', 
                         **kwargs)
        self.new_channels = new_channels

        self.pre = nn.ModuleList()
        for i in range(len(self.in_channels)):
            self.pre.append(
                ConvModule(self.in_channels[i],
                           self.new_channels[i],
                           1,
                           conv_cfg=self.conv_cfg,
                           norm_cfg=self.norm_cfg,
                           act_cfg=self.act_cfg)
            )
        
        self.freqfusions = nn.ModuleList()
        in_channels = new_channels[::-1]
        pre_c = in_channels[0]
        for c in in_channels[1:]:
            freqfusion = FreqFusion(
                hr_channels=c, lr_channels=pre_c, 
                compressed_channels= (pre_c + c) // 4,
                )                
            self.freqfusions.append(freqfusion)
            pre_c += c

        self.align = ConvModule(
            sum(self.new_channels),
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        
        # delattr(self, 'conv_seg')
        self.offset_learning = Offset_Learning(self.num_classes, self.channels)

    def forward(self, inputs):
        """Forward function."""
        inputs = self._transform_inputs(inputs)

        new_inputs = []
        for i in range(len(inputs)):
            new_inputs.append(self.pre[i](inputs[i]))

        inputs = new_inputs

        inputs = inputs[::-1]
        lowres_feat = inputs[0]
        for idx, (hires_feat, freqfusion) in enumerate(zip(inputs[1:], self.freqfusions)):
            _, hires_feat, lowres_feat = freqfusion(hr_feat=hires_feat, lr_feat=lowres_feat)
            b, _, h, w = hires_feat.shape
            lowres_feat = torch.cat([hires_feat.reshape(b * 4, -1, h, w), 
                                    lowres_feat.reshape(b * 4, -1, h, w)], dim=1).reshape(b, -1, h, w)

        inputs = lowres_feat

        output = self.align(inputs)
        output = self.offset_learning(output)
        return output
