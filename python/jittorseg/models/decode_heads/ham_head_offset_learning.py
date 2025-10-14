import jittor as jt
from jittor import nn
from ..layers import resize
from jittorseg.utils.registry import HEADS
from ..basic_operators import ConvModule, build_norm_layer
from .decode_head import BaseDecodeHead
from jittorseg.utils.weight_init import trunc_normal_init, constant_init, trunc_normal_


class Offset_Learning(nn.Module):
    """
    ICCV 2025: Revisiting Efficient Semantic Segmentation: Learning Offsets for Better Spatial and Class Feature Alignment
    <https://arxiv.org/abs/2508.08811>
    """
    def __init__(self, num_classes, embed_dims, init_std=0.02, norm_cfg=dict(type='LN'),):
        super(Offset_Learning, self).__init__()
        self.num_classes = num_classes
        self.cls_repr = nn.Parameter(
            jt.rand(1, num_classes, embed_dims))
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

    def execute(self, x):
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


class _MatrixDecomposition2DBase(nn.Module):

    def __init__(self, args=dict()):
        super().__init__()

        self.spatial = args.setdefault('SPATIAL', True)

        self.S = args.setdefault('MD_S', 1)
        self.D = args.setdefault('MD_D', 512)
        self.R = args.setdefault('MD_R', 64)

        self.train_steps = args.setdefault('TRAIN_STEPS', 6)
        self.eval_steps = args.setdefault('EVAL_STEPS', 7)

        self.inv_t = args.setdefault('INV_T', 100)
        self.eta = args.setdefault('ETA', 0.9)

        self.rand_init = args.setdefault('RAND_INIT', True)

        print('spatial', self.spatial)
        print('S', self.S)
        print('D', self.D)
        print('R', self.R)
        print('train_steps', self.train_steps)
        print('eval_steps', self.eval_steps)
        print('inv_t', self.inv_t)
        print('eta', self.eta)
        print('rand_init', self.rand_init)

    def _build_bases(self, B, S, D, R):
        raise NotImplementedError

    def local_step(self, x, bases, coef):
        raise NotImplementedError

    @jt.no_grad()
    def local_inference(self, x, bases):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        coef = nn.bmm(x.transpose(1, 2), bases)
        coef = nn.softmax(self.inv_t * coef, dim=-1)

        steps = self.train_steps if self.is_training() else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        raise NotImplementedError

    def execute(self, x, return_bases=False):
        B, C, H, W = x.shape

        # (B, C, H, W) -> (B * S, D, N)
        if self.spatial:
            D = C // self.S
            N = H * W
            x = x.view(B * self.S, D, N)
        else:
            D = H * W
            N = C // self.S
            x = x.view(B * self.S, N, D).transpose(1, 2)

        if not self.rand_init and not hasattr(self, 'bases'):
            bases = self._build_bases(1, self.S, D, self.R)
            self.register_buffer('bases', bases)

        # (S, D, R) -> (B * S, D, R)
        if self.rand_init:
            bases = self._build_bases(B, self.S, D, self.R)
        else:
            bases = self.bases.repeat(B, 1, 1)

        bases, coef = self.local_inference(x, bases)

        # (B * S, N, R)
        coef = self.compute_coef(x, bases, coef)

        # (B * S, D, R) @ (B * S, N, R)^T -> (B * S, D, N)
        x = nn.bmm(bases, coef.transpose(1, 2))

        # (B * S, D, N) -> (B, C, H, W)
        if self.spatial:
            x = x.view(B, C, H, W)
        else:
            x = x.transpose(1, 2).view(B, C, H, W)

        # (B * H, D, R) -> (B, H, N, D)
        bases = bases.view(B, self.S, D, self.R)

        return x


class NMF2D(_MatrixDecomposition2DBase):

    def __init__(self, args=dict()):
        super().__init__(args)

        self.inv_t = 1

    def _build_bases(self, B, S, D, R):
        bases = jt.rand((B * S, D, R))

        bases = jt.normalize(bases, dim=1)

        return bases

    def local_step(self, x, bases, coef):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = nn.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ [(B * S, D, R)^T @ (B * S, D, R)] -> (B * S, N, R)
        denominator = nn.bmm(coef, nn.bmm(bases.transpose(1, 2), bases))
        # Multiplicative Update
        coef = coef * numerator / (denominator + 1e-6)

        # (B * S, D, N) @ (B * S, N, R) -> (B * S, D, R)
        numerator = nn.bmm(x, coef)
        # (B * S, D, R) @ [(B * S, N, R)^T @ (B * S, N, R)] -> (B * S, D, R)
        denominator = nn.bmm(bases, nn.bmm(coef.transpose(1, 2), coef))
        # Multiplicative Update
        bases = bases * numerator / (denominator + 1e-6)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = nn.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ (B * S, D, R)^T @ (B * S, D, R) -> (B * S, N, R)
        denominator = nn.bmm(coef, nn.bmm(bases.transpose(1, 2), bases))
        # multiplication update
        coef = coef * numerator / (denominator + 1e-6)

        return coef


class Hamburger(nn.Module):

    def __init__(self,
                 ham_channels=512,
                 ham_kwargs=dict(),
                 norm_cfg=None,
                 **kwargs):
        super().__init__()

        self.ham_in = ConvModule(ham_channels,
                                 ham_channels,
                                 1,
                                 norm_cfg=None,
                                 act_cfg=None)

        self.ham = NMF2D(ham_kwargs)

        self.ham_out = ConvModule(ham_channels,
                                  ham_channels,
                                  1,
                                  norm_cfg=norm_cfg,
                                  act_cfg=None)

    def execute(self, x):
        enjoy = self.ham_in(x)
        enjoy = nn.relu(enjoy)
        enjoy = self.ham(enjoy)
        enjoy = self.ham_out(enjoy)
        ham = nn.relu(x + enjoy)

        return ham


@HEADS.register_module()
class LightHamHeadOffsetLearning(BaseDecodeHead):

    def __init__(self, ham_channels=512, ham_kwargs=dict(), **kwargs):
        super(LightHamHeadOffsetLearning, self).__init__(input_transform='multiple_select',
                                           **kwargs)
        self.ham_channels = ham_channels

        self.squeeze = ConvModule(sum(self.in_channels),
                                  self.ham_channels,
                                  1,
                                  conv_cfg=self.conv_cfg,
                                  norm_cfg=self.norm_cfg,
                                  act_cfg=self.act_cfg)

        self.hamburger = Hamburger(ham_channels, ham_kwargs, **kwargs)

        self.align = ConvModule(self.ham_channels,
                                self.channels,
                                1,
                                conv_cfg=self.conv_cfg,
                                norm_cfg=self.norm_cfg,
                                act_cfg=self.act_cfg)

        self.offset_learning = Offset_Learning(self.num_classes, self.channels)
        delattr(self, 'conv_seg')

    def execute(self, inputs):
        inputs = self._transform_inputs(inputs)

        inputs = [
            resize(level,
                   size=inputs[0].shape[2:],
                   mode='bilinear',
                   align_corners=self.align_corners) for level in inputs
        ]

        inputs = jt.concat(inputs, dim=1)
        x = self.squeeze(inputs)

        x = self.hamburger(x)

        output = self.align(x)
        output = self.offset_learning(output)
        return output
