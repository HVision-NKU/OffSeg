# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
ham_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
data_preprocessor = dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='efficientformerv2_s2_feat',
        style='pytorch',
        init_cfg=dict(type='Pretrained',checkpoint='pretrained/eformer_v2/eformer_s2_450.pth')),
    decode_head=dict(
        type='OffSegHead',
        in_channels=[32, 64, 144, 288],
        new_channels=[32, 64, 128, 256],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=ham_norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg = dict(mode='slide', crop_size=(512, 512), stride=(480, 480)))
