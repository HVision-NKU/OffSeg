_base_ = '../Tiny/segnext_mscan-t_offset_learning_80k_pascal-context-59-480x480.py'
# model settings
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_s_20230227-f33ccdf2.pth'  # noqa
ham_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        embed_dims=[64, 128, 320, 512],
        depths=[2, 2, 4, 2],
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        norm_cfg=dict(type='BN', requires_grad=True)),
    decode_head=dict(
        type='LightHamHeadOffsetLearning',
        in_channels=[128, 320, 512],
        in_index=[1, 2, 3],
        channels=256,
        ham_channels=256,
        ham_kwargs=dict(MD_R=16),
        dropout_ratio=0.1,
        num_classes=59,
        norm_cfg=ham_norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(480, 480), stride=(320, 320)))
