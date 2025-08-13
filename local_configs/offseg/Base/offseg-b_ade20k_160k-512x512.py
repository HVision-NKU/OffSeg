_base_ = [
    '../../_base_/models/offseg.py',
    '../../_base_/datasets/ade20k.py',
    '../../_base_/default_runtime.py', 
    '../../_base_/schedules/schedule_160k.py',
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
ham_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
crop_size = (512, 512)

model = dict(
    type='EncoderDecoder',
    data_preprocessor = dict(size=crop_size),
    backbone=dict(
        type='efficientformerv2_s2_feat',
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
    test_cfg = dict(mode='slide', crop_size=(512, 512), stride=(480, 480))
    )

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=1)
find_unused_parameters=True
