_base_ = [
    './cascade_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    type='CascadeRCNN',
    # backbone=dict(
    #     type='ResNeXt',
    #     depth=101,
    #     groups=64,
    #     base_width=4,
    #     num_stages=4,
    #     out_indices=(0, 1, 2, 3),
    #     frozen_stages=1,
    #     norm_cfg=dict(type='BN', requires_grad=True),
    #     style='pytorch',
    #     init_cfg=dict(
    #         type='Pretrained', checkpoint='open-mmlab://resnext101_64x4d')
    #         )

    #large
    backbone=dict(
        type='ConvNeXt', 
        in_chans=3,
        depths=[3, 3, 27, 3], 
        dims=[192, 384, 768, 1536], 
        drop_path_rate=0.7,
        layer_scale_init_value=1.0,
        out_indices=[0, 1, 2, 3],
        init_cfg=dict(
            type='Pretrained', checkpoint='https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth')
        ),
        neck=dict(in_channels=[192, 384, 768, 1536]),
    # # x-large
    # backbone=dict(
    #     type='ConvNeXt', 
    #     in_chans=3,
    #     depths=[3, 3, 27, 3], 
    #     dims=[256, 512, 1024, 2048], 
    #     drop_path_rate=0.8,
    #     layer_scale_init_value=1.0,
    #     out_indices=[0, 1, 2, 3],
    #     init_cfg=dict(
    #         type='Pretrained', checkpoint='https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth')
    #     ),
    #     neck=dict(in_channels=[256, 512, 1024, 2048]),
)
