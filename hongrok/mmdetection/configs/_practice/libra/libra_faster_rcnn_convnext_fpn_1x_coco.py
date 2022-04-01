_base_ = './libra_faster_rcnn_r50_fpn_1x_coco.py'
model = dict(
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
    #         type='Pretrained', checkpoint='open-mmlab://resnext101_64x4d'))
    # 
    # backbone=dict(
    #     #base
    #     backbone=dict(
    #     type='ConvNeXt',
    #     pretrained= 'https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth',
    #     in_chans=3,
    #     depths=[3, 3, 27, 3], 
    #     dims=[128, 256, 512, 1024], 
    #     drop_path_rate=0.6,
    #     layer_scale_init_value=1.0,
    #     out_indices=[0, 1, 2, 3],
    #     ),
    #     neck=dict(in_channels=[128, 256, 512, 1024]),

        #large
    backbone=dict(
        type='ConvNeXt', 
        pretrained='https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth',
        in_chans=3,
        depths=[3, 3, 27, 3], 
        dims=[192, 384, 768, 1536], 
        drop_path_rate=0.7,
        layer_scale_init_value=1.0,
        out_indices=[0, 1, 2, 3],
        ),
        # neck=dict(in_channels=[192, 384, 768, 1536]),
        
        # #x-large
    # backbone=dict(
        # type='ConvNeXt', 
        # pretrained='https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth',
        # in_chans=3,
        # depths=[3, 3, 27, 3], 
        # dims=[256, 512, 1024, 2048], 
        # drop_path_rate=0.8,
        # layer_scale_init_value=1.0,
        # out_indices=[0, 1, 2, 3],
        # ),
        # neck=dict(in_channels=[256, 512, 1024, 2048]),
    )
