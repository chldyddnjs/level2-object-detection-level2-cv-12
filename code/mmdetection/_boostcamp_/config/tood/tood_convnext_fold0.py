_base_ = [
    '../_base_/models/tood_r50_fpn_anchor_based_1x_coco.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
pretrained = 'https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_384.pth'
  # noqa
model = dict(
    backbone=dict(
        type='ConvNeXt',
        in_chans=3,
        depths=[3, 3, 27, 3], 
        dims=[128, 256, 512, 1024], 
        drop_path_rate=0.2,
        layer_scale_init_value=1e-6,
        out_indices=[0, 1, 2, 3],
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        type='FPN',
        in_channels=[128, 256, 512, 1024],
        out_channels=256,
        num_outs=5),
    )