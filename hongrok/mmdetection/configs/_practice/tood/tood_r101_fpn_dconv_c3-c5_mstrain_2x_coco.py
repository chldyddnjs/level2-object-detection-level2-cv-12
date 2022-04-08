_base_ = './tood_r101_fpn_mstrain_2x_coco.py'

model = dict(
    backbone=dict(
        dcn=dict(type='DCNv2', deformable_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
        # init_cfg=dict(type='Pretrained',
        #               checkpoint='https://download.openmmlab.com/mmdetection/v2.0/tood/tood_r101_fpn_dconv_c3-c5_mstrain_2x_coco/tood_r101_fpn_dconv_c3-c5_mstrain_2x_coco_20211210_213728-4a824142.pth'
        #               )
                ),
    bbox_head=dict(num_dcn=2))
