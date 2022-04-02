# dataset settings
dataset_type = 'CocoDataset'
data_root = '../../dataset/' ### 실행하는 위치 기준
classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_norm_cfg = dict(
    mean=[123.6506697 , 117.39730243, 110.07542563], 
    std=[54.03457934, 53.36968771, 54.78390763], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=(512, 512),
        allow_negative_crop=True),    
    dict(type='Resize',
        img_scale=[(800, 800)],
        ratio_range=(0.5, 1.5),
        # multiscale_mode='range',
        keep_ratio=True), #이미지에 맞게 변경
    dict(
    type='CutOut',
    n_holes=(5, 10),
    cutout_shape=[(4, 4), (4, 8), (8, 4), (8, 8),
                    (16, 8), (8, 16), (16, 16), (16, 32), (32, 16), (32, 32),
                    (32, 48), (48, 32), (48, 48)],
    ),
    ###########################Albu############################################
    dict(type='Albu',
        transforms=[
            # dict(type='ShiftScaleRotate',
            #     shift_limit=0.0625,
            #     scale_limit=0.1,
            #     rotate_limit=0,
            #     interpolation=1,
            #     value=(1,1,1),
            #     p=0.5),
            dict(type='RandomRotate90', p=0.5),
            # dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='OneOf', transforms=[
                                dict(type='RandomFog', alpha_coef=0.25,  fog_coef_upper=0.5),
                                dict(type='Blur', blur_limit=(3, 7), p=1),
                                ], p=0.5
                ),
            dict(type='OneOf', transforms=[
                                dict(type='HueSaturationValue', 
                                    hue_shift_limit=(-20, 20), 
                                    sat_shift_limit=(-30,30), 
                                    val_shift_limit=(-20,20), p=1
                                    ),
                                dict(type='RandomBrightnessContrast', p=1),
                                ], p=0.5
                ),
            ],bbox_params=dict(
                type='BboxParams',
                format='pascal_voc',
                label_fields=['gt_labels'],
                min_visibility=0.2,
                filter_lost_elements=True),
            keymap=dict(img='image', gt_bboxes='bboxes'),
            update_pad_shape=False,
            skip_img_without_anno=True),
    #######################################################################
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),#이미지에 맞게 변경
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2, #batch size
    workers_per_gpu=2, # num of worker
    train=dict(
        type=dataset_type,
        classes = classes, 
        # ann_file=data_root + 'fold_0_train.json',
        img_prefix=data_root,
        pipeline=train_pipeline
        ),
    val=dict(
        type=dataset_type,
        classes = classes, 
        # ann_file=data_root + 'fold_0_val.json',
        img_prefix=data_root,
        pipeline=test_pipeline
        ),
    test=dict(
        type=dataset_type,
        classes = classes, 
        ann_file=data_root + 'test.json',
        img_prefix=data_root,
        pipeline=test_pipeline
        )
    )
evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP_50', classwise=True)

##
# evaluation save_best mAP 추가