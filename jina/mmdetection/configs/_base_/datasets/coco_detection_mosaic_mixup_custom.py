# dataset settings
dataset_type = 'CocoDataset'
data_root = '/opt/ml/detection/dataset/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
#     mean = array([123.6506697 , 117.39730243, 110.07542563]),
# std = array([54.03457934, 53.36968771, 54.78390763])

albu_train_transforms = [
    dict(
    type='OneOf',
    transforms=[
        dict(type='RandomRotate90',p=1.0),
        dict(type='Flip',p=1.0),
    ],
    p=0.5),
    # dict(type='RandomResizedCrop',height=1024, width=1024, scale=(0.5, 1.0), p=0.5),
    dict(type='HueSaturationValue', hue_shift_limit=15, sat_shift_limit=15, val_shift_limit=10, p=0.5),
    dict(type='RandomBrightnessContrast', p=0.5),
    dict(type='GaussNoise', p=0.3),
    dict(
    type='OneOf',
    transforms=[
        dict(type='Blur', blur_limit=3, p=1.0),
        dict(type='GaussianBlur', p=1.0),
        dict(type='MedianBlur', blur_limit=3, p=1.0),
    ],
    p=0.2)
]

img_scale = (1024, 1024)
train_pipeline = [
    dict(
        type='Albu', # albumentation
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(type='MixUp', img_scale=(1024, 1024)), # Mixup
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)), # The image will be enlarged by 4 times after Mosaic processing,so we use affine transformation to restore the image size.
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024), # 이미지 사이즈
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

train_dataset = dict(
    # _delete_ = True, 
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"),
        ann_file=data_root + 'fold_0_train.json',
        img_prefix=data_root,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_empty_gt=False,
    ),
    pipeline=train_pipeline,

)
    
data = dict(
    samples_per_gpu=2, # batch size
    workers_per_gpu=1,
    train=train_dataset,

    val=dict(
        type=dataset_type,
        classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"),
        ann_file=data_root + 'fold_0_val.json',
        img_prefix=data_root,
        pipeline=test_pipeline),

    test=dict(
        type=dataset_type,
        classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"),
        ann_file=data_root + 'test.json',
        img_prefix=data_root,
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='bbox', save_best = 'bbox_mAP_50', classwise=True)
