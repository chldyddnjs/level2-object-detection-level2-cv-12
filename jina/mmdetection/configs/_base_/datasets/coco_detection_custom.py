# dataset settings
dataset_type = 'CocoDataset'
data_root = '/opt/ml/detection/dataset/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
#     mean = array([123.6506697 , 117.39730243, 110.07542563]),
# std = array([54.03457934, 53.36968771, 54.78390763])


# albu_train_transforms = [ # albu_example
#     dict(
#         type='RandomBrightnessContrast',
#         brightness_limit=[0.1, 0.3],
#         contrast_limit=[0.1, 0.3],
#         p=0.2),
#     dict(
#         type='OneOf',
#         transforms=[
#             dict(type='Blur', blur_limit=3, p=1.0),
#             dict(type='MedianBlur', blur_limit=3, p=1.0)
#         ],
#         p=0.1),
# ]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True), # 이미지 사이즈
    dict(type='RandomFlip', flip_ratio=0.5),
    # dict(
    #     type='Albu', # albumentation
    #     transforms=albu_train_transforms,
    #     bbox_params=dict(
    #         type='BboxParams',
    #         format='pascal_voc',
    #         label_fields=['gt_labels'],
    #         min_visibility=0.0,
    #         filter_lost_elements=True),
    #     keymap={
    #         'img': 'image',
    #         'gt_masks': 'masks',
    #         'gt_bboxes': 'bboxes'
    #     },
    #     update_pad_shape=False,
    #     skip_img_without_anno=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
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
data = dict(
    samples_per_gpu=2, # 배치 사이즈
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"),
        ann_file=data_root + 'fold_0_train.json',
        img_prefix=data_root,
        pipeline=train_pipeline),

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

evaluation = dict(interval=1, metric='bbox')
