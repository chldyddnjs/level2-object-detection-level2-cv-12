model = dict(
    type='TOOD',
    neck=dict(
        type='FPN',
        in_channels=[128, 256, 512, 1024],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='TOODHead',
        num_classes=10,
        in_channels=256,
        stacked_convs=6,
        feat_channels=256,
        anchor_type='anchor_based',
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        initial_loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            activated=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            activated=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
    train_cfg=dict(
        initial_epoch=4,
        initial_assigner=dict(type='ATSSAssigner', topk=9),
        assigner=dict(type='TaskAlignedAssigner', topk=13),
        alpha=1,
        beta=6,
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100),
    backbone=dict(
        type='ConvNeXt',
        in_chans=3,
        depths=[3, 3, 27, 3],
        dims=[128, 256, 512, 1024],
        drop_path_rate=0.2,
        layer_scale_init_value=1e-06,
        out_indices=[0, 1, 2, 3],
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_384.pth'
        )))
dataset_type = 'CocoDataset'
data_root = '/opt/ml/detection/dataset/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
albu_train_transforms = [
    dict(type='RandomRotate90', p=0.4),
    dict(
        type='HueSaturationValue',
        hue_shift_limit=23,
        sat_shift_limit=30,
        val_shift_limit=25,
        p=0.4),
    dict(type='RandomBrightnessContrast', p=0.4),
    dict(type='RandomGamma', gamma_limit=(80.0, 148.0), p=0.4),
    dict(
        type='OneOf',
        transforms=[
            dict(type='GaussianBlur', p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.2)
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Albu',
        transforms=[
            dict(type='RandomRotate90', p=0.4),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=23,
                sat_shift_limit=30,
                val_shift_limit=25,
                p=0.4),
            dict(type='RandomBrightnessContrast', p=0.4),
            dict(type='RandomGamma', gamma_limit=(80.0, 148.0), p=0.4),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='GaussianBlur', p=1.0),
                    dict(type='MedianBlur', blur_limit=3, p=1.0)
                ],
                p=0.2)
        ],
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap=dict(img='image', gt_bboxes='bboxes'),
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
    train=dict(
        type='CocoDataset',
        classes=('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
                 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing'),
        ann_file='/opt/ml/detection/dataset/fold_2_train.json',
        img_prefix='/opt/ml/detection/dataset/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Albu',
                transforms=[
                    dict(type='RandomRotate90', p=0.4),
                    dict(
                        type='HueSaturationValue',
                        hue_shift_limit=23,
                        sat_shift_limit=30,
                        val_shift_limit=25,
                        p=0.4),
                    dict(type='RandomBrightnessContrast', p=0.4),
                    dict(type='RandomGamma', gamma_limit=(80.0, 148.0), p=0.4),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(type='GaussianBlur', p=1.0),
                            dict(type='MedianBlur', blur_limit=3, p=1.0)
                        ],
                        p=0.2)
                ],
                bbox_params=dict(
                    type='BboxParams',
                    format='pascal_voc',
                    label_fields=['gt_labels'],
                    min_visibility=0.0,
                    filter_lost_elements=True),
                keymap=dict(img='image', gt_bboxes='bboxes'),
                update_pad_shape=False,
                skip_img_without_anno=True),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]),
    val=dict(
        type='CocoDataset',
        classes=('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
                 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing'),
        ann_file='/opt/ml/detection/dataset/fold_2_val.json',
        img_prefix='/opt/ml/detection/dataset/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CocoDataset',
        classes=('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
                 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing'),
        ann_file='/opt/ml/detection/dataset/test.json',
        img_prefix='/opt/ml/detection/dataset/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
evaluation = dict(
    interval=1, metric='bbox', save_best='bbox_mAP_50', classwise=True)
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=1e-05)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.01,
    min_lr=1e-06)
runner = dict(type='EpochBasedRunner', max_epochs=24)
checkpoint_config = dict(max_keep_ckpts=3, interval=1)
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
pretrained = 'https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_384.pth'
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='detection',
                entity='yolo12',
                name='Rch_fold_2_tood_convnext_augment',
                config=dict(
                    model=dict(
                        type='TOOD',
                        neck=dict(
                            type='FPN',
                            in_channels=[128, 256, 512, 1024],
                            out_channels=256,
                            start_level=1,
                            add_extra_convs='on_output',
                            num_outs=5),
                        bbox_head=dict(
                            type='TOODHead',
                            num_classes=10,
                            in_channels=256,
                            stacked_convs=6,
                            feat_channels=256,
                            anchor_type='anchor_based',
                            anchor_generator=dict(
                                type='AnchorGenerator',
                                ratios=[1.0],
                                octave_base_scale=8,
                                scales_per_octave=1,
                                strides=[8, 16, 32, 64, 128]),
                            bbox_coder=dict(
                                type='DeltaXYWHBBoxCoder',
                                target_means=[0.0, 0.0, 0.0, 0.0],
                                target_stds=[0.1, 0.1, 0.2, 0.2]),
                            initial_loss_cls=dict(
                                type='FocalLoss',
                                use_sigmoid=True,
                                activated=True,
                                gamma=2.0,
                                alpha=0.25,
                                loss_weight=1.0),
                            loss_cls=dict(
                                type='QualityFocalLoss',
                                use_sigmoid=True,
                                activated=True,
                                beta=2.0,
                                loss_weight=1.0),
                            loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
                        train_cfg=dict(
                            initial_epoch=4,
                            initial_assigner=dict(type='ATSSAssigner', topk=9),
                            assigner=dict(type='TaskAlignedAssigner', topk=13),
                            alpha=1,
                            beta=6,
                            allowed_border=-1,
                            pos_weight=-1,
                            debug=False),
                        test_cfg=dict(
                            nms_pre=1000,
                            min_bbox_size=0,
                            score_thr=0.05,
                            nms=dict(type='nms', iou_threshold=0.6),
                            max_per_img=100),
                        backbone=dict(
                            type='ConvNeXt',
                            in_chans=3,
                            depths=[3, 3, 27, 3],
                            dims=[128, 256, 512, 1024],
                            drop_path_rate=0.2,
                            layer_scale_init_value=1e-06,
                            out_indices=[0, 1, 2, 3],
                            init_cfg=dict(
                                type='Pretrained',
                                checkpoint=
                                'https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_384.pth'
                            ))),
                    dataset_type='CocoDataset',
                    data_root='/opt/ml/detection/dataset/',
                    img_norm_cfg=dict(
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    albu_train_transforms=[
                        dict(type='RandomRotate90', p=0.4),
                        dict(
                            type='HueSaturationValue',
                            hue_shift_limit=23,
                            sat_shift_limit=30,
                            val_shift_limit=25,
                            p=0.4),
                        dict(type='RandomBrightnessContrast', p=0.4),
                        dict(
                            type='RandomGamma',
                            gamma_limit=(80.0, 148.0),
                            p=0.4),
                        dict(
                            type='OneOf',
                            transforms=[
                                dict(type='GaussianBlur', p=1.0),
                                dict(type='MedianBlur', blur_limit=3, p=1.0)
                            ],
                            p=0.2)
                    ],
                    train_pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(
                            type='Resize',
                            img_scale=(1024, 1024),
                            keep_ratio=True),
                        dict(type='RandomFlip', flip_ratio=0.5),
                        dict(
                            type='Albu',
                            transforms=[
                                dict(type='RandomRotate90', p=0.4),
                                dict(
                                    type='HueSaturationValue',
                                    hue_shift_limit=23,
                                    sat_shift_limit=30,
                                    val_shift_limit=25,
                                    p=0.4),
                                dict(type='RandomBrightnessContrast', p=0.4),
                                dict(
                                    type='RandomGamma',
                                    gamma_limit=(80.0, 148.0),
                                    p=0.4),
                                dict(
                                    type='OneOf',
                                    transforms=[
                                        dict(type='GaussianBlur', p=1.0),
                                        dict(
                                            type='MedianBlur',
                                            blur_limit=3,
                                            p=1.0)
                                    ],
                                    p=0.2)
                            ],
                            bbox_params=dict(
                                type='BboxParams',
                                format='pascal_voc',
                                label_fields=['gt_labels'],
                                min_visibility=0.0,
                                filter_lost_elements=True),
                            keymap=dict(img='image', gt_bboxes='bboxes'),
                            update_pad_shape=False,
                            skip_img_without_anno=True),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='Pad', size_divisor=32),
                        dict(type='DefaultFormatBundle'),
                        dict(
                            type='Collect',
                            keys=['img', 'gt_bboxes', 'gt_labels'])
                    ],
                    test_pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(
                            type='MultiScaleFlipAug',
                            img_scale=(1024, 1024),
                            flip=False,
                            transforms=[
                                dict(type='Resize', keep_ratio=True),
                                dict(type='RandomFlip'),
                                dict(
                                    type='Normalize',
                                    mean=[123.675, 116.28, 103.53],
                                    std=[58.395, 57.12, 57.375],
                                    to_rgb=True),
                                dict(type='Pad', size_divisor=32),
                                dict(type='ImageToTensor', keys=['img']),
                                dict(type='Collect', keys=['img'])
                            ])
                    ],
                    data=dict(
                        samples_per_gpu=2,
                        workers_per_gpu=1,
                        train=dict(
                            type='CocoDataset',
                            classes=('General trash', 'Paper', 'Paper pack',
                                     'Metal', 'Glass', 'Plastic', 'Styrofoam',
                                     'Plastic bag', 'Battery', 'Clothing'),
                            ann_file=
                            '/opt/ml/detection/dataset/fold_2_train.json',
                            img_prefix='/opt/ml/detection/dataset/',
                            pipeline=[
                                dict(type='LoadImageFromFile'),
                                dict(type='LoadAnnotations', with_bbox=True),
                                dict(
                                    type='Resize',
                                    img_scale=(1024, 1024),
                                    keep_ratio=True),
                                dict(type='RandomFlip', flip_ratio=0.5),
                                dict(
                                    type='Albu',
                                    transforms=[
                                        dict(type='RandomRotate90', p=0.4),
                                        dict(
                                            type='HueSaturationValue',
                                            hue_shift_limit=23,
                                            sat_shift_limit=30,
                                            val_shift_limit=25,
                                            p=0.4),
                                        dict(
                                            type='RandomBrightnessContrast',
                                            p=0.4),
                                        dict(
                                            type='RandomGamma',
                                            gamma_limit=(80.0, 148.0),
                                            p=0.4),
                                        dict(
                                            type='OneOf',
                                            transforms=[
                                                dict(
                                                    type='GaussianBlur',
                                                    p=1.0),
                                                dict(
                                                    type='MedianBlur',
                                                    blur_limit=3,
                                                    p=1.0)
                                            ],
                                            p=0.2)
                                    ],
                                    bbox_params=dict(
                                        type='BboxParams',
                                        format='pascal_voc',
                                        label_fields=['gt_labels'],
                                        min_visibility=0.0,
                                        filter_lost_elements=True),
                                    keymap=dict(
                                        img='image', gt_bboxes='bboxes'),
                                    update_pad_shape=False,
                                    skip_img_without_anno=True),
                                dict(
                                    type='Normalize',
                                    mean=[123.675, 116.28, 103.53],
                                    std=[58.395, 57.12, 57.375],
                                    to_rgb=True),
                                dict(type='Pad', size_divisor=32),
                                dict(type='DefaultFormatBundle'),
                                dict(
                                    type='Collect',
                                    keys=['img', 'gt_bboxes', 'gt_labels'])
                            ]),
                        val=dict(
                            type='CocoDataset',
                            classes=('General trash', 'Paper', 'Paper pack',
                                     'Metal', 'Glass', 'Plastic', 'Styrofoam',
                                     'Plastic bag', 'Battery', 'Clothing'),
                            ann_file=
                            '/opt/ml/detection/dataset/fold_2_val.json',
                            img_prefix='/opt/ml/detection/dataset/',
                            pipeline=[
                                dict(type='LoadImageFromFile'),
                                dict(
                                    type='MultiScaleFlipAug',
                                    img_scale=(1024, 1024),
                                    flip=False,
                                    transforms=[
                                        dict(type='Resize', keep_ratio=True),
                                        dict(type='RandomFlip'),
                                        dict(
                                            type='Normalize',
                                            mean=[123.675, 116.28, 103.53],
                                            std=[58.395, 57.12, 57.375],
                                            to_rgb=True),
                                        dict(type='Pad', size_divisor=32),
                                        dict(
                                            type='ImageToTensor',
                                            keys=['img']),
                                        dict(type='Collect', keys=['img'])
                                    ])
                            ]),
                        test=dict(
                            type='CocoDataset',
                            classes=('General trash', 'Paper', 'Paper pack',
                                     'Metal', 'Glass', 'Plastic', 'Styrofoam',
                                     'Plastic bag', 'Battery', 'Clothing'),
                            ann_file='/opt/ml/detection/dataset/test.json',
                            img_prefix='/opt/ml/detection/dataset/',
                            pipeline=[
                                dict(type='LoadImageFromFile'),
                                dict(
                                    type='MultiScaleFlipAug',
                                    img_scale=(1024, 1024),
                                    flip=False,
                                    transforms=[
                                        dict(type='Resize', keep_ratio=True),
                                        dict(type='RandomFlip'),
                                        dict(
                                            type='Normalize',
                                            mean=[123.675, 116.28, 103.53],
                                            std=[58.395, 57.12, 57.375],
                                            to_rgb=True),
                                        dict(type='Pad', size_divisor=32),
                                        dict(
                                            type='ImageToTensor',
                                            keys=['img']),
                                        dict(type='Collect', keys=['img'])
                                    ])
                            ])),
                    evaluation=dict(
                        interval=1,
                        metric='bbox',
                        save_best='bbox_mAP_50',
                        classwise=True),
                    optimizer=dict(
                        type='AdamW', lr=0.0001, weight_decay=1e-05),
                    optimizer_config=dict(grad_clip=None),
                    lr_config=dict(
                        policy='CosineAnnealing',
                        warmup='linear',
                        warmup_iters=1000,
                        warmup_ratio=0.01,
                        min_lr=1e-06),
                    runner=dict(type='EpochBasedRunner', max_epochs=24),
                    checkpoint_config=dict(max_keep_ckpts=3, interval=1),
                    custom_hooks=[dict(type='NumClassCheckHook')],
                    dist_params=dict(backend='nccl'),
                    log_level='INFO',
                    load_from=None,
                    resume_from=None,
                    workflow=[('train', 1)],
                    opencv_num_threads=0,
                    mp_start_method='fork',
                    pretrained=
                    'https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_384.pth'
                )))
    ])
work_dir = './work_dirs/tood_convnext_fold2'
auto_resume = False
gpu_ids = [0]
