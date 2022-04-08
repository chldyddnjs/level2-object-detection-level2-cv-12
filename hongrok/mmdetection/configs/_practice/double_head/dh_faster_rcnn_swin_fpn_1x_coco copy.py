model = dict(
    type='FasterRCNN',
    backbone=dict(
        type='SwinTransformer',
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'
        )),
    neck=dict(
        type='FPN',
        in_channels=[192, 384, 768, 1536],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8, 16],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
    roi_head=dict(
        type='DoubleHeadRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        reg_roi_scale_factor=1.3,
        bbox_head=dict(
            type='DoubleConvFCBBoxHead',
            num_convs=4,
            num_fcs=2,
            in_channels=256,
            conv_out_channels=1024,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=10,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=2.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=2.0))),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)))
dataset_type = 'CocoDataset'
data_root = '../../dataset/'
classes = ('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic',
           'Styrofoam', 'Plastic bag', 'Battery', 'Clothing')
img_norm_cfg = dict(
    mean=[123.6506697, 117.39730243, 110.07542563],
    std=[54.03457934, 53.36968771, 54.78390763],
    to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Resize',
        img_scale=[(1024, 1024)],
        ratio_range=(0.5, 1.5),
        keep_ratio=True),
    dict(
        type='CutOut',
        n_holes=(5, 10),
        cutout_shape=[(4, 4), (4, 8), (8, 4), (8, 8),
                      (16, 8), (8, 16), (16, 16), (16, 32), (32, 16), (32, 32),
                      (32, 48), (48, 32), (48, 48)]),
    dict(
        type='Albu',
        transforms=[
            dict(type='RandomRotate90', p=0.5),
            dict(
                type='OneOf',
                transforms=[
                    dict(
                        type='RandomFog', alpha_coef=0.25, fog_coef_upper=0.5),
                    dict(type='Blur', blur_limit=(3, 7), p=1)
                ],
                p=0.5),
            dict(
                type='OneOf',
                transforms=[
                    dict(
                        type='HueSaturationValue',
                        hue_shift_limit=(-20, 20),
                        sat_shift_limit=(-30, 30),
                        val_shift_limit=(-20, 20),
                        p=1),
                    dict(type='RandomBrightnessContrast', p=1)
                ],
                p=0.5)
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
        mean=[123.6506697, 117.39730243, 110.07542563],
        std=[54.03457934, 53.36968771, 54.78390763],
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
                mean=[123.6506697, 117.39730243, 110.07542563],
                std=[54.03457934, 53.36968771, 54.78390763],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type='CocoDataset',
        classes=('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
                 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing'),
        img_prefix='../../dataset/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Resize',
                img_scale=[(1024, 1024)],
                ratio_range=(0.5, 1.5),
                keep_ratio=True),
            dict(
                type='CutOut',
                n_holes=(5, 10),
                cutout_shape=[(4, 4), (4, 8), (8, 4), (8, 8), (16, 8), (8, 16),
                              (16, 16), (16, 32), (32, 16), (32, 32), (32, 48),
                              (48, 32), (48, 48)]),
            dict(
                type='Albu',
                transforms=[
                    dict(type='RandomRotate90', p=0.5),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(
                                type='RandomFog',
                                alpha_coef=0.25,
                                fog_coef_upper=0.5),
                            dict(type='Blur', blur_limit=(3, 7), p=1)
                        ],
                        p=0.5),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(
                                type='HueSaturationValue',
                                hue_shift_limit=(-20, 20),
                                sat_shift_limit=(-30, 30),
                                val_shift_limit=(-20, 20),
                                p=1),
                            dict(type='RandomBrightnessContrast', p=1)
                        ],
                        p=0.5)
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
                mean=[123.6506697, 117.39730243, 110.07542563],
                std=[54.03457934, 53.36968771, 54.78390763],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ],
        ann_file='../../dataset/clear_fold_0_train.json'),
    val=dict(
        type='CocoDataset',
        classes=('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
                 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing'),
        img_prefix='../../dataset/',
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
                        mean=[123.6506697, 117.39730243, 110.07542563],
                        std=[54.03457934, 53.36968771, 54.78390763],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        ann_file='../../dataset/clear_fold_0_val.json'),
    test=dict(
        type='CocoDataset',
        classes=('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
                 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing'),
        ann_file='../../dataset/test.json',
        img_prefix='../../dataset/',
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
                        mean=[123.6506697, 117.39730243, 110.07542563],
                        std=[54.03457934, 53.36968771, 54.78390763],
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
runner = dict(type='EpochBasedRunner', max_epochs=50)
checkpoint_config = dict(interval=1, save_last=True, max_keep_ckpts=2)
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='detection',
                entity='yolo12',
                name='hongrok_fold0_dh_faster_rcnn_swin_fpn_1x_coco',
                config=dict(
                    model=dict(
                        type='FasterRCNN',
                        backbone=dict(
                            type='SwinTransformer',
                            embed_dims=192,
                            depths=[2, 2, 18, 2],
                            num_heads=[6, 12, 24, 48],
                            window_size=12,
                            mlp_ratio=4,
                            qkv_bias=True,
                            qk_scale=None,
                            drop_rate=0.0,
                            attn_drop_rate=0.0,
                            drop_path_rate=0.2,
                            patch_norm=True,
                            out_indices=(0, 1, 2, 3),
                            with_cp=False,
                            convert_weights=True,
                            init_cfg=dict(
                                type='Pretrained',
                                checkpoint=
                                'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'
                            )),
                        neck=dict(
                            type='FPN',
                            in_channels=[192, 384, 768, 1536],
                            out_channels=256,
                            num_outs=5),
                        rpn_head=dict(
                            type='RPNHead',
                            in_channels=256,
                            feat_channels=256,
                            anchor_generator=dict(
                                type='AnchorGenerator',
                                scales=[8, 16],
                                ratios=[0.5, 1.0, 2.0],
                                strides=[4, 8, 16, 32, 64]),
                            bbox_coder=dict(
                                type='DeltaXYWHBBoxCoder',
                                target_means=[0.0, 0.0, 0.0, 0.0],
                                target_stds=[1.0, 1.0, 1.0, 1.0]),
                            loss_cls=dict(
                                type='CrossEntropyLoss',
                                use_sigmoid=True,
                                loss_weight=1.0),
                            loss_bbox=dict(
                                type='SmoothL1Loss', beta=1.0,
                                loss_weight=1.0)),
                        roi_head=dict(
                            type='DoubleHeadRoIHead',
                            bbox_roi_extractor=dict(
                                type='SingleRoIExtractor',
                                roi_layer=dict(
                                    type='RoIAlign',
                                    output_size=7,
                                    sampling_ratio=0),
                                out_channels=256,
                                featmap_strides=[4, 8, 16, 32]),
                            reg_roi_scale_factor=1.3,
                            bbox_head=dict(
                                type='DoubleConvFCBBoxHead',
                                num_convs=4,
                                num_fcs=2,
                                in_channels=256,
                                conv_out_channels=1024,
                                fc_out_channels=1024,
                                roi_feat_size=7,
                                num_classes=10,
                                bbox_coder=dict(
                                    type='DeltaXYWHBBoxCoder',
                                    target_means=[0.0, 0.0, 0.0, 0.0],
                                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                                reg_class_agnostic=False,
                                loss_cls=dict(
                                    type='CrossEntropyLoss',
                                    use_sigmoid=False,
                                    loss_weight=2.0),
                                loss_bbox=dict(
                                    type='SmoothL1Loss',
                                    beta=1.0,
                                    loss_weight=2.0))),
                        train_cfg=dict(
                            rpn=dict(
                                assigner=dict(
                                    type='MaxIoUAssigner',
                                    pos_iou_thr=0.7,
                                    neg_iou_thr=0.3,
                                    min_pos_iou=0.3,
                                    match_low_quality=True,
                                    ignore_iof_thr=-1),
                                sampler=dict(
                                    type='RandomSampler',
                                    num=256,
                                    pos_fraction=0.5,
                                    neg_pos_ub=-1,
                                    add_gt_as_proposals=False),
                                allowed_border=-1,
                                pos_weight=-1,
                                debug=False),
                            rpn_proposal=dict(
                                nms_pre=2000,
                                max_per_img=1000,
                                nms=dict(type='nms', iou_threshold=0.7),
                                min_bbox_size=0),
                            rcnn=dict(
                                assigner=dict(
                                    type='MaxIoUAssigner',
                                    pos_iou_thr=0.5,
                                    neg_iou_thr=0.5,
                                    min_pos_iou=0.5,
                                    match_low_quality=False,
                                    ignore_iof_thr=-1),
                                sampler=dict(
                                    type='RandomSampler',
                                    num=512,
                                    pos_fraction=0.25,
                                    neg_pos_ub=-1,
                                    add_gt_as_proposals=True),
                                pos_weight=-1,
                                debug=False)),
                        test_cfg=dict(
                            rpn=dict(
                                nms_pre=1000,
                                max_per_img=1000,
                                nms=dict(type='nms', iou_threshold=0.7),
                                min_bbox_size=0),
                            rcnn=dict(
                                score_thr=0.05,
                                nms=dict(type='nms', iou_threshold=0.5),
                                max_per_img=100))),
                    dataset_type='CocoDataset',
                    data_root='../../dataset/',
                    classes=('General trash', 'Paper', 'Paper pack', 'Metal',
                             'Glass', 'Plastic', 'Styrofoam', 'Plastic bag',
                             'Battery', 'Clothing'),
                    img_norm_cfg=dict(
                        mean=[123.6506697, 117.39730243, 110.07542563],
                        std=[54.03457934, 53.36968771, 54.78390763],
                        to_rgb=True),
                    train_pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='RandomFlip', flip_ratio=0.5),
                        dict(
                            type='Resize',
                            img_scale=[(1024, 1024)],
                            ratio_range=(0.5, 1.5),
                            keep_ratio=True),
                        dict(
                            type='CutOut',
                            n_holes=(5, 10),
                            cutout_shape=[(4, 4), (4, 8), (8, 4), (8, 8),
                                          (16, 8), (8, 16), (16, 16), (16, 32),
                                          (32, 16), (32, 32), (32, 48),
                                          (48, 32), (48, 48)]),
                        dict(
                            type='Albu',
                            transforms=[
                                dict(type='RandomRotate90', p=0.5),
                                dict(
                                    type='OneOf',
                                    transforms=[
                                        dict(
                                            type='RandomFog',
                                            alpha_coef=0.25,
                                            fog_coef_upper=0.5),
                                        dict(
                                            type='Blur',
                                            blur_limit=(3, 7),
                                            p=1)
                                    ],
                                    p=0.5),
                                dict(
                                    type='OneOf',
                                    transforms=[
                                        dict(
                                            type='HueSaturationValue',
                                            hue_shift_limit=(-20, 20),
                                            sat_shift_limit=(-30, 30),
                                            val_shift_limit=(-20, 20),
                                            p=1),
                                        dict(
                                            type='RandomBrightnessContrast',
                                            p=1)
                                    ],
                                    p=0.5)
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
                            mean=[123.6506697, 117.39730243, 110.07542563],
                            std=[54.03457934, 53.36968771, 54.78390763],
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
                                    mean=[
                                        123.6506697, 117.39730243, 110.07542563
                                    ],
                                    std=[
                                        54.03457934, 53.36968771, 54.78390763
                                    ],
                                    to_rgb=True),
                                dict(type='Pad', size_divisor=32),
                                dict(type='ImageToTensor', keys=['img']),
                                dict(type='Collect', keys=['img'])
                            ])
                    ],
                    data=dict(
                        samples_per_gpu=1,
                        workers_per_gpu=2,
                        train=dict(
                            type='CocoDataset',
                            classes=('General trash', 'Paper', 'Paper pack',
                                     'Metal', 'Glass', 'Plastic', 'Styrofoam',
                                     'Plastic bag', 'Battery', 'Clothing'),
                            img_prefix='../../dataset/',
                            pipeline=[
                                dict(type='LoadImageFromFile'),
                                dict(type='LoadAnnotations', with_bbox=True),
                                dict(type='RandomFlip', flip_ratio=0.5),
                                dict(
                                    type='Resize',
                                    img_scale=[(1024, 1024)],
                                    ratio_range=(0.5, 1.5),
                                    keep_ratio=True),
                                dict(
                                    type='CutOut',
                                    n_holes=(5, 10),
                                    cutout_shape=[(4, 4), (4, 8), (8, 4),
                                                  (8, 8), (16, 8), (8, 16),
                                                  (16, 16), (16, 32), (32, 16),
                                                  (32, 32), (32, 48), (48, 32),
                                                  (48, 48)]),
                                dict(
                                    type='Albu',
                                    transforms=[
                                        dict(type='RandomRotate90', p=0.5),
                                        dict(
                                            type='OneOf',
                                            transforms=[
                                                dict(
                                                    type='RandomFog',
                                                    alpha_coef=0.25,
                                                    fog_coef_upper=0.5),
                                                dict(
                                                    type='Blur',
                                                    blur_limit=(3, 7),
                                                    p=1)
                                            ],
                                            p=0.5),
                                        dict(
                                            type='OneOf',
                                            transforms=[
                                                dict(
                                                    type='HueSaturationValue',
                                                    hue_shift_limit=(-20, 20),
                                                    sat_shift_limit=(-30, 30),
                                                    val_shift_limit=(-20, 20),
                                                    p=1),
                                                dict(
                                                    type=
                                                    'RandomBrightnessContrast',
                                                    p=1)
                                            ],
                                            p=0.5)
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
                                    mean=[
                                        123.6506697, 117.39730243, 110.07542563
                                    ],
                                    std=[
                                        54.03457934, 53.36968771, 54.78390763
                                    ],
                                    to_rgb=True),
                                dict(type='Pad', size_divisor=32),
                                dict(type='DefaultFormatBundle'),
                                dict(
                                    type='Collect',
                                    keys=['img', 'gt_bboxes', 'gt_labels'])
                            ],
                            ann_file='../../dataset/clear_fold_0_train.json'),
                        val=dict(
                            type='CocoDataset',
                            classes=('General trash', 'Paper', 'Paper pack',
                                     'Metal', 'Glass', 'Plastic', 'Styrofoam',
                                     'Plastic bag', 'Battery', 'Clothing'),
                            img_prefix='../../dataset/',
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
                                            mean=[
                                                123.6506697, 117.39730243,
                                                110.07542563
                                            ],
                                            std=[
                                                54.03457934, 53.36968771,
                                                54.78390763
                                            ],
                                            to_rgb=True),
                                        dict(type='Pad', size_divisor=32),
                                        dict(
                                            type='ImageToTensor',
                                            keys=['img']),
                                        dict(type='Collect', keys=['img'])
                                    ])
                            ],
                            ann_file='../../dataset/clear_fold_0_val.json'),
                        test=dict(
                            type='CocoDataset',
                            classes=('General trash', 'Paper', 'Paper pack',
                                     'Metal', 'Glass', 'Plastic', 'Styrofoam',
                                     'Plastic bag', 'Battery', 'Clothing'),
                            ann_file='../../dataset/test.json',
                            img_prefix='../../dataset/',
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
                                            mean=[
                                                123.6506697, 117.39730243,
                                                110.07542563
                                            ],
                                            std=[
                                                54.03457934, 53.36968771,
                                                54.78390763
                                            ],
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
                    runner=dict(type='EpochBasedRunner', max_epochs=50),
                    checkpoint_config=dict(
                        interval=1, save_last=True, max_keep_ckpts=2),
                    custom_hooks=[dict(type='NumClassCheckHook')],
                    dist_params=dict(backend='nccl'),
                    log_level='INFO',
                    load_from=None,
                    resume_from=None,
                    workflow=[('train', 1)],
                    opencv_num_threads=0,
                    mp_start_method='fork')))
    ])
work_dir = 'work_dirs/dh_faster_rcnn_swin_fpn_1x_coco_rotate90/'
auto_resume = False
gpu_ids = [0]
