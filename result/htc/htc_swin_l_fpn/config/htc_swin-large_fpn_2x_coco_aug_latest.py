model = dict(
    type='HybridTaskCascade',
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
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)),
    roi_head=dict(
        type='HybridTaskCascadeRoIHead',
        interleaved=True,
        mask_info_flow=True,
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=10,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=10,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=10,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ]),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.001,
            nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05),
            max_per_img=100)))
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
        ann_file='/opt/ml/detection/dataset/fold_3_train.json',
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
        ann_file='/opt/ml/detection/dataset/fold_3_val.json',
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
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
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
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='detection',
                entity='yolo12',
                name='Rch_fold_3_htc_swin_augment',
                config=dict(
                    model=dict(
                        type='HybridTaskCascade',
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
                                scales=[8],
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
                                type='SmoothL1Loss',
                                beta=0.1111111111111111,
                                loss_weight=1.0)),
                        roi_head=dict(
                            type='HybridTaskCascadeRoIHead',
                            interleaved=True,
                            mask_info_flow=True,
                            num_stages=3,
                            stage_loss_weights=[1, 0.5, 0.25],
                            bbox_roi_extractor=dict(
                                type='SingleRoIExtractor',
                                roi_layer=dict(
                                    type='RoIAlign',
                                    output_size=7,
                                    sampling_ratio=0),
                                out_channels=256,
                                featmap_strides=[4, 8, 16, 32]),
                            bbox_head=[
                                dict(
                                    type='Shared2FCBBoxHead',
                                    in_channels=256,
                                    fc_out_channels=1024,
                                    roi_feat_size=7,
                                    num_classes=10,
                                    bbox_coder=dict(
                                        type='DeltaXYWHBBoxCoder',
                                        target_means=[0.0, 0.0, 0.0, 0.0],
                                        target_stds=[0.1, 0.1, 0.2, 0.2]),
                                    reg_class_agnostic=True,
                                    loss_cls=dict(
                                        type='CrossEntropyLoss',
                                        use_sigmoid=False,
                                        loss_weight=1.0),
                                    loss_bbox=dict(
                                        type='SmoothL1Loss',
                                        beta=1.0,
                                        loss_weight=1.0)),
                                dict(
                                    type='Shared2FCBBoxHead',
                                    in_channels=256,
                                    fc_out_channels=1024,
                                    roi_feat_size=7,
                                    num_classes=10,
                                    bbox_coder=dict(
                                        type='DeltaXYWHBBoxCoder',
                                        target_means=[0.0, 0.0, 0.0, 0.0],
                                        target_stds=[0.05, 0.05, 0.1, 0.1]),
                                    reg_class_agnostic=True,
                                    loss_cls=dict(
                                        type='CrossEntropyLoss',
                                        use_sigmoid=False,
                                        loss_weight=1.0),
                                    loss_bbox=dict(
                                        type='SmoothL1Loss',
                                        beta=1.0,
                                        loss_weight=1.0)),
                                dict(
                                    type='Shared2FCBBoxHead',
                                    in_channels=256,
                                    fc_out_channels=1024,
                                    roi_feat_size=7,
                                    num_classes=10,
                                    bbox_coder=dict(
                                        type='DeltaXYWHBBoxCoder',
                                        target_means=[0.0, 0.0, 0.0, 0.0],
                                        target_stds=[
                                            0.033, 0.033, 0.067, 0.067
                                        ]),
                                    reg_class_agnostic=True,
                                    loss_cls=dict(
                                        type='CrossEntropyLoss',
                                        use_sigmoid=False,
                                        loss_weight=1.0),
                                    loss_bbox=dict(
                                        type='SmoothL1Loss',
                                        beta=1.0,
                                        loss_weight=1.0))
                            ]),
                        train_cfg=dict(
                            rpn=dict(
                                assigner=dict(
                                    type='MaxIoUAssigner',
                                    pos_iou_thr=0.7,
                                    neg_iou_thr=0.3,
                                    min_pos_iou=0.3,
                                    ignore_iof_thr=-1),
                                sampler=dict(
                                    type='RandomSampler',
                                    num=256,
                                    pos_fraction=0.5,
                                    neg_pos_ub=-1,
                                    add_gt_as_proposals=False),
                                allowed_border=0,
                                pos_weight=-1,
                                debug=False),
                            rpn_proposal=dict(
                                nms_pre=2000,
                                max_per_img=2000,
                                nms=dict(type='nms', iou_threshold=0.7),
                                min_bbox_size=0),
                            rcnn=[
                                dict(
                                    assigner=dict(
                                        type='MaxIoUAssigner',
                                        pos_iou_thr=0.5,
                                        neg_iou_thr=0.5,
                                        min_pos_iou=0.5,
                                        ignore_iof_thr=-1),
                                    sampler=dict(
                                        type='RandomSampler',
                                        num=512,
                                        pos_fraction=0.25,
                                        neg_pos_ub=-1,
                                        add_gt_as_proposals=True),
                                    mask_size=28,
                                    pos_weight=-1,
                                    debug=False),
                                dict(
                                    assigner=dict(
                                        type='MaxIoUAssigner',
                                        pos_iou_thr=0.6,
                                        neg_iou_thr=0.6,
                                        min_pos_iou=0.6,
                                        ignore_iof_thr=-1),
                                    sampler=dict(
                                        type='RandomSampler',
                                        num=512,
                                        pos_fraction=0.25,
                                        neg_pos_ub=-1,
                                        add_gt_as_proposals=True),
                                    mask_size=28,
                                    pos_weight=-1,
                                    debug=False),
                                dict(
                                    assigner=dict(
                                        type='MaxIoUAssigner',
                                        pos_iou_thr=0.7,
                                        neg_iou_thr=0.7,
                                        min_pos_iou=0.7,
                                        ignore_iof_thr=-1),
                                    sampler=dict(
                                        type='RandomSampler',
                                        num=512,
                                        pos_fraction=0.25,
                                        neg_pos_ub=-1,
                                        add_gt_as_proposals=True),
                                    pos_weight=-1,
                                    debug=False)
                            ]),
                        test_cfg=dict(
                            rpn=dict(
                                nms_pre=1000,
                                max_per_img=1000,
                                nms=dict(type='nms', iou_threshold=0.7),
                                min_bbox_size=0),
                            rcnn=dict(
                                score_thr=0.001,
                                nms=dict(
                                    type='soft_nms',
                                    iou_threshold=0.5,
                                    min_score=0.05),
                                max_per_img=100))),
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
                            '/opt/ml/detection/dataset/fold_3_train.json',
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
                            '/opt/ml/detection/dataset/fold_3_val.json',
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
                        type='AdamW',
                        lr=0.0001,
                        betas=(0.9, 0.999),
                        weight_decay=0.05,
                        paramwise_cfg=dict(
                            custom_keys=dict(
                                absolute_pos_embed=dict(decay_mult=0.0),
                                relative_position_bias_table=dict(
                                    decay_mult=0.0),
                                norm=dict(decay_mult=0.0)))),
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
                    'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'
                )))
    ])
work_dir = './work_dirs/htc_swin-large_fpn_2x_coco_aug_latest'
auto_resume = False
gpu_ids = [0]
