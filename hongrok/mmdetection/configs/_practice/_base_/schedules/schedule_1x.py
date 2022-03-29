# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict()
# optimizer_config['grad_clip'] = dict(max_norm=50, norm_type=2)
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.1,
    min_lr_ratio=1e-5,
    # step=[8, 11]
    )
runner = dict(type='EpochBasedRunner')


##
# GradClip을 제거
# CosineAnnealing추가