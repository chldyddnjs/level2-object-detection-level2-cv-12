# optimizer
# optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.00001,)
optimizer_config=dict(grad_clip=None)
# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# optimizer_config['grad_clip'] = dict(max_norm=50, norm_type=2)
# learning policy
# lr_config = dict(
#     policy='CosineAnnealing',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001, # 0.01 -> 0.001
#     min_lr_ratio=1e-7,
#     # step=[8, 11]
#     )
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[8, 11])

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.1 / 10,
    min_lr=1e-6)

runner = dict(type='EpochBasedRunner')


##
# GradClip을 제거
# CosineAnnealing추가