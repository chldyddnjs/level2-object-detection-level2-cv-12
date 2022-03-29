# optimizer
optimizer = dict(type='SGD', lr=0.00002, momentum=0.9, weight_decay=0.00000001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=30)
