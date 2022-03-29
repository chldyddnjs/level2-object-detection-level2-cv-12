checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook',interval=50,
            init_kwargs=dict(
                project= 'Detection',
                entity = 'yolo12',
                name = 'Rch_fold3_cascade_swin_large_fpn_1x_coco_detection'
            )
         #dict(type='TensorboardLoggerHook')
        )
])

# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
