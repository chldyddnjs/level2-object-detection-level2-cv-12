#!/bin/bash

# config_file='configs/_practice/faster_rcnn_r50_fpn_1x_coco.py'
# model_pth = /opt/ml/detection/hongrok/mmdetection/work_dirs/faster_rcnn_r50_fpn_1x_coco/latest.pth
# out_name = jsonfile_prefix=./mask_rcnn_test-dev_results
# threshold = 0.05
# work_dir = "work_dirs_norm20"
# python tools/train.py configs/_practice/faster_rcnn_r50_fpn_1x_coco.py --work-dir work_dirs_norm20
# python tools/test.py $config_file $model_pth --format-only --options $out_name --show-score-thr $threshold

# args
# data_root='../../dataset/' #실행하는 위치기준
config_file='configs/_practice/cascade_rcnn_swin-t-p4-w7_fpn_1x_coco.py'
work_dir='work_dirs/'
user_name='hongrok'
fold_num=0
# train
python tools/train.py $config_file\
 --work-dir $work_dir\
 --wandb_exp test_test_test_test_test_test\
 --fold_num $fold_num\
 --user_name $user_name

# inference
# python tools/test.py $config_file $model_pth --format-only --options $out_name --show-score-thr $threshold
