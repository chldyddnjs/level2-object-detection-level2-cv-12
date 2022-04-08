#!/bin/bash

python train.py configs/_practice/double_head/dh_faster_rcnn_swin_fpn_1x_coco.py --work-dir work_dirs/dh_fold1 --user_name hongrok --fold_num 1 --wandb_exp dh_fold1 --epochs 18