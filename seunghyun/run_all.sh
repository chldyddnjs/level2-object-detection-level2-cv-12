#!/bin/bash

python tools/train.py /opt/ml/Pstage/level2-object-detection-level2-cv-12/baseline/mmdetection/configs/_boost_/cascade_rcnn/cascade_rcnn_swin_t.py --foldnum 2 --wandbname swin_t --epoch 24