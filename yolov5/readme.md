# yolov5
------------
| Validation set mAP 50 | LB Score |
| --------------------- | -------- |
|      0.5951           |  0.5248  |

------------

## train.py
------------
'''
python train --weights {pretraiend된 wieght 파일} --data trash.yaml --hyp {argument file(hyp.p6.yaml)} --epochs 80 --batch_size 12 --img_size 1024 --project {wandb project 명}
'''