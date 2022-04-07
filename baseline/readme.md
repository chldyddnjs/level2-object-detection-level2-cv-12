# Cascade Rcnn (Swin transformer , Convnext)
------------
|   BackBone Model | Validation set mAP 50 | LB Score |
| :--------------: | :-------------------: | :------: |
|    ConvNeXT      |      0.6403           |  0.6231  |
| Swin Transformer |     0.6367            | 0.6070   |
------------

## train.py (convnext)
```
python tools/train.py cascade_rcnn_convnext2.py --foldnum {kfold 번호} --wandb {wandb name} --epoch {epoch 수}
```
### 관련 파일
```bash
├── configs
│   ├── _boost_
│       ├── _base_
|           ├── datasets
|               └── coco_detection_aug.py
│       ├── cascade_rcnn
│           ├── cascade_rcnn_convnext2.py
├── mmdet
│   ├── models
│       ├── backbones
│           ├── __init__.py
│           └── convnext.py

``` 

## train.py (SwinTransformer)
```
python tools/train.py cascade_rcnn_swin_t.py --foldnum {kfold 번호} --wandb {wandb name} --epoch {epoch 수}
```
### 관련 파일
```bash
├── configs
│   ├── _boost_
│       ├── _base_
|           ├── datasets
|               └── coco_detection_aug.py
│       ├── cascade_rcnn
│           ├── cascade_rcnn_swin_t.py
├── mmdet
│   ├── models
│       ├── backbones
│           ├── __init__.py
│           └── convnext.py

``` 
## inference.ipynb
config 와 pth파일 경로 넣어준 후 결과 생성