# yolor
------------
| Validation set mAP 50 | LB Score |
| --------------------- | -------- |
|      0.5785           |  0.5379  |

------------

## train.py
```
python train.py --batch-size 8 --img 1024 --data coco.yaml --cfg cfg/yolor_w6.cfg --weights ./yolor_w6.pt --device 0 --name yolor_fold4 --hyp hyp.finetune.1280.yaml --epochs 300
```

## detect.py
```
python test.py --weights {best weight 파일} --cfg {config file} --task test --img-size 1024 --data coco.yaml --conf-thres 0.08 --iou-thres 0.5 --name {폴더명} --save-txt --save-conf --verbose
```
```bash
├── label
│   ├── 0001.txt
│   ├── 0002.txt
│   ├── 0003.txt
│   └── ...

``` 



## inference.ipynb
inference.ipynb 파일로 yolo dataset >> coco dataset 형식으로 변환 후 csv파일 저장