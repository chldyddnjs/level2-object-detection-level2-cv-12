# yolov5
------------
| Validation set mAP 50 | LB Score |
| --------------------- | -------- |
|      0.5951           |  0.5248  |

------------

## train.py
```
python train.py --weights {pretraiend된 wieght 파일} --data trash.yaml --hyp {argument file(hyp.p6.yaml)} --epochs 80 --batch_size 12 --img_size 1024 --project {wandb project 명}
```

## detect.py
```
python detect.py --weights {best weight 파일} --source {test dataset 위치} --imgsz 1024 --data trash.yaml --conf-thres 0.08 --iou-thres 0.5 --name {폴더명} --save-txt --save-conf
```

├── detect test folder                   
    ├── 0000.txt                
    ├── 0001.txt                 
    ├── 0002.txt                
    ├── 0003.txt            
    ├── 0004.txt
    └── ...



## inference.ipynb
inference.ipynb 파일로 yolo dataset >> coco dataset 형식으로 변환 후 csv파일 저장