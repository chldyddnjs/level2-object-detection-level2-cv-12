# Cascade RCNN & Hybrid Task Cascade
---

## best score
backbone : swin_large

Model|validation mAP50|LB score
---|---|---|
cascade_rcnn|0.598|0.5549
htc|0.673|0.6278

## train
```python
python tools/train.py {config path} 
```

## inference
config path, work_dir path, checkpoint path, csv file name 지정 후 inference.ipynb 파일 실행
