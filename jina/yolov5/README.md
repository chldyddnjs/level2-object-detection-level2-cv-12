## yolov5
---

#### best score

Model|validation mAP50
---|---|
yolov5|0.5861

#### train

```python
python train.py --weights {pretrained weights path} --cfg {model.yaml path} --data {dataset.yaml path} --hyp {hyperparameters path} --projects {wandb project name} --epochs 100 --img 1024
```

#### detect

```python
python detect.py --weights {best weights path} --source {test dataset path} --data {dataset.yaml path} --img 1024 --save-txt --save-conf
```

#### to_submission

yolov5 결과를 지정된 submission 형식으로 변환 후 csv 저장
```python
python to_submission.py --exp {실험 결과 exp number}
```
