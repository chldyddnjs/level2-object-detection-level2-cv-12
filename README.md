바야흐로 대량 생산, 대량 소비의 시대. 우리는 많은 물건이 대량으로 생산되고, 소비되는 시대를 살고 있습니다. 하지만 이러한 문화는 '쓰레기 대란', '매립지 부족'과 같은 여러 사회 문제를 낳고 있습니다. 

분리수거는 이러한 환경 부담을 줄일 수 있는 방법 중 하나입니다. 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립 또는 소각되기 때문입니다.

따라서 우리는 사진에서 쓰레기를 Detection 하는 모델을 만들어 이러한 문제점을 해결해보고자 합니다. 문제 해결을 위한 데이터셋으로는 일반 쓰레기, 플라스틱, 종이, 유리 등 10 종류의 쓰레기가 찍힌 사진 데이터셋이 제공됩니다.

여러분에 의해 만들어진 우수한 성능의 모델은 쓰레기장에 설치되어 정확한 분리수거를 돕거나, 어린아이들의 분리수거 교육 등에 사용될 수 있을 것입니다. 부디 지구를 위기로부터 구해주세요! 🌎

!cd /opt/ml/detection/baseline/mmedetction>

conda activate detection
train
python tools/train.py {path/to/file}

test
python tools/test.py {path/to/file}

list up and configs for mmdetection
[basic] faster_rcnn [basic] cascade_rcnn with swin + [optinal]augmentation + cosinAnealing + adamW [basic] htc with swin + [optional]augmentation + cosinAnealing + adamW [basic] tood with convnext + [optional]augmentation + cosinAnealing + adamW [basic] deformable detr [basic] yolox [basic] detr

list up and configs for naive model come from github or kaggle
[basic] efficientdet [basic] DINO

experiment
faster_rcnn -> 0.4010, epoch=12, max_epoch=12
cascade_rcnn + -> 0.5825, epoch =12, max_epoch=12
htc with swin -> 0.5567, epoch=12, max_epoch=12
htc with swin + aug -> 0.5816, epoch=12, max_epoch=24 원래 24epoch까지 올라가야하지만 중간에 실수로 중간에 꺼짐
detr -> 0.0000, epoch=100, max_epoch=150
yolox -> 0.1034, epoch=100, max_epoch=300
efficientdet ->, 0.3443 epoch=50, max_epoch=50
efficientdet + aug -> 0.2431 epoch=50, max_epoch=50
DINO -> (backbone을 vit -> swin으로 바꾸려고 시도한 실험)
tood with convnext + aug -> 0.6171 epoch=12
deformable detr -> cuda version 불일치로 실험 x
ensemble =[
htc_swin_adamW_aug_cosinAnealing(0.6272) + cascade_rcnn_convnext_adamW_cosinAnealing(0.6070) +
cascade_rcnn_convnext_adamW_cosinAnealing_aug(0.6132) +
cascade_rcnn_swin_adamW_aug_stepLR(0.5825) ]
LB -> 0.6641
augmentation
사용한 augmentation은 one of { [brightness,contrest,Hue,saturation], [RGB Shift, Image Roate], [bulr, medium blur] } 를 사용했으며 일반화 성능을 높여줄 기대를 하며 위와 같은 기법을 사용했다. 일반화 성능을 조금더 높이기 위해 autoAug,mosaic,mixup 등의 기법을 사용하고 싶었지만 error를 처리하지 못해 사용하지 못했다.

loss
cascade_rcnn의 smoothL1 loss함수를 Giou loss로 바꾸었을 때 성능이 굉장히 안좋다 기본으로 들어가 있는 loss 함수를 쓰는 것이 오히려 낫다는 말을 듣고 모든 실험에서 일부러 loss를 바꾸지 않았습니다. EfficientDet을 사용하면서 loss가 적은 것과 loss가 상대적으로 큰것을 제출해봤을 때 당연한 얘기지만 성능의 차이가 꽤 있었습니다.

optimizer and scheduler
adamW sgd(momentom) 모델에 따라 optimizer가 기본적으로 다르지만 일반적으로 adamW가 성능이 되게 좋았고 adamW가 loss를 잘 줄이지 못할 때 sgd를 쓰면 loss가 잘 줄어드는 현상도 볼 수 있었다. 뭐든지 만능은 없습니다.

stepLR cosinAnealing stepLR과 cosinAnealing을 두고 실험을 한 결과 모든 실험에 대해서 cosinAnealing이 더 성능이 더 뛰어났다.

느낀점
mmdetection 기준으로 최대한 다양한 모델을 사용해보려고 노력했지만 이론적으로 굉장히 좋다고 생각되는 모델들이 오히려 안좋다고 생각한 모델들 보다 성능이 떨어지는 결과를 보고 task에 따라 적절한 detector model이 존재한다고 느꼈습니다. backbone으로는 transformer,convnext가 가장 성능이 좋았으며 나머지(resnet,darknet,VGG16)backbone은 좋은 성능을 내지 못했습니다. (yolov5,yoloR 제외) feature map을 뽑는데 있어서는 모델이 정해져있는것 같은 느낌을 받았습니다. 모든 모델에 같은 augmentation을 적용해보았지만 어떤 모델은 성능이 올라가고 나머지는 그렇지 못했습니다. augmentation이 만능은 아닌것 같고 모델에 맞는 task에 맞는 augmentation이 따로 존재합니다. 그리고 대회를 진행하는데 있어 다양한 모델 with 어느정도의 성능이 나오는 모델을 WBF로 앙상블 했을 때 성능이 올라가는 것을 확인했습니다.

아쉬운점
mmdetection 사용법에 익숙해지는데 시간이 오래걸렸고 detr,DINO model의 backbone을 바꾸려고 시도했지만 결국엔 못바꾸고 시간만 버렸습니다. 체계적으로 실험을 구상하고 행동으로 옮긴다는 생각을 가지고 있었지만 제대로 이루어지지 않았습니다. 다른 팀원들의 실험 내용을 다 알지 못했고 팀에 기여도가 낮다고 느꼈고 대회를 진행하면서 도움을 주기보다는 도움을 많이 받았습니다. EDA를 해봤지만 클래스의 불균형이 있다는 것만 확인을 하였고 팀원이 만들어준 코드(by 진아님)를 통해 이미지 내에서 bbox를 시각화 할 수있었지만 아직 인사이트가 부족해서 아 그렇구나라는 생각만 들고 대회를 진행함에 있어 EDA를 잘활용하지 못한것 같아 아쉽습니다. mmdetection에서 클래스별 mAP를 출력을 해보고 general trash를 잘 못잡고 못잡는 이유는 small object이고 EDA로 봤을 때 상대적으로 데이터가 적기 때문이라는 것을 인지 했지만 small object를 잘 잡기 위해 여러가지 모델을 써보며 mAP_s를 기준으로 무작정 실험을 했습니다.(아직도 모르겠네요)

앞으로 해야할 점
실험을 덜 하더라도 논문을 완벽하게 읽을 것 입니다. 하나의 모델을 완벽하게 알고 있어야 더 잘 활용할수 있기 때문입니다. 무작정 실험하지 말고 머릿속으로 개념을 알고 사용할 것이고 협업의 관점에서 더 커뮤니케이션이 잘 이루어지고 프로젝트의 모든 것을 알도록 노력할 것입니다.

autoAug,mosaic,mixup 등의 기법을 사용하고 싶었지만 error를 처리하지 못해 사용하지 못했습니다. 다음엔 꼭 이러한 방법들을 적용할 것입니다.

협업이 잘 이루어지도록 많은 노력을 할것 입니다.
