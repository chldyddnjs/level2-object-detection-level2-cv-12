# object detection 팀 랩업리포트

# 1. 프로젝트 개요

사진에서 쓰레기를 Detection 하는 모델을 만들어 이러한 문제점을 해결해보고자 한다. 문제 해결을 위한 데이터셋으로는 일반 쓰레기, 플라스틱, 종이, 유리 등 10 종류의 쓰레기가 찍힌 사진 데이터셋이 제공된다.

이 데이터셋으로 쓰레기 사진이 들어있는 input data 에서 Object detection task를 진행하여 분리수거를 하는 것이 이번 competition의 목표이다.

- 모델 개요

![KakaoTalk_20220408_170052363.jpg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8c2b781d-fe1d-4136-a2e7-7a614d26ce33/KakaoTalk_20220408_170052363.jpg)

# 2. 프로젝트 팀 구성 및 역할

- 김승현: YOLO 모델 테스트, ensemble
- 노창현: tood 모델 테스트, data augmentation
- 최홍록: detector 테스트 , data EDA
- 최진아: htc, swin backbone 모델 테스트 , data augmentation
- 최용원: DETR 테스트, ensemble

 

# 3. 프로젝트 수행 절차 및 결과

## 탐색적 분석 및 전처리

- **Detection 예시**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7b0db154-dccd-418f-ad40-a32c9f07805e/Untitled.png)

- **class별 분포**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d0f2a495-261b-47e2-9c13-a4b54fd5daf6/Untitled.png)

- **bbox area 분포**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/256781e9-930a-438b-9d48-a5aa78baa72e/Untitled.png)

- **class 별 bbox area 분포 - S(0~32**2), M(32**2~96**2), L(96**2~)**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/6e3bdaeb-6e2c-4b84-bf8d-a86031f7a1f8/Untitled.png)

- **학습시 class별 AP**
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8b662558-1ea9-46cb-900c-5be689aadb13/Untitled.png)
    
    - class의 개수는 model 성능에 유의미한 영향을 미치지는 않는 것으로 파악

train.json파일에서 bbox정보를 가져와 시각화를 진행 후 class별 area의 사이즈를 파악 그리고 이미지 당 몇 개의 class, bbox를 가지고 있는지 파악 후 어떻게 할 지 정하였다.

## K-FOLD

KFOLD를 적용시키기 위해 각 클래스 별 분포에 맞게 FOLD를 나눠주었다.

각 FOLD 당 train set : validation set = 8 : 2 비율로 나누어 주어 FOLD 별 mAP를 확인해 일반화 성능을 확인하였다. 이를 통해 Cross validation을 통해 모델의 일반화를 하였다.

## Data Augmentation

augmentation은 학습 시 torchvision보다 빠르다고 알려져 있는 albumentation 라이브러리를 사용했으며 generalization performance를 끌어올리기 위해 여러가지 transform을 적용하였다.

augmentation은 학습 시 torchvision보다 빠르다고 알려져 있는 albumentation 라이브러리를 사용했으며 generalization performance를 끌어올리기 위해 여러가지 transform을 적용하였다.

또한 일반화 성능을 끌어올리기 위한 trick으로 TTA를 적용하였다.

- brightness
- contrast
- saturation
- Hue
- RGBshift
- random image rotate
- random flip
- blur
- medium blur
- Mixup
- mosaic
- resize
- normalize

## Optimizer , Learning Scheduler

learning rate scheduler 및 optimizer를 다양한 종류로 실험해가며 빠르고 정확하게 Local minima 찾아갈 수 있게 하였다. 최종적으로 momentum SGD, adamW와 cosine annealing을 사용하였다.

learning rate을 0.01에서 1/10씩 줄여가며 총 세 번 실험을 하였다. (learning rate이 줄어들수록 그래프가 부드러워지는 효과를 볼 수 있고 local minimum에 근접할 수 있는 효과를 봄)

task에 맞는 detector와 extractor를 결정하기 위해 여러가지 model을 사용함으로 써 최적의 조합을 찾는 model search를 진행하였다.

대회를 진행하는 도중에 다시 EDA를 진행하여 class 별 mAP를 확인하였고 general trash의 mAP가 낮은 것을 확인한 후 small area 부분을 더 잘 찾는 model을 search하였다.

## Model

- One Stage detector
    - Yolor
    - yolov5
    - yolox
    - detr
    - TOOD
        - resnet
        - convnext
- Two Stage detector
    - Cascade Rcnn
        - ConvNeXt
        - resnet
        - Swin Transformer
    - Faster Rcnn
        - vgg16
        - swin Transformer
    - HTC
        - swin Transformer
    - Double Head Faster Rcnn
        - Swin Transformer
    - Efficientdet
        - efficientnet b7

small object에 대한 mAP가 낮은 것을 확인하였고 이에 anchor box를 조정하였다.

다양한 실험을 위해 rpn_haed 변경 및 cascade stage 수 추가를 시도해보았다.

## Ensemble

cross validation을 포함한 model을 학습시키며 performance가 잘  나온 것들을 추려내었고 그 중에서도 mAP_s, mAP_l, mAP_m 성능이 상대적으로 뛰어난 것들을 추려내어 WBF를 적용하였다. (총 30개의 csv 파일을 사용)

### 결과물

 

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/3ad7a56d-ca1a-4109-af62-2b415a25e2a0/Untitled.png)

- wandb 결과 예시

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2b950fb6-2e0b-4976-953c-ce7af78b9c02/Untitled.png)

- 대회 결과
LB score - **public : 0.7152**
              - **private : 0.7017**
    
    **Final Rank : 6/19 th place**
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/0714e3a0-46da-44c1-885c-84990fab6dbc/Untitled.png)
    
- **최종 모델**
    
    
    | Detector | backbone | fold |
    | --- | --- | --- |
    | cascade rcnn | convnext  | 0~5 |
    | cascade rcnn | swin transformer | 0, 1, 3 |
    | TOOD             | convnext  | 0~5 |
    | htc | swin transformer | 0~5 |
    | double head | swin transformer | 0 |
    | YOLOr | - | 0~5 |
    | YOLOv5 | - | 4 |

# 5. 자체 평가 의견

- **잘한 점들**
    - 다양한 모델들을 앙상블하여 성능 향상을 할 수 있었다.
    - 협업을 통해 긴 시간 학습하는 모델의 단점을 보완할 수 있었다.
    - 협업툴로써 **[Weights & Biases](https://wandb.ai/site)**를 잘 활용했다
    - 여러 모델들을 사용하며 mmdetection 에 조금 더 적응하고 wandb 를 통한 협업을 경험하였다
- **시도 했으나 잘 되지 않았던 것들**
    - general trash에 대한 mAP가 낮아 data manufacturing을 추천받아 시도하려고 하였지만       실패하였다.
    - SOTA로 알려진 모델들을 사용하였지만, cascade RCNN + Swin 이외의 모델들은 성능이 높지 않았다.
    - 최근 나온 모델들을 더 사용해보고 싶었지만 적용을 제대로 못한 모델들이 있었다
    
- **아쉬웠던 점들**
    - 모델에 대한 이해가 부족한 상태로 성능 위주의 실험을 진행했다.
    - mmdetection 라이브러리의 이해 부족으로 cutmix, mixup, mosaic 같은 augmentation을 늦게 사용하여 적은 수의 모델에만 적용하였다.
    - github을 활용한 협업이 미숙했다.
