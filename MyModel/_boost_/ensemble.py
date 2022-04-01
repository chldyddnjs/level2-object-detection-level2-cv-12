import pandas as pd
from ensemble_boxes import *
import numpy as np
from pycocotools.coco import COCO
from ImgShow import *

# ensemble csv files
submission_files = ['/opt/ml/detection/baseline/mmdetection/configs/_boost_/ensembleData/cascade_swinB.csv',
                   '/opt/ml/detection/baseline/mmdetection/configs/_boost_/ensembleData/htc_swin_large_2x_resize1024_aug_submission_15.csv',
                   '/opt/ml/detection/baseline/mmdetection/configs/_boost_/ensembleData/htc_swin_large_aug.csv'] # submission lists
submission_df = [pd.read_csv(file) for file in submission_files]

image_ids = submission_df[0]['image_id'].tolist()

# ensemble 할 file의 image 정보를 불러오기 위한 json
annotation = '../../dataset/test.json'
coco = COCO(annotation)

prediction_strings = []
file_names = []
# ensemble 시 설정할 iou threshold 이 부분을 바꿔가며 대회 metric에 알맞게 적용해봐요!
iou_thr = 0.55
skip_box_thr = 0.1
sigma = 0.1

# 각 image id 별로 submission file에서 box좌표 추출
for i, image_id in enumerate(image_ids):
    prediction_string = ''
    boxes_list = []
    scores_list = []
    labels_list = []
    image_info = coco.loadImgs(i)[0]
#     각 submission file 별로 prediction box좌표 불러오기
    for df in submission_df:
        predict_string = df[df['image_id'] == image_id]['PredictionString'].tolist()[0]
        predict_list = str(predict_string).split()
        
        if len(predict_list)==0 or len(predict_list)==1:
            continue
            
        predict_list = np.reshape(predict_list, (-1, 6))
        box_list = []
        
        for box in predict_list[:, 2:6].tolist():
            box[0] = float(box[0]) / image_info['width']
            box[1] = float(box[1]) / image_info['height']
            box[2] = float(box[2]) / image_info['width']
            box[3] = float(box[3]) / image_info['height']
            box_list.append(box)
            
        boxes_list.append(box_list)
        scores_list.append(list(map(float, predict_list[:, 1].tolist())))
        labels_list.append(list(map(int, predict_list[:, 0].tolist())))
    
    # 예측 box가 있다면 이를 ensemble 수행
    if len(boxes_list):
        boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        
        for box, score, label in zip(boxes, scores, labels):
            prediction_string += str(int(label)) + ' ' + str(score) + ' ' + str(box[0] * image_info['width']) + ' ' + str(box[1] * image_info['height']) + ' ' + str(box[2] * image_info['width']) + ' ' + str(box[3] * image_info['height']) + ' '
    
    prediction_strings.append(prediction_string)
    file_names.append(image_id)

submission = pd.DataFrame()
submission['PredictionString'] = prediction_strings
submission['image_id'] = file_names
submission.to_csv('/opt/ml/detection/baseline/mmdetection/configs/_boost_/ensembleData/submission_ensemble_WBF1.csv',index=False)

submission.head()
