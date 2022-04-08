import os
import pandas as pd
import argparse

string_pred = []
name = []

import os.path 

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--exp', help='exp number', type=int,)
    args = parser.parse_args()
    return args

args = parse_args()
exp_number = args.exp

for idx in range(4871):
    word =  "%04d" % (idx)
    path = f"/opt/ml/yolo/yolov5/runs/detect/exp{exp_number}/labels/{word}.txt"
    empty_path = f"/opt/ml/yolo/yolov5/runs/detect/exp2/labels/{word}.txt"

    if(os.path.isfile(path) == False):
        print('empty:', word, empty_path) #empty_path
        path = empty_path

    f = open(path,"rt")
    input_word = ''
    line = f.readlines()
    for i in line:
        arr = i.split()
        # print(arr, idx)
        xc, yc, w, h = float(arr[1]) * 1024, float(arr[2]) * 1024, float(arr[3]) * 1024, float(arr[4]) * 1024   
        xmin = xc - w / 2
        ymin = yc - h / 2
        w += xmin
        h += ymin
        arr[1] = arr[5]
        arr[2], arr[3], arr[4], arr[5] = str(round(xmin,3)), str(round(ymin,3)), str(round(w, 3)), str(round(h,3))
        temp = ' '.join(arr)
        input_word = temp + ' ' +input_word

    string_pred.append(input_word)    
    name.append(f'test/{word}.jpg')

submission = pd.DataFrame()
submission['PredictionString'] = string_pred
submission['image_id'] = name
submission.to_csv(os.path.join('/opt/ml/yolo/yolov5/runs/submission', f'submission_exp{exp_number}.csv'), index=None)
submission.head()