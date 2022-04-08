import os 
import sys
import pickle
import argparse
from os import path as osp

import pandas as pd
from mmcv import Config
from pycocotools.coco import COCO

# export PYTHONPATH="${PYTHONPATH}:/opt/ml/detection/hongrok/mmdetection"

def make_submission(output, config_file, model_ver, work_dir):
    # submission 양식에 맞게 output 후처리
    cfg = Config.fromfile(config_file)   
    prediction_strings = []
    file_names = []
    coco = COCO(cfg.data.test.ann_file)
    # img_ids = coco.getImgIds()

    class_num = 10
    for i, out in enumerate(output):
        prediction_string = ''
        image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
        for j in range(class_num):
            for o in out[j]:
                prediction_string += str(j) + ' ' + str(o[4]) + ' ' + str(o[0]) + ' ' + str(o[1]) + ' ' + str(
                    o[2]) + ' ' + str(o[3]) + ' '
            
        prediction_strings.append(prediction_string)
        file_names.append(image_info['file_name'])

    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    path = f'{work_dir}outputs/'
    if not osp.isdir(path):
        os.mkdir(path)
    submission.to_csv(f'{path}submission_{model_ver.split(".")[0]}.csv', index=None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=1, type=int)
    parser.add_argument('--inference', default=0, type=int)
    args = parser.parse_args()
    ## parmas
    # config_file='configs/_practice/cascade_rcnn_swin-t-p4-w7_fpn_1x_coco.py'
    # config_file='configs/_practice/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py' # config file
    # config_file='configs/_practice/convnext/cascade_mask_rcnn_convnext_xlarge_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco_in22k.py'
    # config_file = 'configs/_practice/vfnet/vfnet_r2_101_fpn_mstrain_2x_coco.py'
    # config_file = 'configs/_practice/tood/tood_r101_fpn_dconv_c3-c5_mstrain_2x_coco.py'
    # config_file = 'configs/_practice/double_head/dh_faster_rcnn_r50_fpn_1x_coco.py'
    # config_file = 'configs/_practice/double_head/dh_faster_rcnn_swin_fpn_1x_coco.py'
    # config_file = 'configs/_practice/libra/libra_faster_rcnn_convnext_fpn_1x_coco.py'
    # config_file = 'configs/_practice/cascade/cascade_rcnn_convnext_fpn_1x_coco.py'
    # config_file = 'configs/_jina/htc/htc_swin-large_fpn_2x_coco_aug_latest.py'
    # config_file = 'configs/_practice/sparseRCNN/sparse_rcnn_convnext.py'
    # config_file = 'configs/_practice/sparseRCNN/sparse_rcnn_swinS.py'
    # config_file = 'configs/_practice/sparseRCNN/sparse_rcnn_pvt_v2_b2_fpn_300_proposals_crop_mstrain_480-800_3x_coco.py'
    config_file = 'configs/_practice/cascade/cascade_rcnn_pvt_v2_b2_fpn_3x_mstrain_fp16.py'
    # config_file = 'configs/_practice/double_head/dh_faster_rcnn_swin_fpn_1x_coco.py'
    basename = osp.basename(config_file).split('.')[0]
    work_dir=f'work_dirs/{basename}/' # 학습결과가 저장될 폴더
    user_name='hongrok' # wandb에 올라갈 실험한 유저이름
    fold_num=0 # 사용할 데이터 fold 번호
    wandb_exp=basename # wandb에 올라갈 실험이름
    epochs = 24 # 실행할 epochs
    resume_from = None
    # resume_from = '/opt/ml/detection/hongrok/mmdetection/work_dirs/dh_faster_rcnn_swin_fpn_1x_coco_rotate90/epoch_5.pth'
    if args.train:
        # train
        cmd =f'python tools/train.py {config_file}\
                --work-dir {work_dir}\
                --user_name {user_name}\
                --fold_num {fold_num}\
                --wandb_exp {wandb_exp}\
                --epochs {epochs}'
        if resume_from:
            cmd += f' --resume-from {resume_from}'
        os.system(cmd)
        # print(cmd)

    # inference
    if args.inference:
        
        ## params
        model_ver = 'best.pth' # 사용하고자하는 모델 버전

        if model_ver == 'best.pth':
            model_ver = sorted([file for file in os.listdir(work_dir) if 'best' in file])[0]

        model_pth = work_dir + model_ver
        score_thr = 0.05 #  detections with scores below this threshold will be removed.
        out_name = f'{work_dir}outputs/output.pkl'


        # 결과폴더 생성
        path = f'{work_dir}outputs/'
        if not osp.isdir(path):
            os.mkdir(path)

        cmd = f'python tools/test.py {config_file} {model_pth}\
                --out {out_name}\
                --show-score-thr {score_thr}'
        os.system(cmd)
        
        with open(out_name, mode='rb') as f:
            output = pickle.load(f)
        make_submission(output, config_file, model_ver, work_dir)


