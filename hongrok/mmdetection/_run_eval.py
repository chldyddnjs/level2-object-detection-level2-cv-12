import os
config_file = 'configs/_practice/sparseRCNN/sparse_rcnn_pvt_v2_b2_fpn_300_proposals_crop_mstrain_480-800_3x_coco.py'
check_point='/opt/ml/detection/hongrok/mmdetection/work_dirs/sparse_rcnn_pvt_v2_b2_fpn_300_proposals_crop_mstrain_480-800_3x_coco/latest.pth'
cmd = f'python tools/test.py {config_file} {check_point} --eval bbox --gpu_id 0' 

os.system(cmd)