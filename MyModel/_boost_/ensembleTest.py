from ensemble_boxes import *
# from ImgShow import *
boxes_list = [[
    # [0.00, 0.51, 0.81, 0.91],
    [0.10, 0.31, 0.71, 0.61],
    [0.01, 0.32, 0.83, 0.93],
    [0.02, 0.53, 0.11, 0.94],
    [0.03, 0.24, 0.12, 0.35],
],[
    [0.04, 0.56, 0.84, 0.92],
    [0.12, 0.33, 0.72, 0.64],
    [0.38, 0.66, 0.79, 0.95],
    [0.08, 0.49, 0.21, 0.89],
]]
scores_list = [[0.8, 0.2, 0.4, 0.7], [0.5, 0.8, 0.7, 0.3]]
labels_list = [[1, 0, 1, 1], [1, 1, 1, 0]]
weights = [2, 1]

iou_thr = 0.55
skip_box_thr = 0.1
sigma = 0.1


# show_boxes(boxes_list, scores_list, labels_list)
boxes, scores, labels = soft_nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, sigma=sigma, thresh=skip_box_thr)

boxes0, scores0, labels0 = nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr)
boxes2, scores2, labels2= non_maximum_weighted(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
boxes3, scores3, labels3 = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)

# print(len(scores0),len(scores2),len(scores3))
print(scores0)
# print(scores1)
# print(scores2)
# print(scores3)
# print(boxes0)
# print(boxes2
# print(boxes3)
# boxes = [boxes0,boxes2,boxes3]
# labels = [labels0,labels2,labels3]
# scores = [scores0,scores2,scores3]
# target_idxs = []
# cnt=0 #몇번 째 area인지
# for s0,s2,s3 in zip(scores0,scores2,scores3):
    
#     target_list = [s0,s2,s3]
#     target = max(target_list)
#     for i in range(len(target_list)):
#         if target_list[i] == target:
#             target_idx = i #몇번 째 box인지
#             target_idxs.append((cnt,i))
#             print(boxes[i][cnt],labels[i][cnt],scores[i][cnt])
#     cnt+=1
# print(target_idxs)
# show_boxes([boxes], [scores], [labels.astype(np.int32)])
