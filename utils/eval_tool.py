from __future__ import division

from collections import defaultdict
import itertools
import numpy as np
import six
import torch
from tqdm import tqdm

from model.utils.bbox_tools import bbox_iou
from utils import array_tool as at

def run_test(net, test_loader):
    # Evaluation for VOC dataset
    pred_bboxes, pred_labels, pred_scores = [], [], []
    gt_bboxes, gt_labels, gt_difficults = [], [], []
    with torch.no_grad():
        for img, bbox, label, scale, ori_size, difficult in tqdm(test_loader):
            scale = at.scalar(scale)
            original_size = [ori_size[0][0].item(), ori_size[1][0].item()]
            pred_bbox, pred_label, pred_score = net(img, None, None, scale,original_size)
            gt_bboxes += list(bbox.numpy())
            gt_labels += list(label.numpy())
            gt_difficults += list(difficult.numpy())
            pred_bboxes += [pred_bbox]
            pred_labels += [pred_label]
            pred_scores += [pred_score]
    
    return pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, gt_difficults


def voc_ap(net, test_loader):
    # Evaluation for VOC dataset

    # initialize evaluation metrics
    map_result = {'mAP': 0, 'mAP_0.5': 0, 'mAP_0.75': 0}
    iou_threshes = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    # run test on test dataset
    pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, gt_difficults = run_test(net, test_loader)
    f1_list , recall_list ,precision_list,  = [], [], []
    # evaluate results regardless of area size
    for iou_thresh in iou_threshes:
        result = eval_voc(pred_bboxes, pred_labels, pred_scores,
                          gt_bboxes, gt_labels, gt_difficults,
                          iou_thresh, use_07_metric=True)
        # accumulate results
        map_result['mAP'] += result['map']
        f1_list.append(result['f1'])
        recall_list.append(result['recall'])
        precision_list.append(result['precision'])
        # save map for iou 0.5 & 0.75
        if iou_thresh == 0.5:
            map_result['mAP_0.5'] = result['map']
        elif iou_thresh == 0.75:
            map_result['mAP_0.75'] = result['map']
    
    map_result['mAP'] /= 10
    mean_f1 = np.mean(f1_list)
    mean_recall = np.mean(recall_list)
    mean_precision = np.mean(precision_list)
    # print results
    print('Result: mAP=={:.2f} | mAP@0.5== {:.2f} | mAP@0.75=={:.2f}'.format(map_result['mAP']*100, map_result['mAP_0.5']*100, map_result['mAP_0.75']*100))
    print(f'Result: F1-Score=={mean_f1} | Recall== {mean_recall} | Precision=={mean_precision}')

    return map_result['mAP']


def eval_voc(pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, gt_difficults=None,
        iou_thresh=0.5, use_07_metric=False):
    """Calculate average precisions based on evaluation code of PASCAL VOC."""

    precision, recall = voc_prec_rec(pred_bboxes, 
                                          pred_labels, 
                                          pred_scores,
                                          gt_bboxes, 
                                          gt_labels, 
                                          gt_difficults,
                                          iou_thresh=iou_thresh)

    ap = calc_detection_voc_ap(precision, recall, use_07_metric=use_07_metric)
    mean_f1, mean_recall, mean_precision = calc_f1_prec_recall(precision, recall)
    #print(f'Result: F1-Score=={mean_f1} | Recall== {mean_recall} | Precision=={mean_precision}')
    return {'ap': ap, 'map': np.nanmean(ap), 'precision': mean_precision, 'recall' : mean_recall, 'f1': mean_f1}


def voc_prec_rec(
        pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels,
        gt_difficults=None,
        iou_thresh=0.5):

    pred_bboxes = iter(pred_bboxes)
    pred_labels = iter(pred_labels)
    pred_scores = iter(pred_scores)
    gt_bboxes = iter(gt_bboxes)
    gt_labels = iter(gt_labels)
    if gt_difficults is None:
        gt_difficults = itertools.repeat(None)
    else:
        gt_difficults = iter(gt_difficults)

    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)

    for pred_bbox, pred_label, pred_score, gt_bbox, gt_label, gt_difficult in \
            six.moves.zip(
                pred_bboxes, pred_labels, pred_scores,
                gt_bboxes, gt_labels, gt_difficults):

        if gt_difficult is None:
            gt_difficult = np.zeros(gt_bbox.shape[0], dtype=bool)

        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            pred_mask_l = pred_label == l
            pred_bbox_l = pred_bbox[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]
            # sort by score
            order = pred_score_l.argsort()[::-1]
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]

            gt_mask_l = gt_label == l
            gt_bbox_l = gt_bbox[gt_mask_l]
            gt_difficult_l = gt_difficult[gt_mask_l]

            n_pos[l] += np.logical_not(gt_difficult_l).sum()
            score[l].extend(pred_score_l)

            if len(pred_bbox_l) == 0:
                continue
            if len(gt_bbox_l) == 0:
                match[l].extend((0,) * pred_bbox_l.shape[0])
                continue

            # VOC evaluation follows integer typed bounding boxes.
            pred_bbox_l = pred_bbox_l.copy()
            pred_bbox_l[:, 2:] += 1
            gt_bbox_l = gt_bbox_l.copy()
            gt_bbox_l[:, 2:] += 1

            iou = bbox_iou(pred_bbox_l, gt_bbox_l)
            gt_index = iou.argmax(axis=1)
            # set -1 if there is no matching ground truth
            gt_index[iou.max(axis=1) < iou_thresh] = -1
            del iou

            selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
            for gt_idx in gt_index:
                if gt_idx >= 0:
                    if gt_difficult_l[gt_idx]:
                        match[l].append(-1)
                    else:
                        if not selec[gt_idx]:
                            match[l].append(1)
                        else:
                            match[l].append(0)
                    selec[gt_idx] = True
                else:
                    match[l].append(0)

    for iter_ in (
            pred_bboxes, pred_labels, pred_scores,
            gt_bboxes, gt_labels, gt_difficults):
        if next(iter_, None) is not None:
            raise ValueError('Length of input iterables need to be same.')

    n_fg_class = max(n_pos.keys()) + 1
    precision = [None] * n_fg_class
    recall = [None] * n_fg_class

    for l in n_pos.keys():
        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)

        order = score_l.argsort()[::-1]
        match_l = match_l[order]

        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)

        # If an element of fp + tp is 0,
        # the corresponding element of prec[l] is nan.
        precision[l] = tp / (fp + tp)
        # If n_pos[l] is 0, rec[l] is None.
        if n_pos[l] > 0:
            recall[l] = tp / n_pos[l]

    return precision, recall


def calc_detection_voc_ap(prec, rec, use_07_metric=False):
    """Calculate average precisions based on evaluation code of PASCAL VOC.

    This function calculates average precisions
    from given precisions and recalls.
    The code is based on the evaluation code used in PASCAL VOC Challenge.

    Args:
        prec (list of numpy.array): A list of arrays.
            :obj:`prec[l]` indicates precision for class :math:`l`.
            If :obj:`prec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        rec (list of numpy.array): A list of arrays.
            :obj:`rec[l]` indicates recall for class :math:`l`.
            If :obj:`rec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        use_07_metric (bool): Whether to use PASCAL VOC 2007 evaluation metric
            for calculating average precision. The default value is
            :obj:`False`.

    Returns:
        ~numpy.ndarray:
        This function returns an array of average precisions.
        The :math:`l`-th value corresponds to the average precision
        for class :math:`l`. If :obj:`prec[l]` or :obj:`rec[l]` is
        :obj:`None`, the corresponding value is set to :obj:`numpy.nan`.

    """

    n_fg_class = len(prec)
    ap = np.empty(n_fg_class)
    for l in six.moves.range(n_fg_class):
        if prec[l] is None or rec[l] is None:
            ap[l] = np.nan
            continue

        if use_07_metric:
            # 11 point metric
            ap[l] = 0
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec[l] >= t) == 0:
                    p = 0
                else:
                    p = np.max(np.nan_to_num(prec[l])[rec[l] >= t])
                ap[l] += p / 11
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mpre = np.concatenate(([0], np.nan_to_num(prec[l]), [0]))
            mrec = np.concatenate(([0], rec[l], [1]))

            mpre = np.maximum.accumulate(mpre[::-1])[::-1]

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap[l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap



def calc_f1_prec_recall(precision_list, recall_list):
    """
    Calculate the maximum F1-score for each class from per-class
    precision and recall arrays, and then average these maxima
    across valid classes.
    
    This function will:
      1. Iterate over each class's precision and recall arrays.
      2. Compute the element-wise F1-score = 2 * p * r / (p + r).
      3. Identify the maximum F1-score, and the corresponding precision and recall.
      4. Collect these values for each class and then average them.
    
    Parameters
    ----------
    precision_list : list of numpy.ndarray or None
        precision_list[l] is the array of precision values for class l.
        A None value indicates there were no valid detections or ground-truths
        for that class.
    recall_list : list of numpy.ndarray or None
        recall_list[l] is the array of recall values for class l.
        A None value indicates there were no valid detections or ground-truths
        for that class.
    
    Returns
    -------
    mean_f1 : float
        The average (across classes) of the maximum F1-score.
        If no classes are valid (all None), returns 0.0.
    mean_recall : float
        The average recall across classes at the threshold
        giving maximum F1. If no classes are valid, returns 0.0.
    mean_precision : float
        The average precision across classes at the threshold
        giving maximum F1. If no classes are valid, returns 0.0.
    """
    f1_values, recall_vals, precision_vals = [], [], []

    for p_arr, r_arr in zip(precision_list, recall_list):
        # Skip classes with no predictions or no ground truth.
        if p_arr is None or r_arr is None or len(p_arr) == 0 or len(r_arr) == 0:
            continue
        
        # Compute F1 = 2 * p * r / (p + r) (element-wise).
        denominator = (p_arr + r_arr + 1e-12)
        f1_arr = (2 * p_arr * r_arr) / denominator
        
        # Index of maximum F1 for this class
        idx_max_f1 = np.argmax(f1_arr)
        
        f1_values.append(f1_arr[idx_max_f1])
        recall_vals.append(r_arr[idx_max_f1])
        precision_vals.append(p_arr[idx_max_f1])
    
    if len(f1_values) == 0:
        # If no valid classes, return zeros or you can raise a warning or return np.nan
        return 0.0, 0.0, 0.0
    
    # Compute the average across valid classes
    mean_f1 = np.mean(f1_values)
    mean_recall = np.mean(recall_vals)
    mean_precision = np.mean(precision_vals)
    
    return mean_f1, mean_recall, mean_precision


def voc_f1_recall_precision(net, test_loader, iou_thresh=0.5):
    """
    Compute the mean F1-score, recall, and precision for detections
    given an IoU threshold, using the same data pipeline as your
    VOC evaluation (including `run_test` and `voc_prec_rec`).
    
    This function:
      1. Runs inference on the test_loader to gather predicted boxes, labels, scores,
         and the corresponding ground-truth data (via `run_test`).
      2. Calculates per-class precision and recall arrays (via `voc_prec_rec`).
      3. Derives maximum F1, recall, and precision for each class, then averages
         over valid classes.
    
    Parameters
    ----------
    net : torch.nn.Module
        Your trained detection network.
    test_loader : torch.utils.data.DataLoader
        The DataLoader for your test set.
    iou_thresh : float, optional (default=0.5)
        The IoU threshold for a prediction to be considered correct.
    
    Returns
    -------
    mean_f1 : float
        The average of the maximum F1-scores over classes with at least one ground truth.
    mean_recall : float
        The average recall corresponding to the max F1 per class.
    mean_precision : float
        The average precision corresponding to the max F1 per class.

    Notes
    -----
    - This function follows the same iterative scheme used in PASCAL VOC to
      define a true-positive (IoU >= iou_thresh to a ground-truth bounding box).
    - The existing logic (run_test, voc_prec_rec) ensures consistent calculation
      across mAP, precision, recall, and F1.
    """
    # 1. Run detection/ inference on the test dataset
    pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, gt_difficults = run_test(net, test_loader)

    # 2. Compute per-class precision and recall arrays
    precision_list, recall_list = voc_prec_rec(
        pred_bboxes=pred_bboxes,
        pred_labels=pred_labels,
        pred_scores=pred_scores,
        gt_bboxes=gt_bboxes,
        gt_labels=gt_labels,
        gt_difficults=gt_difficults,
        iou_thresh=iou_thresh
    )

    # 3. Compute max-F1 from the precision-recall curves
    mean_f1, mean_recall, mean_precision = calc_f1_prec_recall(precision_list, recall_list)

    return mean_f1, mean_recall, mean_precision

#cloner174