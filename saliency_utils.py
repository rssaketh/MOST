import numpy as np
from sklearn.metrics import jaccard_score
import pdb


def saliency_metrics(pred_list, gt_img):
    # pred_list is list of masks from different seeds
    # add them all up
    if not pred_list:
        return 0, 0, 0
    # pdb.set_trace()
    pred = np.stack(pred_list, axis=0).max(0)
    if pred.shape != gt_img.shape:
        pred = pred[:gt_img.shape[0], :gt_img.shape[1]]
    # check if gt has values between 0-1
    if gt_img.max() > 1:
        gt_img = gt_img.astype(float)
        gt_img /= 255.
    if gt_img.ndim > 2:
        pdb.set_trace()
    pred_sig = 1./(1+np.exp(-pred))
    # get fbmax
    fbmax = compute_fbmax(gt_img, pred_sig)
    # get IoU with binarization threshold 0.5
    pred_bin = pred_sig > 0.5
    try:
        iou = jaccard_score(gt_img.reshape(-1).astype(bool), pred_bin.reshape(-1))
    except:
        pdb.set_trace()

    # get Acc with binarization threshold 0.5
    acc = (pred_bin == gt_img).sum()/np.prod(gt_img.shape)

    return fbmax, iou, acc


def compute_fbmax(gt, pred, levels=255):
    beta = 0.3
    thresholds = np.linspace(0, 1-1e-10, levels)
    prec = np.zeros(levels)
    recall = np.zeros(levels)

    for i, th in enumerate(thresholds):
        bin_pred = (pred >= th).astype(float)
        tp = (bin_pred * gt).sum()
        prec[i] = tp/(bin_pred.sum() + 1e-15)
        recall[i] = tp/(gt.sum() + 1e-15)
    f_score = (1+beta**2)*prec * recall/((beta**2*prec) + recall)
    f_score[np.isnan(f_score)] = 0
    return f_score.max()
