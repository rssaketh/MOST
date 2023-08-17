import cv2
import torch
import skimage.io
import numpy as np
import torch.nn as nn
from PIL import Image
import pdb

import matplotlib.pyplot as plt


def visualize_map(image, pred, gt_img, vis_folder, im_name, all_preds=None):
    """
    Visualization of the predicted box and the corresponding seed patch.
    """
    if not isinstance(pred, list):
        pred = [pred]
    pred = np.stack(pred, axis=0).max(0)
    if pred.shape != gt_img.shape:
        pred = pred[:gt_img.shape[0], :gt_img.shape[1]]
    if image.shape != gt_img.shape:
        image = image[:gt_img.shape[0], :gt_img.shape[1]]

    pred_sig = pred
    widths = [image.shape[1], pred_sig.shape[1]]
    heights = [image.shape[0], pred_sig.shape[0]]
    gap_img = Image.fromarray(255*np.ones((image.shape[0], 5)))

    ims = [Image.fromarray(image), gap_img,
           Image.fromarray(255*(pred_sig>0).astype(float)), gap_img]
    if all_preds:
        # others are given
        stack_preds = np.stack(all_preds, axis=0).max(0)
        widths.append(stack_preds.shape[1])
        heights.append(stack_preds.shape[0])
        ims.append(Image.fromarray(255*(stack_preds>0).astype(float)))
        ims.append(gap_img)
    widths.append(gt_img.shape[1])
    heights.append(gt_img.shape[0])
    ims.append(Image.fromarray(gt_img))
    total_width = sum(widths) + 15
    max_height = max(heights)
    new_im = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im in ims:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    pltname = f"{vis_folder}/MOST_{im_name}.png"
    new_im.save(pltname)
    print(f"Predictions saved at {pltname}.")

def visualize_predictions(image, pred, seed, scales, dims, vis_folder, im_name,
                          k_suffix, plot_seed=False):
    """
    Visualization of the predicted box and the corresponding seed patch.
    """
    w_featmap, h_featmap = dims
    if not isinstance(pred, list):
        pred = [pred]
    for ii, pre in enumerate(pred):
        area = (pre[3] - pre[1]) * (pre[2] - pre[0])
        if area < 1000:
            continue
        # Plot the box
        cv2.rectangle(
            image,
            (int(pre[0]), int(pre[1])),
            (int(pre[2]), int(pre[3])),
            (255, 0, 0), 3,
        )

        # Plot the seed
        if plot_seed:
            s_ = np.unravel_index(seed.cpu().numpy(), (w_featmap, h_featmap))
            size_ = np.asarray(scales) / 2
            cv2.rectangle(
                image,
                (int(s_[1] * scales[1] - (size_[1] / 2)), int(s_[0] * scales[0] - (size_[0] / 2))),
                (int(s_[1] * scales[1] + (size_[1] / 2)), int(s_[0] * scales[0] + (size_[0] / 2))),
                (0, 255, 0), -1,
            )

    pltname = f"{vis_folder}/LOST_{im_name}_{k_suffix}.png"
    Image.fromarray(image).save(pltname)
    print(f"Predictions saved at {pltname}.")

