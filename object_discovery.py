import torch
import scipy
import scipy.ndimage
from collections import Counter
import time
import datetime
import os
import random

import numpy as np
from torchvision.ops import batched_nms
from itertools import product
from sklearn.cluster import DBSCAN


def entropy_tensor(vals):
    _, counts = torch.unique(vals, return_counts=True)
    freq = counts/torch.sum(counts)
    ent = -1 * torch.sum(freq *
                         torch.log(freq)/torch.log(torch.tensor([2.])).to(freq.device))
    return ent.item()


def compute_block_entropy(map_, poolers):
    with torch.no_grad():
        f = [l(map_.unsqueeze(0).unsqueeze(0).cuda()).reshape(-1) for l in poolers]
    ents = [entropy_tensor(l) for l in f]
    return ents

def most(feats, dims, scales, init_image_size, k_patches=100,
          dbscan_eps=2, return_mask=False, ks=[1,2,3,4,5]):
    """
    Implementation of MOST method.
    Inputs
        feats: the pixel/patche features of an image
        dims: dimension of the map from which the features are used
        scales: from image to map scale
        init_image_size: size of the image
        k_patches: number of k patches retrieved that are compared to the seed at seed expansion
        dbscan_eps: threshold for clustering
        return_mask: Flag to return mask for saliency detection
        ks: Pooling filter sizes
    """
    if scales[0] != scales[1]:
        raise Exception('Scales values should be the same', scales)
    # Get number of features
    A = (feats @ feats.transpose(1, 2)).squeeze()
    og_ks = 1

    seeds = []

    for i in range(A.shape[0]):
        # Get each map
        map_ = A[i].clone()
        map_[map_ <= 0] = 0
        map_ = map_.reshape(dims[0], dims[1])
        # Get thresholds for entropy
        ks = list(map(int, ks))
        ks = list(set([min(k, min(dims)) for k in ks]))
        ks.sort()
        pool_dims = [[dims[0] - k+1, dims[1]-k+1] for k in ks]
        feat_pool = [torch.nn.AdaptiveAvgPool2d(k) for k in pool_dims]
        thresh = [1+(np.log(d[0])/np.log(2)) for d in pool_dims]
        # compute entropy at each resolution
        ents = compute_block_entropy(map_, feat_pool)
        # Check if map contain any object
        pass_ = [l < t for l, t in zip(ents, thresh)]
        # If atleast 50% of the maps agree there is an object, we pick it
        if sum(pass_) >= 0.5 * len(pass_):
            seeds.append(i)
    seeds.sort()
    # If there are no seeds, then there are no objects
    if not seeds:
        return [], A, seeds, []

    # Since we are using manhattan distance, any of the eight neighbors are
    # considered neighbors
    dbscan = DBSCAN(eps=dbscan_eps, metric='cityblock', min_samples=1)
    # We use min_samples 1 to allow a single seed to be its own cluster
    # Unravel linear index to coordinates to cluster them
    seed_coords = np.stack([np.unravel_index(l, dims) for l in seeds])
    assign = dbscan.fit(seed_coords)
    seed_labels = assign.labels_

    A_clone = A.clone()
    A_clone.fill_diagonal_(0)
    A_clone[A_clone < 0] = 0

    preds = []
    masks = []
    req_feats = []
    cent = -torch.sum(A_clone > 0, dim=1).type(torch.float32)

    max_cluster_id = Counter(seed_labels).most_common(1)[0][0]
    pick = 0
    # For each cluster, get all seeds belonging to cluster and construct a box
    for ii, cl in enumerate(np.unique(seed_labels)):
        # First collect all seeds that belong to this cluster
        similars = [seeds[i] for i in np.where(seed_labels == cl)[0]]
        # Find the seed with the maximum outgoing degree
        seed_value = [cent[l] for l in similars]
        pot_seed = max(seed_value)
        # Find all pixels that have similarity with the seed with highest
        # degree
        seed = torch.tensor(similars[seed_value.index(pot_seed)])
        similars = [l for l in similars if A[seed, torch.tensor([l])] > 0]
        # Find the mask 
        M = torch.sum(A[similars, :], dim=0)
        # Detect the box using the mask
        pred, small_pred = detect_box(
            M, seed, dims, scales=scales, initial_im_size=init_image_size[1:])

        if pred:
            boxes = np.stack(pred)
            areas = (boxes[:,3] - boxes[:,1]) * (boxes[:,2] - boxes[:,0])
            widths = boxes[:,2] - boxes[:,0]
            heights = (boxes[:,3] - boxes[:,1])
            # Remove trivial boxes
            # Boxes are kept if they have a height/width greater than 16 pixels
            keep_hw = np.bitwise_and(widths > 16, heights > 16)
            # Remove boxes if they cover the full image
            keep_ar = np.bitwise_and(areas > 256, areas <
                                     0.9*np.prod(init_image_size[1:]))
            keep = np.where(np.bitwise_and(keep_hw, keep_ar))[0]

            pred = [pred[l] for l in keep]

            if not return_mask:
                # If not returning a mask, then just stack them
                preds+= [torch.tensor(l) for l in pred]
            elif pred:
                # If output is a mask, then postprocess and  upsample the mask
                # to the image resolution
                if cl == max_cluster_id:
                    pick = ii
                full_map = refine_mask(M, seed, dims)
                full_map = torch.nn.functional.interpolate(full_map.unsqueeze(0).unsqueeze(0),
                                                scale_factor=scales[0],
                                                mode='nearest')
                full_map = full_map.squeeze(0).squeeze(0)
                preds.append(full_map.cpu().numpy())

        else:
            # If there is no box, then move on
            continue

    others = []
    if preds:
        if not return_mask:
            preds_tensor = torch.stack(preds)
            preds = [l.numpy() for l in preds]
            others = []
        else:
            others = [l for i, l in enumerate(preds)]
            if pick < len(preds):
                preds = [preds[pick]]
            else:
                preds = [preds[0]]
    return preds, A, seeds, others



def refine_mask(A, seed, dims):
    w_featmap, h_featmap = dims

    correl = A.reshape(w_featmap, h_featmap).float()

    # Compute connected components
    labeled_array, num_features = scipy.ndimage.label(correl.cpu().numpy() > 0.0)

    # Find connected component corresponding to the initial seed
    cc = labeled_array[np.unravel_index(seed.cpu().numpy(), (w_featmap, h_featmap))]

    if cc == 0:
        return [], []
    mask = A.reshape(w_featmap, h_featmap).cpu().numpy()
    mask[labeled_array != cc] = -1

    return torch.tensor(mask)


def detect_box(A, seed, dims, initial_im_size=None, scales=None):
    """
    Extract a box corresponding to the seed patch. Among connected components extract from the affinity matrix, select the one corresponding to the seed patch.
    """
    w_featmap, h_featmap = dims

    correl = A.reshape(w_featmap, h_featmap).float()

    # Compute connected components
    labeled_array, num_features = scipy.ndimage.label(correl.cpu().numpy() > 0.0)

    # Find connected component corresponding to the initial seed
    cc = labeled_array[np.unravel_index(seed.cpu().numpy(), (w_featmap, h_featmap))]

    # Should not happen with LOST
    if cc == 0:
        # raise ValueError("The seed is in the background component.")
        return [], []

    # Find box
    mask = np.where(labeled_array == cc)
    # Add +1 because excluded max
    ymin, ymax = min(mask[0]), max(mask[0]) + 1
    xmin, xmax = min(mask[1]), max(mask[1]) + 1

    # Rescale to image size
    r_xmin, r_xmax = scales[1] * xmin, scales[1] * xmax
    r_ymin, r_ymax = scales[0] * ymin, scales[0] * ymax

    pred = [r_xmin, r_ymin, r_xmax, r_ymax]

    # Check not out of image size (used when padding)
    if initial_im_size:
        pred[2] = min(pred[2], initial_im_size[1])
        pred[3] = min(pred[3], initial_im_size[0])

    # Coordinate predictions for the feature space
    # Axis different then in image space
    pred_feats = [ymin, xmin, ymax, xmax]

    return [pred], [pred_feats]



