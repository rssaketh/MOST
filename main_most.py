import os
import argparse
import random
import pickle
import pdb

import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from PIL import Image

from networks import get_model
from datasets import ImageDataset, Dataset, bbox_iou
from visualizations import visualize_predictions, visualize_map
from object_discovery import most
from saliency_utils import saliency_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Arguments for MOST")
    parser.add_argument(
        "--arch",
        default="vit_small",
        type=str,
        choices=[
            "vit_tiny",
            "vit_small",
            "vit_base",
            "ibot_base",
        ],
        help="Model architecture.",
    )
    parser.add_argument(
        "--patch_size", default=16, type=int, help="Patch resolution of the model."
    )

    # Use a dataset
    parser.add_argument(
        "--dataset",
        default="VOC07",
        type=str,
        choices=[None, "VOC07", "VOC12", "COCO20k", "COCO", "COCOminival",
                 "ECSSD", "DUTS", "DUT-OMRON"],
        help="Dataset name.",
    )
    parser.add_argument(
        "--set",
        default="train",
        type=str,
        choices=["val", "train", "trainval", "test"],
        help="Path of the image to load.",
    )
    # Or use a single image
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="If want to apply only on one image, give file path.",
    )

    # Folder used to output visualizations and 
    parser.add_argument(
        "--output_dir", type=str, default="outputs", help="Output directory to store predictions and visualizations."
    )

    # Evaluation setup
    parser.add_argument("--no_hard", action="store_true", help="Only used in the case of the VOC_all setup (see the paper).")
    parser.add_argument("--no_evaluation", action="store_true", help="Compute the evaluation.")
    parser.add_argument("--save_predictions", default=True, type=bool, help="Save predicted bouding boxes.")

    # Visualization
    parser.add_argument(
        "--visualize",
        type=str,
        choices=["pred", None],
        default=None,
        help="Select the different type of visualizations.",
    )

    # MOST parameters
    parser.add_argument(
        "--which_features",
        type=str,
        default="k",
        choices=["k", "q", "v"],
        help="Which features to use",
    )
    parser.add_argument(
        "--k_patches",
        type=int,
        default=100,
        help="Number of patches with the lowest degree considered."
    )
    parser.add_argument(
        "--dbscan_eps",
        type=int,
        default=2,
        help="DBSCAN min distance per sample"
    )
    parser.add_argument(
        "--ks",
        nargs='+',
        default=[1,2,3,4,5]
    )


    args = parser.parse_args()

    if args.image_path is not None:
        args.save_predictions = False
        args.no_evaluation = True
        args.dataset = None

    # -------------------------------------------------------------------------------------------------------
    # Dataset

    # If an image_path is given, apply the method only to the image
    if args.image_path is not None:
        dataset = ImageDataset(args.image_path)
    else:
        dataset = Dataset(args.dataset, args.set, args.no_hard)

    # -------------------------------------------------------------------------------------------------------
    # Model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = get_model(args.arch, args.patch_size, device)
    # -------------------------------------------------------------------------------------------------------
    # Directories
    if args.image_path is None:
        args.output_dir = os.path.join(args.output_dir, dataset.name)
    os.makedirs(args.output_dir, exist_ok=True)

    # Naming
    # Experiment with MOST
    exp_name = f"MOST-{args.arch}"
    if "vit" in args.arch:
        exp_name += f"{args.patch_size}_{args.which_features}"

    print(f"Running MOST on the dataset {dataset.name} (exp: {exp_name})")

    # Visualization 
    if args.visualize:
        vis_folder = f"{args.output_dir}/visualizations/{exp_name}"
        os.makedirs(vis_folder, exist_ok=True)

    # -------------------------------------------------------------------------------------------------------
    # Loop over images
    preds_dict = {}
    cnt = 0
    corloc = np.zeros(len(dataset.dataloader))
    fbmax = np.zeros(len(dataset.dataloader))
    iou_arr = np.zeros(len(dataset.dataloader))
    acc = np.zeros(len(dataset.dataloader))
    pbar = tqdm(dataset.dataloader)
    for im_id, inp in enumerate(pbar):
        # ------------ IMAGE PROCESSING -------------------------------------------
        img = inp[0]
        init_image_size = img.shape
        if (torch.tensor(init_image_size[1:]) > 1000).any():
            continue
        # Get the name of the image
        if not inp[1]:
            continue
        im_name = dataset.get_image_name(inp[1])

        # Pass in case of no gt boxes in the image
        if im_name is None:
            continue

        # Padding the image with zeros to fit multiple of patch-size
        size_im = (
            img.shape[0],
            int(np.ceil(img.shape[1] / args.patch_size) * args.patch_size),
            int(np.ceil(img.shape[2] / args.patch_size) * args.patch_size),
        )
        paded = torch.zeros(size_im)
        paded[:, : img.shape[1], : img.shape[2]] = img
        img = paded

        # Move to gpu
        img = img.cuda(non_blocking=True)
        # Size for transformers
        w_featmap = img.shape[-2] // args.patch_size
        h_featmap = img.shape[-1] // args.patch_size

        # ------------ GROUND-TRUTH -------------------------------------------
        if not args.no_evaluation:
            gt_bbxs, gt_cls = dataset.extract_gt(inp[1], im_name)
            if gt_bbxs is not None:
                # Discard images with no gt annotations
                # Happens only in the case of VOC07 and VOC12
                if gt_bbxs.shape[0] == 0 and args.no_hard:
                    continue

        # ------------ EXTRACT FEATURES -------------------------------------------
        with torch.no_grad():

            # ------------ FORWARD PASS -------------------------------------------
            if "vit" in args.arch or "ibot" in args.arch:
                # Store the outputs of qkv layer from the last attention layer
                feat_out = {}
                def hook_fn_forward_qkv(module, input, output):
                    feat_out["qkv"] = output
                model._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)

                # Forward pass in the model
                attentions = model.get_last_selfattention(img[None, :, :, :])

                # Scaling factor
                scales = [args.patch_size, args.patch_size]

                # Dimensions
                nb_im = attentions.shape[0]  # Batch size
                nh = attentions.shape[1]  # Number of heads
                nb_tokens = attentions.shape[2]  # Number of tokens
                # Extract the qkv features of the last attention layer
                qkv = (
                    feat_out["qkv"]
                    .reshape(nb_im, nb_tokens, 3, nh, -1 // nh)
                    .permute(2, 0, 3, 1, 4)
                )
                q, k, v = qkv[0], qkv[1], qkv[2]
                k = k.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
                q = q.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
                v = v.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
                # Modality selection
                if args.which_features == "k":
                    feats = k[:, 1:, :]
                elif args.which_features == "q":
                    feats = q[:, 1:, :]
                elif args.which_features == "v":
                    feats = v[:, 1:, :]
            else:
                raise ValueError("Unknown model.")

        # ------------ Apply MOST -------------------------------------------
        pred, A, seed, others = most(
                                    feats,
                                    [w_featmap, h_featmap],
                                    scales,
                                    init_image_size,
                                    k_patches=args.k_patches,
                                    dbscan_eps=args.dbscan_eps,
                                    return_mask = (args.dataset in ['ECSSD', "DUTS",
                                                                    "DUT-OMRON"]),
                                    ks=args.ks
        )
        # ------------ Visualizations -------------------------------------------
        if args.visualize == "pred":
            image = dataset.load_image(im_name)
            if args.dataset in ['ECSSD', 'DUTS', 'DUT-OMRON']:
                visualize_map(
                            image,
                            pred,
                            gt_bbxs,
                            vis_folder,
                            im_name,
                            others
                )
            else:
                visualize_predictions(
                                    image,
                                    pred,
                                    seed,
                                    scales,
                                    [w_featmap, h_featmap],
                                    vis_folder,
                                    im_name
                )
        # Save the prediction
        if args.dataset in ['ECSSD', 'DUTS', 'DUT-OMRON']:
            if pred:
                preds_dict[im_name] = np.max(np.stack(pred, axis=0),axis=0)
            else:
                preds_dict[im_name] = np.asarray([])
        else:
            preds_dict[im_name] = pred

        # Evaluation
        if args.no_evaluation:
            continue

        # Compare prediction to GT boxes
        if args.dataset in ['ECSSD', 'DUTS', 'DUT-OMRON']:
            # For saliency detection, compute different metrics
            fbmax_, iou_, acc_ = saliency_metrics(pred, gt_bbxs)
            fbmax[im_id] = fbmax_
            iou_arr[im_id] = iou_
            acc[im_id] = acc_
            cnt += 1
            # if cnt % 50 == 0:
            cur_fbmax = np.sum(fbmax)/cnt
            cur_iou = np.sum(iou_arr)/cnt
            cur_acc = np.sum(acc)/cnt
            pbar.set_description(f"F1: {cur_fbmax}, IoU: {cur_iou}, Acc: {cur_acc}")

        else:
            if isinstance(pred, list):
                ious = []
                for p in pred:
                    try:
                        iou = bbox_iou(torch.from_numpy(p), torch.from_numpy(gt_bbxs))
                    except:
                        pdb.set_trace()
                    ious.append(iou)
                if ious:
                    ious = torch.cat(ious)
                else:
                    ious = torch.tensor([])
            else:
                ious = bbox_iou(torch.from_numpy(pred), torch.from_numpy(gt_bbxs))
            if ious.shape[0] > 0:
                if torch.any(ious >= 0.5):
                    corloc[im_id] = 1

            cnt += 1
            pbar.set_description(f"Found {int(np.sum(corloc))}/{cnt}")


    # Save predicted bounding boxes
    if args.save_predictions:
        folder = f"{args.output_dir}/{exp_name}"
        os.makedirs(folder, exist_ok=True)
        filename = os.path.join(folder, "preds.pkl")
        with open(filename, "wb") as f:
            pickle.dump(preds_dict, f)
        print("Predictions saved at %s" % filename)

    # Evaluate
    if not args.no_evaluation:
        result_file = os.path.join(folder, 'results.txt')
        if args.dataset in ['ECSSD', 'DUTS', 'DUT-OMRON']:
            print(f"F1: {int(np.sum(fbmax))}/{cnt}, IoU: {int(np.sum(iou_arr))}/{cnt}, Acc: {int(np.sum(acc))}/{cnt}")
            res_str = 'fbmax,%.1f,iou,%.1f,acc,%.1f,,\n'%(100*np.sum(fbmax)/cnt,
                                                100*np.sum(iou_arr)/cnt,
                                                100*np.sum(acc)/cnt)
        else:
            print(f"corloc: {100*np.sum(corloc)/cnt:.2f} ({int(np.sum(corloc))}/{cnt})")
            res_str = 'corloc,%.1f,,\n'%(100*np.sum(corloc)/cnt)
        with open(result_file, 'w') as f:
            f.write(res_str)
        print('File saved at %s'%result_file)

