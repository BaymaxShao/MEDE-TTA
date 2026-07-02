from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np
from tqdm import tqdm
import time

import torch
from torch.utils.data import DataLoader
import scipy.stats as st

from utils.layers import disp_to_depth
from utils.utils import compute_errors
from options import MonodepthOptions
import datasets
import models.endodac as endodac
from tta import build_tta_adapter

cv2.setNumThreads(0)


def load_depther(opt):
    depther_path = os.path.join(opt.load_weights_folder, "depth_model.pth")
    depther_dict = torch.load(depther_path, map_location="cpu")
    depther = endodac.endodac(
        backbone_size=opt.backbone_size,
        r=opt.lora_rank,
        lora_type=opt.lora_type,
        num_experts=opt.num_experts,
        image_shape=(224, 280),
        pretrained_path=opt.pretrained_path,
        residual_block_indexes=opt.residual_block_indexes,
        include_cls_token=opt.include_cls_token,
    )
    model_dict = depther.state_dict()
    depther.load_state_dict({k: v for k, v in depther_dict.items() if k in model_dict})
    depther.cuda()
    return depther


def build_dataloader(opt):
    frame_ids = [0]
    if opt.eval_split == "hamlyn":
        dataset = datasets.HamlynDataset(
            opt.data_path, opt.height, opt.width, frame_ids, 4, is_train=False)
        max_depth = 150.0
    elif opt.eval_split == "c3vd":
        dataset = datasets.C3VDDataset(
            opt.data_path, opt.height, opt.width, frame_ids, 4, is_train=False,
            sequences=opt.c3vd_sequences)
        max_depth = 100.0
    else:
        raise ValueError("Unsupported eval split: {} (use hamlyn or c3vd)".format(opt.eval_split))

    dataloader = DataLoader(
        dataset, 1, shuffle=False, num_workers=opt.num_workers,
        pin_memory=True, drop_last=False)
    return dataloader, max_depth


def get_gt_depth(data):
    gt = data["depth_gt"]
    if torch.is_tensor(gt):
        return gt.squeeze().cpu().numpy().astype(np.float32)
    return np.squeeze(gt).astype(np.float32)


def evaluate(opt):
    min_depth = 1e-3

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)
    assert os.path.isdir(opt.load_weights_folder), \
        "Cannot find a folder at {}".format(opt.load_weights_folder)

    print("-> Loading weights from {}".format(opt.load_weights_folder))
    print("-> TTA method: {}".format(opt.tta_method))

    method = opt.tta_method.lower()
    if method in ("none", "source"):
        depther = load_depther(opt)
        adapter = None
    else:
        depther = load_depther(opt)
        adapter = build_tta_adapter(opt.tta_method, depther, opt)

    dataloader, max_depth = build_dataloader(opt)

    inference_times = []
    errors = []
    ratios = []

    print("-> Computing predictions with size {}x{}".format(opt.width, opt.height))

    for i, data in tqdm(enumerate(dataloader)):
        input_color = data[("color", 0, 0)].cuda()
        gt_depth = get_gt_depth(data)

        time_start = time.time()
        if adapter is None:
            depther.eval()
            with torch.no_grad():
                output = depther(input_color)
        else:
            output = adapter.adapt_and_predict(input_color)
        inference_time = time.time() - time_start
        inference_times.append(inference_time)

        pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
        pred_disp = pred_disp.cpu()[:, 0].numpy()[0]

        gt_height, gt_width = gt_depth.shape[:2]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1.0 / pred_disp
        mask = np.logical_and(gt_depth > min_depth, gt_depth < max_depth)

        pred_depth = pred_depth[mask]
        gt_depth_masked = gt_depth[mask]

        if pred_depth.size == 0 or not np.isfinite(pred_depth).any():
            continue

        pred_depth *= opt.pred_depth_scale_factor
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth_masked) / np.median(pred_depth)
            if not np.isnan(ratio).all():
                ratios.append(ratio)
            pred_depth *= ratio
        pred_depth[pred_depth < min_depth] = min_depth
        pred_depth[pred_depth > max_depth] = max_depth
        error = compute_errors(gt_depth_masked, pred_depth)
        if not np.isnan(error).all():
            errors.append(error)

    if not opt.disable_median_scaling and ratios:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    if len(errors) == 0:
        print("\n-> No valid depth predictions were evaluated.")
        print("-> Done!")
        return

    errors = np.array(errors)
    mean_errors = np.mean(errors, axis=0)
    cls = []
    for j in range(len(mean_errors)):
        cl = st.t.interval(alpha=0.95, df=len(errors) - 1, loc=mean_errors[j], scale=st.sem(errors[:, j]))
        cls.append(cl[0])
        cls.append(cl[1])
    cls = np.array(cls)

    print("\n       " + ("{:>11}      | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print("mean:" + ("&{: 12.3f}      " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("cls: " + ("& [{: 6.3f}, {: 6.3f}] " * 7).format(*cls.tolist()) + "\\\\")
    print("average inference time: {:0.1f} ms".format(np.mean(np.array(inference_times)) * 1000))
    print("\n-> Done!")


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
