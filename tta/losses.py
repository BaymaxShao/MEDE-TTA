"""Loss functions adapted from ProxyTTA and Zero-Shot Depth Completion."""

from __future__ import absolute_import, division, print_function

import torch
import torch.nn.functional as F


def gradient_yx(tensor):
    dy = tensor[:, :, 1:, :] - tensor[:, :, :-1, :]
    dx = tensor[:, :, :, 1:] - tensor[:, :, :, :-1]
    return dy, dx


def sparse_depth_consistency_loss(pred_depth, sparse_depth, validity_map):
    """L1 consistency at sparse depth locations (ProxyTTA)."""
    if validity_map.sum() < 1:
        return pred_depth.sum() * 0.0
    delta = torch.abs(pred_depth - sparse_depth)
    loss = torch.sum(validity_map * delta, dim=[1, 2, 3])
    return torch.mean(loss / (torch.sum(validity_map, dim=[1, 2, 3]) + 1e-7))


def smoothness_loss(pred_depth, image):
    """Edge-aware smoothness on depth (ProxyTTA)."""
    pred_dy, pred_dx = gradient_yx(pred_depth)
    image_dy, image_dx = gradient_yx(image)
    weights_x = torch.exp(-torch.mean(torch.abs(image_dx), dim=1, keepdim=True))
    weights_y = torch.exp(-torch.mean(torch.abs(image_dy), dim=1, keepdim=True))
    smoothness_x = torch.mean(weights_x * torch.abs(pred_dx))
    smoothness_y = torch.mean(weights_y * torch.abs(pred_dy))
    return smoothness_x + smoothness_y


def local_smoothness_loss(structure_guidance, pred_depth):
    """Local smoothness w.r.t. structure guidance (Zero-Shot Alignment)."""
    guidance_dy, guidance_dx = gradient_yx(structure_guidance)
    pred_dy, pred_dx = gradient_yx(pred_depth)
    weights_x = torch.exp(-torch.mean(torch.abs(guidance_dx), dim=1, keepdim=True))
    weights_y = torch.exp(-torch.mean(torch.abs(guidance_dy), dim=1, keepdim=True))
    smoothness_x = torch.mean(weights_x * torch.abs(pred_dx))
    smoothness_y = torch.mean(weights_y * torch.abs(pred_dy))
    return smoothness_x + smoothness_y


def _gaussian_window(window_size, channel, device, dtype):
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * 0.5 ** 2))
    g = g / g.sum()
    window = g.unsqueeze(1) @ g.unsqueeze(0)
    window = window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim_loss(pred, target, window_size=11):
    """SSIM-based loss (1 - SSIM), adapted from Zero-Shot Depth Completion."""
    channel = pred.size(1)
    window = _gaussian_window(window_size, channel, pred.device, pred.dtype)
    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=channel) - mu1_mu2
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-7)
    return 1.0 - ssim_map.mean()


def disparity_teacher_loss(student_disp, teacher_disp):
    """CoTTA-style consistency loss between student and teacher disparity."""
    return F.mse_loss(student_disp, teacher_disp.detach())


def disparity_teacher_si_loss(
    student_disp,
    teacher_disp,
    min_depth=1e-3,
    max_depth=150.0,
    variance_focus=0.85,
    scale=10.0,
):
    """Scale-invariant teacher consistency (converts disparity to depth first)."""
    from utils.layers import disp_to_depth
    _, student_depth = disp_to_depth(student_disp, min_depth, max_depth)
    _, teacher_depth = disp_to_depth(teacher_disp.detach(), min_depth, max_depth)
    return scale_invariant_depth_loss(
        student_depth, teacher_depth, min_depth, variance_focus, scale)


def disparity_soft_entropy(disp_i, disp_j):
    """Cross-view soft consistency (ReM), treating spatial locations as a distribution."""
    pi = F.log_softmax(disp_i.flatten(1), dim=1)
    pj = F.softmax(disp_j.flatten(1), dim=1)
    return -(pj.detach() * pi).sum(dim=1)


def disparity_entropy(disp):
    """Spatial entropy of a disparity map (ReM ranked entropy term)."""
    p = F.softmax(disp.flatten(1), dim=1)
    return -(p * p.log()).sum(dim=1)


def scale_invariant_depth_loss(depth, pseudo_depth, min_depth=1e-3, variance_focus=0.85, scale=10.0):
    """Scale-invariant log loss from Ada-Depth (ICRA 2023)."""
    mask = (pseudo_depth > min_depth) & (depth > min_depth) & torch.isfinite(depth) & torch.isfinite(pseudo_depth)
    if mask.sum() < 1:
        return depth.sum() * 0.0
    d = torch.log(depth[mask].clamp(min=min_depth)) - torch.log(pseudo_depth[mask].clamp(min=min_depth))
    variance = (d ** 2).mean() - variance_focus * (d.mean() ** 2)
    variance = torch.clamp(variance, min=1e-7)
    return torch.sqrt(variance) * scale
