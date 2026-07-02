"""Shared utilities for MEDE test-time adaptation."""

from __future__ import absolute_import, division, print_function

from copy import deepcopy

import torch
import torch.nn as nn


ADAPT_PARAM_KEYS = ("lora_A", "lora_B", "lora_U", "lora_V", "conv_depth", "residual_")


def collect_mede_param_groups(model, lr_uv, lr_base):
    """Split adapt params into lora_U/V vs other branches."""
    uv_params = []
    base_params = []
    for nm, m in model.named_modules():
        for np_name, p in m.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            full_name = f"{nm}.{np_name}" if nm else np_name
            if not any(key in full_name for key in ADAPT_PARAM_KEYS):
                continue
            if "lora_U" in full_name or "lora_V" in full_name:
                uv_params.append(p)
            else:
                base_params.append(p)
    groups = []
    if uv_params:
        groups.append({"params": uv_params, "lr": lr_uv})
    if base_params:
        groups.append({"params": base_params, "lr": lr_base})
    return groups


def configure_model_for_tta(model):
    model.train()
    model.requires_grad_(False)
    for nm, m in model.named_modules():
        for np_name, p in m.named_parameters(recurse=False):
            full_name = f"{nm}.{np_name}" if nm else np_name
            if any(key in full_name for key in ADAPT_PARAM_KEYS):
                p.requires_grad_(True)
    return model


def copy_model_with_ema(model):
    model_state = deepcopy(model.state_dict())
    model_anchor = deepcopy(model)
    model_anchor.eval()
    for p in model_anchor.parameters():
        p.detach_()
    model_ema = deepcopy(model)
    model_ema.eval()
    for p in model_ema.parameters():
        p.detach_()
    return model_state, model_anchor, model_ema


def update_ema_variables_mede(ema_model, model, alpha_teacher, alpha_uv):
    for ema_param, (name, param) in zip(ema_model.parameters(), model.named_parameters()):
        alpha = alpha_uv if ("lora_U" in name or "lora_V" in name) else alpha_teacher
        ema_param.data[:] = alpha * ema_param[:].data[:] + (1 - alpha) * param[:].data[:]
    return ema_model


def set_dvlora_scales(model, scale_u, scale_v):
    for m in model.modules():
        if hasattr(m, "lora_U") and hasattr(m, "lora_V"):
            m.vida_scale_u = scale_u
            m.vida_scale_v = scale_v


def stochastic_restore(model, model_state, rst_m):
    """Reset: randomly restore adapted parameters toward source weights."""
    device = next(model.parameters()).device
    with torch.no_grad():
        for nm, m in model.named_modules():
            for np_name, p in m.named_parameters(recurse=False):
                if not p.requires_grad:
                    continue
                full_name = f"{nm}.{np_name}" if nm else np_name
                if full_name not in model_state:
                    continue
                src = model_state[full_name].to(device=device, dtype=p.dtype)
                mask = (torch.rand(p.shape, device=device) < rst_m).float()
                p.data = src * mask + p.data * (1.0 - mask)
