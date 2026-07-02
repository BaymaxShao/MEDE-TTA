"""MEDE: Monocular Endoscopic Depth Estimation via Test-Time Adaptation.

Aug-Teacher + Reset + scale-invariant consistency for continual TTA on EndoDAC.
"""

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from .base import (
    collect_mede_param_groups,
    configure_model_for_tta,
    copy_model_with_ema,
    set_dvlora_scales,
    stochastic_restore,
    update_ema_variables_mede,
)
from .losses import disparity_teacher_loss, disparity_teacher_si_loss


class _ColorJitterTTA(nn.Module):
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02):
        super().__init__()
        self.jitter = transforms.ColorJitter(brightness, contrast, saturation, hue)

    def forward(self, x):
        return self.jitter(x)


class MEDEAdapter(nn.Module):
    """MEDE wrapper: optional HKA + Aug-Teacher + SI/MSE consistency + Reset."""

    def __init__(
        self,
        model,
        optimizer,
        steps=1,
        mt_alpha=0.99,
        mt_uv=0.99,
        rst_m=0.01,
        unc_thr=0.2,
        num_aug=6,
        unc_scale=0.1,
        ap=0.92,
        min_depth=0.1,
        max_depth=150.0,
        variance_focus=0.85,
        episodic=False,
        use_si_loss=True,
    ):
        super().__init__()
        self.model = configure_model_for_tta(model)
        self.optimizer = optimizer
        self.steps = steps
        self.mt = mt_alpha
        self.mt_uv = mt_uv
        self.rst = rst_m
        self.thr = unc_thr
        self.num_aug = num_aug
        self.unc_scale = unc_scale
        self.ap = ap
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.variance_focus = variance_focus
        self.episodic = episodic
        self.use_si_loss = use_si_loss

        self.model_state, self.model_anchor, self.model_ema = copy_model_with_ema(self.model)
        self.model_ema.eval()
        self.model_anchor.eval()
        self.transform = _ColorJitterTTA()

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)
        self.model_state, self.model_anchor, self.model_ema = copy_model_with_ema(self.model)
        self.model_ema.eval()
        self.model_anchor.eval()

    def _make_aug_batch(self, image):
        return torch.cat([self.transform(image) for _ in range(self.num_aug)], dim=0)

    @torch.no_grad()
    def _estimate_uncertainty(self, aug_batch, batch_size):
        patch_means = self.model_ema.get_patch_token_means(aug_batch)
        patch_means = patch_means.view(self.num_aug, batch_size, -1)
        variance = torch.var(patch_means, dim=0)
        return variance.mean() * self.unc_scale

    def _apply_hka(self, uncertainty):
        if uncertainty >= self.thr:
            scale_u = 1.0 + uncertainty
            scale_v = 1.0 - uncertainty
        else:
            scale_u = 1.0 - uncertainty
            scale_v = 1.0 + uncertainty
        set_dvlora_scales(self.model, scale_u, scale_v)
        set_dvlora_scales(self.model_ema, scale_u, scale_v)

    @torch.no_grad()
    def _get_teacher_disp(self, image, aug_batch, batch_size):
        """Aug-Teacher: aug-averaged EMA with anchor-confidence gating."""
        anchor_disp = self.model_anchor(image)[("disp", 0)]
        anchor_conf = 1.0 - torch.std(anchor_disp, dim=[2, 3], keepdim=True)
        anchor_prob = anchor_conf.mean()

        aug_disps = self.model_ema(aug_batch)[("disp", 0)]
        aug_ema = aug_disps.view(self.num_aug, batch_size, *aug_disps.shape[1:]).mean(0)
        standard_ema = self.model_ema(image)[("disp", 0)]

        if anchor_prob < self.ap:
            return aug_ema
        return standard_ema

    @torch.enable_grad()
    def forward_and_adapt(self, image):
        self.model_ema.eval()
        self.model_anchor.eval()
        batch_size = image.shape[0]

        with torch.no_grad():
            aug_batch = self._make_aug_batch(image)
            uncertainty = self._estimate_uncertainty(aug_batch, batch_size)
            self._apply_hka(uncertainty)
            teacher_disp = self._get_teacher_disp(image, aug_batch, batch_size)

        self.optimizer.zero_grad()
        outputs = self.model(image)
        student_disp = outputs[("disp", 0)]
        if self.use_si_loss:
            loss = disparity_teacher_si_loss(
                student_disp,
                teacher_disp,
                min_depth=self.min_depth,
                max_depth=self.max_depth,
                variance_focus=self.variance_focus,
            )
        else:
            loss = disparity_teacher_loss(student_disp, teacher_disp)
        if torch.isfinite(loss):
            loss.backward()
            self.optimizer.step()

        self.model_ema = update_ema_variables_mede(
            self.model_ema, self.model, self.mt, self.mt_uv)
        stochastic_restore(self.model, self.model_state, self.rst)

        with torch.no_grad():
            return self.model(image)

    def adapt_and_predict(self, image):
        if self.episodic:
            self.reset()
        outputs = None
        for _ in range(self.steps):
            outputs = self.forward_and_adapt(image)
        self.model.eval()
        return outputs


def setup_mede(
    model,
    lr=5e-4,
    lr_uv=None,
    lr_base=None,
    steps=1,
    mt_alpha=0.999,
    mt_uv=0.995,
    rst_m=0.005,
    unc_thr=0.1,
    num_aug=6,
    unc_scale=0.1,
    ap=0.92,
    min_depth=0.1,
    max_depth=150.0,
    variance_focus=0.85,
    episodic=False,
    use_si_loss=True,
):
    configure_model_for_tta(model)
    if lr_uv is None:
        lr_uv = lr
    if lr_base is None:
        lr_base = lr * 0.1

    if not any(hasattr(m, "lora_U") for m in model.modules()):
        print("-> MEDE: model has no lora_U/lora_V; set --lora_type dvlora for HKA")

    loss_name = "scale-invariant depth" if use_si_loss else "MSE disparity"
    print("-> MEDE: consistency loss = {}".format(loss_name))

    param_groups = collect_mede_param_groups(model, lr_uv=lr_uv, lr_base=lr_base)
    optimizer = torch.optim.Adam(param_groups)

    return MEDEAdapter(
        model,
        optimizer,
        steps=steps,
        mt_alpha=mt_alpha,
        mt_uv=mt_uv,
        rst_m=rst_m,
        unc_thr=unc_thr,
        num_aug=num_aug,
        unc_scale=unc_scale,
        ap=ap,
        min_depth=min_depth,
        max_depth=max_depth,
        variance_focus=variance_focus,
        episodic=episodic,
        use_si_loss=use_si_loss,
    )
