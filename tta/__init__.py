from __future__ import absolute_import, division, print_function

from .mede import MEDEAdapter, setup_mede


def build_tta_adapter(method, model, opt):
    method = method.lower()
    if method in ("none", "source"):
        return None
    if method == "mede":
        return setup_mede(
            model,
            lr=opt.tta_lr,
            lr_uv=opt.tta_lr_uv,
            lr_base=opt.tta_lr_base,
            steps=opt.tta_steps,
            mt_alpha=opt.tta_mt,
            mt_uv=opt.tta_ema_uv,
            rst_m=opt.tta_rst,
            unc_thr=opt.tta_unc_thr,
            num_aug=opt.tta_num_aug,
            unc_scale=opt.tta_unc_scale,
            ap=opt.tta_ap,
            min_depth=opt.min_depth,
            max_depth=opt.max_depth,
            variance_focus=opt.tta_variance_focus,
            episodic=opt.tta_episodic,
            use_si_loss=opt.tta_mede_si_loss,
        )
    raise ValueError("Unknown TTA method '{}'. Choose from: source, mede".format(method))


__all__ = ["MEDEAdapter", "setup_mede", "build_tta_adapter"]
