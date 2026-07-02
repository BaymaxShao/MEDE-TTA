from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(__file__)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


class MonodepthOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="MEDE-TTA: cross-scenario endoscopic depth estimation")

        # paths
        self.parser.add_argument("--data_path", type=str, required=True)
        self.parser.add_argument("--load_weights_folder", type=str, required=True)
        self.parser.add_argument("--pretrained_path", type=str,
                                 default=os.path.join(file_dir, "pretrained_model"))

        # model
        self.parser.add_argument("--backbone_size", type=str, default="base",
                                 choices=["small", "base", "large"])
        self.parser.add_argument("--lora_type", type=str, default="dvlora",
                                 choices=["lora", "dvlora", "modvlora", "none"])
        self.parser.add_argument("--lora_rank", type=int, default=4)
        self.parser.add_argument("--num_experts", type=int, default=4)
        self.parser.add_argument("--residual_block_indexes", nargs="*", type=int,
                                 default=[2, 5, 8, 11])
        self.parser.add_argument("--include_cls_token", type=str2bool, default=True)

        # eval
        self.parser.add_argument("--eval_split", type=str, default="hamlyn",
                                 choices=["hamlyn", "c3vd"])
        self.parser.add_argument("--c3vd_sequences", nargs="+", type=str,
                                 default=[
                                     "cecum_t4_a", "cecum_t4_b",
                                     "desc_t4_a_down", "desc_t4_a_up",
                                     "sigmoid_t3_a", "sigmoid_t3_b",
                                     "trans_t4_a", "trans_t4_b",
                                 ],
                                 help="C3VD sequences to evaluate")
        self.parser.add_argument("--eval_mono", action="store_true")
        self.parser.add_argument("--eval_stereo", action="store_true")
        self.parser.add_argument("--height", type=int, default=256)
        self.parser.add_argument("--width", type=int, default=320)
        self.parser.add_argument("--min_depth", type=float, default=1e-3)
        self.parser.add_argument("--max_depth", type=float, default=150.0)
        self.parser.add_argument("--disable_median_scaling", action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor", type=float, default=1.0)
        self.parser.add_argument("--num_workers", type=int, default=4)

        # MEDE TTA
        self.parser.add_argument("--tta_method", type=str, default="mede",
                                 choices=["none", "source", "mede"])
        self.parser.add_argument("--tta_steps", type=int, default=1)
        self.parser.add_argument("--tta_lr", type=float, default=1e-3)
        self.parser.add_argument("--tta_lr_uv", type=float, default=1e-3)
        self.parser.add_argument("--tta_lr_base", type=float, default=1e-4)
        self.parser.add_argument("--tta_mt", type=float, default=0.999,
                                 help="EMA momentum for teacher (non U/V params)")
        self.parser.add_argument("--tta_ema_uv", "--tta_ema_vida", type=float, default=0.99,
                                 help="EMA momentum for lora_U/lora_V")
        self.parser.add_argument("--tta_rst", type=float, default=0.01,
                                 help="stochastic reset probability")
        self.parser.add_argument("--tta_ap", type=float, default=1.0,
                                 help="anchor confidence threshold for Aug-Teacher")
        self.parser.add_argument("--tta_num_aug", type=int, default=None,
                                 help="augmentations for Aug-Teacher (split default if unset)")
        self.parser.add_argument("--tta_episodic", type=str2bool, default=False)
        self.parser.add_argument("--tta_unc_thr", type=float, default=0.2)
        self.parser.add_argument("--tta_unc_scale", type=float, default=None,
                                 help="HKA uncertainty scale (split default if unset; 0 disables HKA)")
        self.parser.add_argument("--tta_variance_focus", type=float, default=0.85)
        self.parser.add_argument("--tta_mede_si_loss", type=str2bool, default=True,
                                 help="use scale-invariant loss (False = MSE on disparity)")

    _SPLIT_TTA_DEFAULTS = {
        "hamlyn": {"tta_unc_scale": 0.0, "tta_num_aug": 2},
        "c3vd": {"tta_unc_scale": 0.1, "tta_num_aug": 6},
    }

    def parse(self):
        opt = self.parser.parse_args()
        split_defaults = self._SPLIT_TTA_DEFAULTS.get(opt.eval_split, {})
        if opt.tta_unc_scale is None:
            opt.tta_unc_scale = split_defaults.get("tta_unc_scale", 0.0)
        if opt.tta_num_aug is None:
            opt.tta_num_aug = split_defaults.get("tta_num_aug", 2)
        print("-> {} | MEDE TTA defaults applied:".format(opt.eval_split))
        print("   steps={} lr={} lr_uv={} lr_base={}".format(
            opt.tta_steps, opt.tta_lr, opt.tta_lr_uv, opt.tta_lr_base))
        print("   mt={} rst={} ap={} ema_uv={}".format(
            opt.tta_mt, opt.tta_rst, opt.tta_ap, opt.tta_ema_uv))
        print("   unc_thr={} unc_scale={} num_aug={} variance_focus={}".format(
            opt.tta_unc_thr, opt.tta_unc_scale, opt.tta_num_aug, opt.tta_variance_focus))
        return opt
