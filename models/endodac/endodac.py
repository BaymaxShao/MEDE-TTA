import os
import torch
import torch.nn as nn
import models.backbones as backbones
from models.backbones.mylora import Linear as LoraLinear
from models.backbones.mylora import DVLinear as DVLinear
from models.backbones.mylora import MoDVLinear
from .layers import HeadDepth
from .layers import mark_only_part_as_trainable,_make_scratch, _make_fusion_block

class DPTHead(nn.Module):
    def __init__(self, in_channels, features=128, use_bn=False, out_channels=[96, 192, 384, 768], use_clstoken=False):
        super(DPTHead, self).__init__()

        self.use_clstoken = use_clstoken
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.conv_depth_1 = HeadDepth(features)
        self.conv_depth_2 = HeadDepth(features)
        self.conv_depth_3 = HeadDepth(features)
        self.conv_depth_4 = HeadDepth(features)
        
        self.sigmoid = nn.Sigmoid()
    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        outputs = {}
        outputs[("disp", 3)] = self.sigmoid(self.conv_depth_4(path_4))
        outputs[("disp", 2)] = self.sigmoid(self.conv_depth_3(path_3))
        outputs[("disp", 1)] = self.sigmoid(self.conv_depth_2(path_2))
        outputs[("disp", 0)] = self.sigmoid(self.conv_depth_1(path_1))

        return outputs
    
class endodac(nn.Module):
    """Applies low-rank adaptation to a ViT model's image encoder.

    Args:
        backbone_size: size of pretrained Dinov2 choice from: "small", "base", "large", "giant"
        r: rank of LoRA
        image_shape: input image shape, h,w need to be multiplier of 14, default:(224,280)
        lora_layer: which layer we apply LoRA.
    """

    def __init__(self, 
                 backbone_size = "base", 
                 r=4, 
                 image_shape=(224,280), 
                 lora_type="lora",
                 num_experts=4,
                 pretrained_path=None,
                 residual_block_indexes=[],
                 include_cls_token=True,
                 use_cls_token=False,
                 use_bn=False):
        super(endodac, self).__init__()

        assert r > 0
        self.r = r
        self.backbone_size = backbone_size
        vit_kwargs = dict(
            residual_block_indexes=residual_block_indexes,
            include_cls_token=include_cls_token,
            input_size=image_shape,
        )
        vit_factory = {
            "small": backbones.vits.vit_small,
            "base": backbones.vits.vit_base,
            "large": backbones.vits.vit_large,
        }
        self.backbone_archs = {
            "small": "vits14",
            "base": "vitb14",
            "large": "vitl14",
        }
        self.intermediate_layers = {
            "small": [2, 5, 8, 11],
            "base": [2, 5, 8, 11],
            "large": [2, 5, 8, 11],
        }
        self.embedding_dims = {
            "small": 384,
            "base": 768,
            "large": 1024,
        }
        self.depth_head_features = {
            "small": 64,
            "base": 128,
            "large": 256,
        }
        self.depth_head_out_channels = {
            "small": [48, 96, 192, 384],
            "base": [96, 192, 384, 768],
            "large": [256, 512, 1024, 1024],
        }
        self.backbone_arch = self.backbone_archs[self.backbone_size]
        self.embedding_dim = self.embedding_dims[self.backbone_size]
        self.depth_head_feature = self.depth_head_features[self.backbone_size]
        self.depth_head_out_channel = self.depth_head_out_channels[self.backbone_size]
        encoder = vit_factory[self.backbone_size](**vit_kwargs)

        self.image_shape = image_shape
        
        if lora_type != "none":
            for t_layer_i, blk in enumerate(encoder.blocks):
                mlp_in_features = blk.mlp.fc1.in_features
                mlp_hidden_features = blk.mlp.fc1.out_features
                mlp_out_features = blk.mlp.fc2.out_features
                if lora_type == "dvlora":
                    blk.mlp.fc1 = DVLinear(mlp_in_features, mlp_hidden_features, r=self.r, lora_alpha=self.r)
                    blk.mlp.fc2 = DVLinear(mlp_hidden_features, mlp_out_features, r=self.r, lora_alpha=self.r)
                elif lora_type == "modvlora":
                    blk.mlp.fc1 = MoDVLinear(mlp_in_features, mlp_hidden_features, r=self.r,
                                             lora_alpha=self.r, num_experts=num_experts)
                    blk.mlp.fc2 = MoDVLinear(mlp_hidden_features, mlp_out_features, r=self.r,
                                             lora_alpha=self.r, num_experts=num_experts)
                elif lora_type == "lora":
                    blk.mlp.fc1 = LoraLinear(mlp_in_features, mlp_hidden_features, r=self.r)
                    blk.mlp.fc2 = LoraLinear(mlp_hidden_features, mlp_out_features, r=self.r)
            
        self.encoder = encoder
        self.depth_head = DPTHead(self.embedding_dim, self.depth_head_feature, use_bn, out_channels=self.depth_head_out_channel, use_clstoken=use_cls_token)
        
        if pretrained_path is not None:
            pretrained_path = os.path.join(pretrained_path, "depth_anything_{}.pth".format(self.backbone_arch))
            pretrained_dict = torch.load(pretrained_path, map_location="cpu")
            remapped_dict = {}
            for key, value in pretrained_dict.items():
                if key.startswith("pretrained."):
                    key = key.replace("pretrained.", "encoder.", 1)
                remapped_dict[key] = value
            missing, unexpected = self.load_state_dict(remapped_dict, strict=False)
            print("load pretrained weight from {}".format(pretrained_path))
            print("  loaded {} / {} keys (missing: {}, unexpected: {})\n".format(
                len(remapped_dict) - len(unexpected), len(remapped_dict),
                len(missing), len(unexpected)))

        mark_only_part_as_trainable(self.encoder)
        mark_only_part_as_trainable(self.depth_head)
    def _encode(self, pixel_values, patch_mask=None):
        pixel_values = torch.nn.functional.interpolate(
            pixel_values, size=self.image_shape, mode="bilinear", align_corners=True)
        h, w = pixel_values.shape[-2:]
        layer_ids = 4
        features = self.encoder.get_intermediate_layers(
            pixel_values, layer_ids, return_class_token=True, patch_mask=patch_mask)
        patch_h, patch_w = h // 14, w // 14
        return pixel_values, features, patch_h, patch_w

    def get_patch_token_means(self, pixel_values, patch_mask=None):
        """Per-patch mean of normalized token features for DaM uncertainty."""
        pixel_values = torch.nn.functional.interpolate(
            pixel_values, size=self.image_shape, mode="bilinear", align_corners=True)
        tokens = self.encoder.get_last_patch_tokens(pixel_values, patch_mask=patch_mask)
        return tokens.mean(dim=-1)

    def get_cls_patch_attention(self, pixel_values):
        """CLS-to-patch attention scores from the ViT encoder (for ReM TTA)."""
        pixel_values = torch.nn.functional.interpolate(
            pixel_values, size=self.image_shape, mode="bilinear", align_corners=True)
        return self.encoder.get_cls_patch_attention(pixel_values)

    def forward(self, pixel_values, patch_mask=None, return_patch_tokens=False):
        pixel_values, features, patch_h, patch_w = self._encode(pixel_values, patch_mask=patch_mask)
        disp = self.depth_head(features, patch_h, patch_w)
        if return_patch_tokens:
            return disp, features[-1][0]
        return disp
