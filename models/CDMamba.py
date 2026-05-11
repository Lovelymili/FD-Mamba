from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.convolutions import Convolution
from monai.networks.blocks.segresnet_block import ResBlock, get_conv_layer, get_upsample_layer
from monai.networks.layers.factories import Dropout
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils import UpsampleMode
from einops import rearrange
from mamba_ssm import Mamba as Mamba_ssm

from models.mamba_customer import ConvMamba, L_GF_Mamba, G_GL_Mamba




def get_dwconv_layer(
        spatial_dims: int, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1,
        bias: bool = False
):
    depth_conv = Convolution(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=in_channels,
                             strides=stride, kernel_size=kernel_size, bias=bias, conv_only=True, groups=in_channels)
    point_conv = Convolution(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=out_channels,
                             strides=stride, kernel_size=1, bias=bias, conv_only=True, groups=1)
    return torch.nn.Sequential(depth_conv, point_conv)


class SRCMLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2, conv_mode='deepwise'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.convmamba = ConvMamba(
            d_model=input_dim,  
            d_state=d_state, 
            d_conv=d_conv,  
            expand=expand,  
            bimamba_type="v2",
            conv_mode=conv_mode
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.convmamba(x_norm) + self.skip_scale * x_flat
        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        return out


def get_srcm_layer(
        spatial_dims: int, in_channels: int, out_channels: int, stride: int = 1, conv_mode: str = "deepwise"
):
    srcm_layer = SRCMLayer(input_dim=in_channels, output_dim=out_channels, conv_mode=conv_mode)
    if stride != 1:
        if spatial_dims == 2:
            return nn.Sequential(srcm_layer, nn.MaxPool2d(kernel_size=stride, stride=stride))
    return srcm_layer


class SRCMBlock(nn.Module):
    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            norm: tuple | str,
            kernel_size: int = 3,
            conv_mode: str = "deepwise",
            act: tuple | str = ("RELU", {"inplace": True}),
    ) -> None:
        super().__init__()

        if kernel_size % 2 != 1:
            raise AssertionError("kernel_size should be an odd number.")
        self.norm1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.norm2 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.act = get_act_layer(act)
        self.conv1 = get_srcm_layer(
            spatial_dims, in_channels=in_channels, out_channels=in_channels, conv_mode=conv_mode
        )
        self.conv2 = get_srcm_layer(
            spatial_dims, in_channels=in_channels, out_channels=in_channels, conv_mode=conv_mode
        )

    def forward(self, x):
        identity = x
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)
        x += identity
        return x


class FineGrainedDiffAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()


        self.diff_conv_s = nn.Sequential(
            nn.Conv2d(dim * 2, dim // 2, kernel_size=3, padding=1, groups=dim//2, bias=False),
            nn.InstanceNorm2d(dim // 2),
            nn.GELU()
        )
        self.diff_conv_l = nn.Sequential(
            nn.Conv2d(dim * 2, dim // 2, kernel_size=3, padding=2, dilation=2, groups=dim//2, bias=False),
            nn.InstanceNorm2d(dim // 2),
            nn.GELU()
        )
        self.diff_fuse = nn.Conv2d(dim, dim, 1, bias=False)


        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )


        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.LayerNorm([dim // 4, 1, 1]), 
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )

        self.out_proj = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x1, x2):
        diff_abs = torch.abs(x1 - x2)
        feat_max = torch.maximum(x1, x2)

        cat_input = torch.cat([diff_abs, feat_max], dim=1) # [B, 2C, H, W]

        d_s = self.diff_conv_s(cat_input)
        d_l = self.diff_conv_l(cat_input)
        diff_feat = self.diff_fuse(torch.cat([d_s, d_l], dim=1))

        max_out, _ = torch.max(diff_feat, dim=1, keepdim=True)
        avg_out = torch.mean(diff_feat, dim=1, keepdim=True)
        spatial_desc = torch.cat([max_out, avg_out], dim=1)

        s_mask = self.spatial_att(spatial_desc)

        c_mask = self.channel_att(diff_feat)

        refined_feat = diff_feat * s_mask * c_mask
        out = self.out_proj(refined_feat) + diff_abs

        return out

class EnhancedSEFusionModule(nn.Module):
    def __init__(self, dim, reduction=4): 
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(dim * 3, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )


        self.spatial_refine = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

        mid_dim = max(8, dim // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, mid_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x1, x2, diff):
        cat_feat = torch.cat([x1, x2, diff], dim=1)
        x = self.project(cat_feat)

        x = self.spatial_refine(x)

        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        return x * y.expand_as(x)

import clip

class CLIPScoreMapHead(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 2,
        embed_dim: int = 512,
        clip_name: str = "ViT-B/32",
        freeze_clip: bool = True,
        cache_text: bool = True,
        ctx_scale_init: float = 0.1,
        ctx_mlp_ratio: float = 1.0,
    ):
        super().__init__()
        self.out_channels = int(out_channels)
        self.embed_dim = int(embed_dim)
        self.cache_text = bool(cache_text)

        self.clip_model, _ = clip.load(clip_name, device="cpu")
        self.clip_model = self.clip_model.eval()
        if freeze_clip:
            for p in self.clip_model.parameters():
                p.requires_grad = False

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        hidden = max(64, int(embed_dim * ctx_mlp_ratio))
        self.ctx_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, embed_dim),
        )
        self.ctx_scale = nn.Parameter(torch.tensor(float(ctx_scale_init)))

        self._text_cache: dict[str, torch.Tensor] = {}


    def _default_category_prompts(self) -> list[str]:
        if self.out_channels == 2:
            return ["no change", "significant land cover change"]
        else:
            prompts = ["no change", "farmland change to bareland", "farmland change to building", "farmland change to road","farmland change to vegetation", "farmland change to water"]
            if len(prompts) < self.out_channels:
                prompts += [f"class {i}" for i in range(len(prompts), self.out_channels)]
            return prompts[: self.out_channels]

    def _normalize_context_prompts(self, prompts) -> list[str]:

        if prompts is None:
            return []

        if isinstance(prompts, dict):
            for k in ["prompts", "prompt", "text", "texts"]:
                if k in prompts:
                    prompts = prompts[k]
                    break
            else:
                return []

        if torch.is_tensor(prompts):
            return []

        if isinstance(prompts, (list, tuple)):
            if len(prompts) == 0:
                return []
            if isinstance(prompts[0], (list, tuple)):
                prompts = prompts[0]
            return [str(p) for p in list(prompts) if p is not None]

        return []


    @torch.no_grad()
    def _encode_text_list(self, prompt_list: list[str], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Returns normalized text features of shape (N, embed_dim) on target device/dtype.
        Uses CPU cache (float32) keyed by exact prompt strings ordering.
        """
        if len(prompt_list) == 0:
            # return empty
            return torch.empty((0, self.embed_dim), device=device, dtype=dtype)

        key = "||".join(prompt_list)
        if self.cache_text and key in self._text_cache:
            return self._text_cache[key].to(device=device, dtype=dtype)

        # Move CLIP model once-per-device as needed (avoid per-iter .to overhead)
        if next(self.clip_model.parameters()).device != device:
            self.clip_model = self.clip_model.to(device)

        tokens = clip.tokenize(prompt_list).to(device)
        txt = self.clip_model.encode_text(tokens).float()              # (N,512) float32
        txt = F.normalize(txt, p=2, dim=-1)                            # unit vectors

        if self.cache_text:
            self._text_cache[key] = txt.detach().cpu()                 # store float32 on CPU

        return txt.to(device=device, dtype=dtype)


    def forward(self, x: torch.Tensor, prompts=None) -> torch.Tensor:
        """
        x: (B, C_in, H, W)
        prompts: context prompt pool from loader (e.g., len=6). Treated as context.
        output: score_map (B, out_channels, H, W)
        """
        # visual embeddings
        v = self.proj(x)                               # (B, embed_dim, H, W)
        v = F.normalize(v, p=2, dim=1)

        device, dtype = v.device, v.dtype

        cat_prompts = self._default_category_prompts()
        t_cat = self._encode_text_list(cat_prompts, device=device, dtype=dtype)  # (K, embed_dim)

        ctx_prompts = self._normalize_context_prompts(prompts)

        if len(ctx_prompts) > 0:
            t_ctx_all = self._encode_text_list(ctx_prompts, device=device, dtype=dtype)  # (M, embed_dim)
            if t_ctx_all.numel() > 0:
                t_ctx = t_ctx_all.mean(dim=0)  # (embed_dim,)
                delta = self.ctx_mlp(t_ctx.unsqueeze(0)).squeeze(0)  # (embed_dim,)
                t_cat = F.normalize(t_cat + self.ctx_scale * delta.unsqueeze(0), p=2, dim=-1)

        score_map = torch.einsum("bchw,nc->bnhw", v, t_cat)            # (B, K, H, W)
        score_map = score_map * self.logit_scale.exp().to(dtype=score_map.dtype)

        return score_map


class CDMamba(nn.Module):
    def __init__(
            self,
            spatial_dims: int = 2,
            init_filters: int = 16,
            in_channels: int = 3,
            out_channels: int = 2,
            conv_mode: str = "deepwise",
            local_query_model="orignal_dinner",
            dropout_prob: float | None = None,
            act: tuple | str = ("RELU", {"inplace": True}),
            norm: tuple | str = ("GROUP", {"num_groups": 8}),
            norm_name: str = "",
            num_groups: int = 8,
            use_conv_final: bool = True,
            blocks_down: tuple = (1, 2, 2, 4),
            blocks_up: tuple = (1, 1, 1),
            mode: str = "",
            up_mode="SRCM",
            up_conv_mode="deepwise",
            resdiual=False,
            stage=4,
            diff_abs="later",
            mamba_act="silu",
            upsample_mode: UpsampleMode | str = UpsampleMode.NONTRAINABLE,
            use_score_inject: bool = True,
            score_embed_dim: int = 256,
            meta_dim: int = 7,
            
            detach_score_feat: bool = False,   
    ):
        super().__init__()

        if spatial_dims not in (2, 3):
            raise ValueError("`spatial_dims` can only be 2 or 3.")

        self.spatial_dims = spatial_dims
        self.init_filters = init_filters
        self.channels_list = [self.init_filters, self.init_filters * 2, self.init_filters * 4, self.init_filters * 8]
        self.in_channels = in_channels
        self.blocks_down = blocks_down
        self.blocks_up = blocks_up
        self.dropout_prob = dropout_prob
        self.act = act
        self.act_mod = get_act_layer(act)
        self.conv_mode = conv_mode
        self.use_conv_final = use_conv_final
        self.up_mode = up_mode
        self.up_conv_mode = up_conv_mode
        self.upsample_mode = UpsampleMode(upsample_mode)

        self.use_score_inject = use_score_inject
        self.detach_score_feat = bool(detach_score_feat)
        self.gating_conv = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        if norm_name:
            if norm_name.lower() != "group":
                raise ValueError(f"Deprecating option 'norm_name={norm_name}', please use 'norm' instead.")
            norm = ("group", {"num_groups": num_groups})
        self.norm = norm

        self.convInit = get_conv_layer(spatial_dims, in_channels, init_filters)

        self.srcm_encoder_layers = self._make_srcm_encoder_layers()
        self.srcm_decoder_layers, self.up_samples = self._make_srcm_decoder_layers(up_mode=self.up_mode)

        self.diff_attentions = nn.ModuleList()
        self.fusion_modules = nn.ModuleList()
        for dim in self.channels_list:
            self.diff_attentions.append(FineGrainedDiffAttention(dim))
            self.fusion_modules.append(EnhancedSEFusionModule(dim))

        self.score_head = CLIPScoreMapHead(in_channels=init_filters, out_channels=out_channels, embed_dim=512)


        self.conv_final = self._make_final_conv(out_channels, in_channels=init_filters)

        if dropout_prob is not None:
            self.dropout = Dropout[Dropout.DROPOUT, spatial_dims](dropout_prob)

    def _make_srcm_encoder_layers(self):
        srcm_encoder_layers = nn.ModuleList()
        blocks_down, spatial_dims, filters, norm, conv_mode = (
            self.blocks_down, self.spatial_dims, self.init_filters, self.norm, self.conv_mode
        )
        for i, item in enumerate(blocks_down):
            layer_in_channels = filters * 2 ** i
            downsample_mamba = (
                get_srcm_layer(spatial_dims, layer_in_channels // 2, layer_in_channels, stride=2, conv_mode=conv_mode)
                if i > 0
                else nn.Identity()
            )
            down_layer = nn.Sequential(
                downsample_mamba,
                *[SRCMBlock(spatial_dims, layer_in_channels, norm=norm, act=self.act, conv_mode=conv_mode) for _ in range(item)]
            )
            srcm_encoder_layers.append(down_layer)
        return srcm_encoder_layers

    def _make_srcm_decoder_layers(self, up_mode):
        srcm_decoder_layers, up_samples = nn.ModuleList(), nn.ModuleList()
        upsample_mode, blocks_up, spatial_dims, filters, norm = (
            self.upsample_mode,
            self.blocks_up,
            self.spatial_dims,
            self.init_filters,
            self.norm,
        )
        Block_up = SRCMBlock

        n_up = len(blocks_up)
        for i in range(n_up):
            sample_in_channels = filters * 2 ** (n_up - i)
            srcm_decoder_layers.append(
                nn.Sequential(
                    *[
                        Block_up(spatial_dims, sample_in_channels // 2, norm=norm, act=self.act, conv_mode=self.up_conv_mode)
                        for _ in range(blocks_up[i])
                    ]
                )
            )
            up_samples.append(
                nn.Sequential(
                    *[
                        get_conv_layer(spatial_dims, sample_in_channels, sample_in_channels // 2, kernel_size=1),
                        get_upsample_layer(spatial_dims, sample_in_channels // 2, upsample_mode=upsample_mode),
                    ]
                )
            )
        return srcm_decoder_layers, up_samples

    def _safe_groupnorm(self, num_groups: int, num_channels: int):
        g = min(num_groups, num_channels)
        while g > 1 and (num_channels % g != 0):
            g -= 1
        return nn.GroupNorm(g, num_channels)

    def _make_final_conv(self, out_channels: int, in_channels: int | None = None):
        in_ch = self.init_filters if in_channels is None else in_channels

        if isinstance(self.norm, (tuple, list)) and str(self.norm[0]).upper() == "GROUP":
            ng = self.norm[1].get("num_groups", 8)
            norm_layer = self._safe_groupnorm(ng, in_ch)
        else:
            norm_layer = get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=in_ch)

        return nn.Sequential(
            norm_layer,
            self.act_mod,
            get_conv_layer(self.spatial_dims, in_ch, out_channels, kernel_size=1, bias=True),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.convInit(x)
        if self.dropout_prob is not None:
            x = self.dropout(x)
        down_x = []
        for down in self.srcm_encoder_layers:
            x = down(x)
            down_x.append(x)
        return x, down_x

    def decode(self, x: torch.Tensor, down_x: list[torch.Tensor]) -> torch.Tensor:
        for i, (up, upl) in enumerate(zip(self.up_samples, self.srcm_decoder_layers)):
            x = up(x) + down_x[i + 1]
            x = upl(x)
        return x

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        meta=None,
        prompts=None,
        **kwargs
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

        x1_feat, down_x1 = self.encode(x1)
        x2_feat, down_x2 = self.encode(x2)

        down_x_combined = []
        for i in range(len(down_x1)):
            feat1, feat2 = down_x1[i], down_x2[i]
            diff_feat = self.diff_attentions[i](feat1, feat2)
            combined_feat = self.fusion_modules[i](feat1, feat2, diff_feat)
            down_x_combined.append(combined_feat)

        down_x_combined.reverse()
        feat = self.decode(down_x_combined[0], down_x_combined)     # (B, init_filters, H, W)

        score_in = feat.detach() if self.detach_score_feat else feat
        score_map = self.score_head(score_in, prompts=prompts)  # (B, 2, H, W)

        gate = self.gating_conv(score_map) # (B, 1, H, W) range [0, 1]
        

        feat_refined = feat * (1 + gate) 

        logits = self.conv_final(feat_refined)



        if self.training:
            return logits, score_map
        return logits, score_map



if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = CDMamba(
        spatial_dims=2,
        in_channels=3,
        out_channels=2,
        init_filters=32,
        norm=("GROUP", {"num_groups": 8}),
        mode="DAM",
        conv_mode='orignal',
        stage=4,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1)
    ).to(device)

    x = torch.randn(2, 3, 256, 256).to(device)
    y = model(x, x)
    print(f"CDMamba (DAM version) Output: {y.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")