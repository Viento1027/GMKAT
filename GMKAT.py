from src.efficient_kan import KAN
import numpy as np
import torch
from torch import nn
from timm.models.layers import trunc_normal_


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def _init_vit_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


class Cls_head(nn.Module):
    def __init__(self, embed_dim, num_classes):

        super().__init__()

        self.cls = KAN([embed_dim, num_classes])

    def forward(self, x):
        x = nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        out = self.cls(x)
        return out


def dpr_generator(drop_path_rate, num_layers, num_stages):
    """Generate drop path rate list following linear decay rule."""
    dpr_list = [
        x.item() for x in torch.linspace(0, drop_path_rate, sum(num_layers))
    ]
    dpr = []
    cur = 0
    for i in range(num_stages):
        dpr_per_stage = dpr_list[cur:cur + num_layers[i]]
        dpr.append(dpr_per_stage)
        cur += num_layers[i]

    return dpr


class Patch_Embed_stage(nn.Module):
    """Depthwise Convolutional Patch Embedding stage comprised of
    `DWCPatchEmbed` layers."""

    def __init__(self, embed_dim, num_path=3, isPool=False):
        super(Patch_Embed_stage, self).__init__()

        self.patch_embeds = nn.ModuleList([
            DWCPatchEmbed(
                in_chans=embed_dim,
                embed_dim=embed_dim,
                patch_size=3,
                stride=2 if isPool and idx == 0 else 1,
            ) for idx in range(num_path)
        ])

    def forward(self, x):
        att_inputs = []
        for pe in self.patch_embeds:
            x = pe(x)
            att_inputs.append(x)

        return att_inputs


class DWCPatchEmbed(nn.Module):
    """Depthwise Convolutional Patch Embedding layer Image to Patch
    Embedding."""

    def __init__(self,
                 in_chans=3,
                 embed_dim=768,
                 patch_size=16,
                 stride=1,
                 act_layer=nn.Hardswish):
        super().__init__()

        self.dwconv = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size - 1) // 2,
            groups=embed_dim,
            bias=False,
        )
        self.pwconv = nn.Conv2d(embed_dim, embed_dim, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(embed_dim)
        self.act = act_layer() if act_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.dwconv(x)
        x = self.pwconv(x)
        x = self.bn(x)
        x = self.act(x)

        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class KATBlock(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 kan_ratio=2.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 norm_layer=nn.LayerNorm):
        super(KATBlock, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.kan = KAN([dim, int(dim / kan_ratio), dim])

    def forward(self, x):
        b, t, d = x.shape
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.kan(self.norm2(x).reshape(-1, x.shape[-1])).reshape(b, t, d))

        return x


class KATEncoder(nn.Module):

    def __init__(
            self,
            dim,
            num_layers=1,
            num_heads=8,
            kan_ratio=2,
            drop_path_list=[],
            qk_scale=None,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.MKAT_layers = nn.ModuleList([
            KATBlock(
                dim,
                num_heads=num_heads,
                kan_ratio=kan_ratio,
                qk_scale=qk_scale,
                drop_ratio=drop_path_list[idx],
                attn_drop_ratio=0.,
                drop_path_ratio=0.,
                norm_layer=nn.LayerNorm
            ) for idx in range(self.num_layers)
        ])

    def forward(self, x, size):
        H, W = size
        B = x.shape[0]
        for layer in self.MKAT_layers:
            x = layer(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x


class MKAT_stage(nn.Module):
    """Multipath Kolmogorov-Arnold Transformer stage comprised of `KATEncoder`
    layers."""

    def __init__(
            self,
            embed_dim,
            out_embed_dim,
            img_size=16,
            num_layers=1,
            num_heads=8,
            kan_ratio=3,
            num_path=4,
            drop_path_list=[],
    ):
        super().__init__()

        self.mtk_blks = nn.ModuleList([
            KATEncoder(
                embed_dim,
                num_layers,
                num_heads,
                kan_ratio,
                drop_path_list=drop_path_list,
            ) for _ in range(num_path)
        ])
        self.aggregate = nn.Sequential(
            nn.Conv2d(embed_dim * num_path,
                      out_embed_dim, kernel_size=1,
                      stride=1,
                      padding=0),
            nn.SiLU()
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, img_size * img_size, embed_dim))

    def forward(self, inputs):
        att_outputs = []
        for x, encoder in zip(inputs, self.mtk_blks):
            # [B, C, H, W] -> [B, N, C]
            _, _, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)
            att_outputs.append(encoder(x, size=(H, W)))

        out_concat = torch.cat(att_outputs, dim=1)
        out = self.aggregate(out_concat)

        return out


class Res_Block(nn.Module):
    def __init__(self, planes):
        super(Res_Block, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=planes,
                out_channels=planes,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(planes),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=planes,
                out_channels=planes,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(planes),
        )
        self.silu = nn.SiLU()
        self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.conv1(x)
        out = self.silu(out)
        out = self.conv2(out)
        out = out + self.shortcut(x)
        out = self.silu(out)

        return out


class Resnet_stage(nn.Module):

    def __init__(
            self, in_c, out_c, num_layers, isPool=False,
    ):
        super(Resnet_stage, self).__init__()
        self.cnn_block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_c,
                out_channels=out_c,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(out_c),
            nn.SiLU(),
        )
        self.res_block = nn.ModuleList([
            Res_Block(
                planes=out_c,
            ) for _ in range(num_layers - 1)
        ])
        self.pool = nn.MaxPool2d(kernel_size=2 if isPool else 1, stride=2 if isPool else 1)

    def forward(self, x):
        x = self.cnn_block(x)
        for layer in self.res_block:
            x = layer(x)
        x = self.pool(x)
        return x


class GMKAT(nn.Module):
    """Geospatial Multipath Kolmogorov-Arnold Transformer"""

    def __init__(
            self,
            img_size=16,
            num_stages=4,
            num_path=[4, 4, 4, 4],
            num_layers=[1, 1, 1, 1],
            res_depth=[3, 3, 3, 3],
            embed_dims=[36, 72, 144, 288],
            kan_ratios=[8, 8, 4, 4],
            num_heads=[8, 8, 8, 8],
            drop_path_rate=0.0,
            in_chans=9,
            num_classes=2,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_stages = num_stages

        dpr = dpr_generator(drop_path_rate, num_layers, num_stages)

        self.stem = nn.Conv2d(in_channels=in_chans,
                              out_channels=embed_dims[0],
                              kernel_size=3,
                              stride=1,
                              padding=1)
        # Patch embeddings.
        self.patch_embed_stages = nn.ModuleList([
            Patch_Embed_stage(
                embed_dims[idx],
                num_path=num_path[idx],
                isPool=False if idx == 0 else True,
            ) for idx in range(self.num_stages)
        ])

        # Multipath Kolmogorov-Arnold Transformer Branch (MKAT)
        self.mkat_stages = nn.ModuleList([
            MKAT_stage(
                embed_dims[idx],
                embed_dims[idx + 1]
                if not (idx + 1) == self.num_stages else embed_dims[idx],
                int(img_size * np.power(0.5, idx)),
                num_layers[idx],
                num_heads[idx],
                kan_ratios[idx],
                num_path[idx],
                drop_path_list=dpr[idx],
            ) for idx in range(self.num_stages)
        ])

        # ResNet Branch
        self.resnet_block = nn.ModuleList([
            Resnet_stage(
                embed_dims[idx],
                embed_dims[idx + 1]
                if not (idx + 1) == self.num_stages else embed_dims[idx],
                res_depth[idx],
                isPool=False if idx == 0 else True
            ) for idx in range(self.num_stages)
        ])
        self.ca = nn.ModuleList([
            ChannelAttention(
                embed_dims[idx + 1] * 2 if not idx == self.num_stages - 1 else embed_dims[idx] * 2,
                ratio=4
            ) for idx in range(self.num_stages)
        ])
        # Classification head.
        self.cls_head = Cls_head(embed_dims[-1] * 2, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """initialization"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_classifier(self):
        """get classifier function"""
        return self.head

    def forward_features(self, x):
        """forward feature function"""
        # x's shape : [B, C, H, W]

        x = self.stem(x)
        gx = x
        lx = x

        for idx in range(self.num_stages):
            lx = self.resnet_block[idx](lx)
            att_inputs = self.patch_embed_stages[idx](gx)
            gx = self.mkat_stages[idx](att_inputs)
            x = torch.cat((lx, gx), dim=1)  # [b, 2c, h, w]
            x = self.ca[idx](x)
            lx, gx = torch.chunk(x, 2, dim=1)  # [b, c, h, w], [b, c, h, w]

        x = torch.cat((gx, lx), dim=1)

        return x

    def forward(self, x):
        """foward function"""
        x = self.forward_features(x)
        # cls head
        out = self.cls_head(x)
        return out


def gmkat_base():
    model = GMKAT(
        img_size=16,
        num_stages=4,
        num_path=[3, 3, 2, 2],
        num_layers=[3, 3, 6, 3],
        res_depth=[3, 3, 3, 3],
        embed_dims=[36, 72, 144, 288],
        kan_ratios=[4, 4, 4, 4],
        num_heads=[4, 8, 16, 32],
        drop_path_rate=0.3,
        in_chans=9,
        num_classes=2,
    )
    return model


def gmkat_large():
    model = GMKAT(
        img_size=16,
        num_stages=4,
        num_path=[3, 3, 2, 2],
        num_layers=[4, 4, 8, 4],
        res_depth=[4, 4, 4, 4],
        embed_dims=[48, 96, 192, 384],
        kan_ratios=[4, 4, 4, 4],
        num_heads=[4, 8, 16, 32],
        drop_path_rate=0.3,
        in_chans=9,
        num_classes=2,
    )
    return model


if __name__ == '__main__':
    model = gmkat_base()
    input_image = torch.randn(2, 9, 16, 16)
    output = model(input_image)
    print(output.shape)
