"""
## Uformer: A General U-Shaped Transformer for Image Restoration
## Zhendong Wang, Xiaodong Cun, Jianmin Bao, Jianzhuang Liu
## https://arxiv.org/abs/2106.03106
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.layers import DropPath, to_2tuple, trunc_normal_


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, strides=1):
        super(ConvBlock, self).__init__()

        Conv2d = nn.Conv2d
        ReLU = nn.LeakyReLU

        self.strides = strides
        self.block = nn.Sequential(
            Conv2d(in_channel, out_channel, kernel_size=3, stride=strides, padding=1),
            ReLU(inplace=True),
            Conv2d(out_channel, out_channel, kernel_size=3, stride=strides, padding=1),
            ReLU(inplace=True),
        )
        self.conv11 = Conv2d(in_channel, out_channel, kernel_size=1, stride=strides, padding=0)

    def forward(self, x):
        out1 = self.block(x)
        out2 = self.conv11(x)
        out = out1 + out2
        return out


class UNet(nn.Module):
    def __init__(self, block=ConvBlock, dim=32):
        super(UNet, self).__init__()

        Conv2d = nn.Conv2d
        ConvTranspose2d = nn.ConvTranspose2d

        self.ConvBlock1 = ConvBlock(3, dim, strides=1)
        self.pool1 = Conv2d(dim, dim, kernel_size=4, stride=2, padding=1)

        self.ConvBlock2 = block(dim, dim * 2, strides=1)
        self.pool2 = Conv2d(dim * 2, dim * 2, kernel_size=4, stride=2, padding=1)

        self.ConvBlock3 = block(dim * 2, dim * 4, strides=1)
        self.pool3 = Conv2d(dim * 4, dim * 4, kernel_size=4, stride=2, padding=1)

        self.ConvBlock4 = block(dim * 4, dim * 8, strides=1)
        self.pool4 = Conv2d(dim * 8, dim * 8, kernel_size=4, stride=2, padding=1)

        self.ConvBlock5 = block(dim * 8, dim * 16, strides=1)

        self.upv6 = ConvTranspose2d(dim * 16, dim * 8, 2, stride=2)
        self.ConvBlock6 = block(dim * 16, dim * 8, strides=1)

        self.upv7 = ConvTranspose2d(dim * 8, dim * 4, 2, stride=2)
        self.ConvBlock7 = block(dim * 8, dim * 4, strides=1)

        self.upv8 = ConvTranspose2d(dim * 4, dim * 2, 2, stride=2)
        self.ConvBlock8 = block(dim * 4, dim * 2, strides=1)

        self.upv9 = ConvTranspose2d(dim * 2, dim, 2, stride=2)
        self.ConvBlock9 = block(dim * 2, dim, strides=1)

        self.conv10 = Conv2d(dim, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        conv1 = self.ConvBlock1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.ConvBlock2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.ConvBlock3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.ConvBlock4(pool3)
        pool4 = self.pool4(conv4)

        conv5 = self.ConvBlock5(pool4)

        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.ConvBlock6(up6)

        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.ConvBlock7(up7)

        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.ConvBlock8(up8)

        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.ConvBlock9(up9)

        conv10 = self.conv10(conv9)
        out = x + conv10

        return out


class PosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim=768, s=1):
        super(PosCNN, self).__init__()

        Conv2d = nn.Conv2d

        self.proj = nn.Sequential(Conv2d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim))
        self.s = s

    def forward(self, x, H=None, W=None):
        B, N, C = x.shape
        H = H or int(math.sqrt(N))
        W = W or int(math.sqrt(N))
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return x

    def no_weight_decay(self):
        return ["proj.%d.weight" % i for i in range(4)]


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()

        Linear = nn.Linear
        ReLU = nn.ReLU

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            Linear(channel, channel // reduction, bias=False),
            ReLU(inplace=True),
            Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):  # x: [B, N, C]
        x = torch.transpose(x, 1, 2)  # [B, C, N]
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        x = x * y.expand_as(x)
        x = torch.transpose(x, 1, 2)  # [B, N, C]
        return x


class SepConv2d(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        act_layer="ReLU",
    ):
        super(SepConv2d, self).__init__()

        Conv2d = nn.Conv2d
        if act_layer == "ReLU":
            act_layer = nn.ReLU

        self.depthwise = Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
        )
        self.pointwise = Conv2d(in_channels, out_channels, kernel_size=1)
        self.act_layer = act_layer() if act_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.act_layer(x)
        x = self.pointwise(x)
        return x


class ConvProjection(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        kernel_size=3,
        q_stride=1,
        k_stride=1,
        v_stride=1,
        dropout=0.0,
        last_stage=False,
        bias=True,
    ):
        super().__init__()

        inner_dim = dim_head * heads
        self.heads = heads
        pad = (kernel_size - q_stride) // 2
        self.to_q = SepConv2d(dim, inner_dim, kernel_size, q_stride, pad, bias)
        self.to_k = SepConv2d(dim, inner_dim, kernel_size, k_stride, pad, bias)
        self.to_v = SepConv2d(dim, inner_dim, kernel_size, v_stride, pad, bias)

    def forward(self, x, attn_kv=None):
        _, n, _, h = *x.shape, self.heads
        l = int(math.sqrt(n))
        w = int(math.sqrt(n))

        attn_kv = x if attn_kv is None else attn_kv
        x = rearrange(x, "b (l w) c -> b c l w", l=l, w=w)
        attn_kv = rearrange(attn_kv, "b (l w) c -> b c l w", l=l, w=w)
        # print(attn_kv)
        q = self.to_q(x)
        q = rearrange(q, "b (h d) l w -> b h (l w) d", h=h)

        k = self.to_k(attn_kv)
        v = self.to_v(attn_kv)
        k = rearrange(k, "b (h d) l w -> b h (l w) d", h=h)
        v = rearrange(v, "b (h d) l w -> b h (l w) d", h=h)
        return q, k, v


class LinearProjection(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, bias=True):
        super().__init__()

        Linear = nn.Linear

        inner_dim = dim_head * heads
        self.heads = heads
        self.to_q = Linear(dim, inner_dim, bias=bias)
        self.to_kv = Linear(dim, inner_dim * 2, bias=bias)

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        attn_kv = x if attn_kv is None else attn_kv
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(attn_kv).reshape(B_, N, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k, v = kv[0], kv[1]
        return q, k, v


class LinearProjection_Concat_kv(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, bias=True):
        super().__init__()

        Linear = nn.Linear

        inner_dim = dim_head * heads
        self.heads = heads
        self.to_qkv = Linear(dim, inner_dim * 3, bias=bias)
        self.to_kv = Linear(dim, inner_dim * 2, bias=bias)

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        attn_kv = x if attn_kv is None else attn_kv
        qkv_dec = self.to_qkv(x).reshape(B_, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv_enc = self.to_kv(attn_kv).reshape(B_, N, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k_d, v_d = (
            qkv_dec[0],
            qkv_dec[1],
            qkv_dec[2],
        )  # make torchscript happy (cannot use tensor as tuple)
        k_e, v_e = kv_enc[0], kv_enc[1]
        k = torch.cat((k_d, k_e), dim=2)
        v = torch.cat((v_d, v_e), dim=2)
        return q, k, v


class WindowAttention(nn.Module):
    def __init__(
        self,
        dim,
        win_size,
        num_heads,
        token_projection="linear",
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        se_layer=False,
    ):
        super().__init__()

        Linear = nn.Linear
        Dropout = nn.Dropout
        Softmax = nn.Softmax

        self.dim = dim
        self.win_size = win_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.win_size[0])  # [0,...,Wh-1]
        coords_w = torch.arange(self.win_size[1])  # [0,...,Ww-1]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.win_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if token_projection == "conv":
            self.qkv = ConvProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)
        elif token_projection == "linear_concat":
            self.qkv = LinearProjection_Concat_kv(dim, num_heads, dim // num_heads, bias=qkv_bias)
        else:
            self.qkv = LinearProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)

        self.attn_drop = Dropout(attn_drop)
        self.proj = Linear(dim, dim)
        self.se_layer = SELayer(dim) if se_layer else nn.Identity()
        self.proj_drop = Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = Softmax(dim=-1)

    def forward(self, x, attn_kv=None, mask=None):
        B_, N, C = x.shape
        q, k, v = self.qkv(x, attn_kv)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        ratio = attn.size(-1) // relative_position_bias.size(-1)
        relative_position_bias = repeat(relative_position_bias, "nH l c -> nH l (c d)", d=ratio)

        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            mask = repeat(mask, "nW m n -> nW m (n d)", d=ratio)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N * ratio) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N * ratio)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.se_layer(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, win_size={self.win_size}, num_heads={self.num_heads}"


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=None,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = nn.Linear
        Dropout = nn.Dropout
        if act_layer is None:
            act_layer = nn.GELU

        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LeFF(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=None, drop=0.0):
        super().__init__()

        Linear = nn.Linear
        Conv2d = nn.Conv2d
        if act_layer is None:
            act_layer1 = act_layer2 = nn.GELU
        else:
            act_layer1 = act_layer2 = act_layer

        self.linear1 = nn.Sequential(Linear(dim, hidden_dim), act_layer1())
        self.dwconv = nn.Sequential(
            Conv2d(
                hidden_dim,
                hidden_dim,
                groups=hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            act_layer2(),
        )
        self.linear2 = nn.Sequential(Linear(hidden_dim, dim))

    def forward(self, x):
        # bs x hw x c
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))

        x = self.linear1(x)

        # spatial restore
        x = rearrange(x, " b (h w) (c) -> b c h w ", h=hh, w=hh)
        # bs,hidden_dim,32x32

        x = self.dwconv(x)

        # flaten
        x = rearrange(x, " b c h w -> b (h w) c", h=hh, w=hh)

        x = self.linear2(x)

        return x


def window_partition(x, win_size):
    B, H, W, C = x.shape
    x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C)
    return windows


def window_reverse(windows, win_size, H, W):
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Downsample, self).__init__()

        Conv2d = nn.Conv2d

        self.conv = nn.Sequential(
            Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        B, L, C = x.shape
        # import pdb;pdb.set_trace()
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.conv(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        return out


class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample, self).__init__()

        ConvTranspose2d = nn.ConvTranspose2d

        self.deconv = nn.Sequential(
            ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
        )

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.deconv(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        return out


class InputProj(nn.Module):
    def __init__(
        self,
        in_channel=3,
        out_channel=64,
        kernel_size=3,
        stride=1,
        norm_layer=None,
        act_layer=None,
    ):
        super().__init__()

        Conv2d = nn.Conv2d
        if act_layer is None:
            act_layer = nn.LeakyReLU

        self.proj = nn.Sequential(
            Conv2d(
                in_channel,
                out_channel,
                kernel_size=3,
                stride=stride,
                padding=kernel_size // 2,
            ),
            act_layer(inplace=True),
        )
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        if self.norm is not None:
            x = self.norm(x)
        return x


class OutputProj(nn.Module):
    def __init__(
        self,
        in_channel=64,
        out_channel=3,
        kernel_size=3,
        stride=1,
        norm_layer=None,
        act_layer=None,
    ):
        super().__init__()

        Conv2d = nn.Conv2d

        self.proj = nn.Sequential(
            Conv2d(
                in_channel,
                out_channel,
                kernel_size=3,
                stride=stride,
                padding=kernel_size // 2,
            ),
        )
        if act_layer is not None:
            self.proj.add_module("act_layer", act_layer(inplace=True))
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


class LeWinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        win_size=8,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=None,
        norm_layer=None,
        token_projection="linear",
        token_mlp="leff",
        se_layer=False,
    ):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)

        if norm_layer is None:
            norm_layer = nn.LayerNorm
        VariableDropPath = DropPath

        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.win_size = win_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.win_size:
            self.shift_size = 0
            self.win_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.win_size, "shift_size must in 0-win_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            win_size=to_2tuple(self.win_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            token_projection=token_projection,
            se_layer=se_layer,
        )

        self.drop_path = VariableDropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = (
            Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
            )
            if token_mlp == "ffn"
            else LeFF(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)
        )

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, "
            f"win_size={self.win_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"
        )

    def forward(self, x, mask=None):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))

        # input mask
        if mask is not None:
            input_mask = F.interpolate(mask, size=(H, W)).permute(0, 2, 3, 1)
            input_mask_windows = window_partition(input_mask, self.win_size)  # nW, window_size, window_size, 1
            attn_mask = input_mask_windows.view(-1, self.win_size * self.win_size)  # nW, window_size*window_size
            attn_mask = attn_mask.unsqueeze(2) * attn_mask.unsqueeze(
                1
            )  # nW, window_size*window_size, window_size*window_size
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        # shift mask
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            shift_mask = torch.zeros((1, H, W, 1)).type_as(x)
            h_slices = (
                slice(0, -self.win_size),
                slice(-self.win_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.win_size),
                slice(-self.win_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    shift_mask[:, h, w, :] = cnt
                    cnt += 1
            shift_mask_windows = window_partition(shift_mask, self.win_size)  # nW, window_size, window_size, 1
            shift_mask_windows = shift_mask_windows.view(
                -1, self.win_size * self.win_size
            )  # nW, window_size*window_size
            shift_attn_mask = shift_mask_windows.unsqueeze(1) - shift_mask_windows.unsqueeze(
                2
            )  # nW, window_size*window_size, window_size*window_size
            attn_mask = attn_mask or shift_attn_mask
            attn_mask = attn_mask.masked_fill(shift_attn_mask != 0, float(-100.0))

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.win_size)  # nW*B, win_size, win_size, C
        x_windows = x_windows.view(-1, self.win_size * self.win_size, C)  # nW*B, win_size*win_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, win_size*win_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.win_size, self.win_size, C)
        shifted_x = window_reverse(attn_windows, self.win_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        del attn_mask
        return x


class BasicUformerLayer(nn.Module):
    def __init__(
        self,
        dim,
        output_dim,
        input_resolution,
        depth,
        num_heads,
        win_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=None,
        use_checkpoint=False,
        token_projection="linear",
        token_mlp="ffn",
        se_layer=False,
    ):
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.LayerNorm

        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        # build blocks
        self.blocks = nn.ModuleList(
            [
                LeWinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    win_size=win_size,
                    shift_size=0 if (i % 2 == 0) else win_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    token_projection=token_projection,
                    token_mlp=token_mlp,
                    se_layer=se_layer,
                )
                for i in range(depth)
            ]
        )

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def forward(self, x, mask=None):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, mask)
        return x


class Uformer(nn.Module):
    def __init__(
        self,
        img_size=128,
        in_chans=3,
        out_chans=3,
        embed_dim=32,
        depths=(2, 2, 2, 2, 2, 2, 2, 2, 2),
        num_heads=(1, 2, 4, 8, 16, 16, 8, 4, 2),
        win_size=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        patch_norm=True,
        use_checkpoint=False,
        token_projection="linear",
        token_mlp="ffn",
        se_layer=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        norm_layer = nn.LayerNorm
        Dropout = nn.Dropout

        dowsample = Downsample
        upsample = Upsample

        self.num_enc_layers = len(depths) // 2
        self.num_dec_layers = len(depths) // 2
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.token_projection = token_projection
        self.mlp = token_mlp
        self.win_size = win_size

        self.pos_drop = Dropout(p=drop_rate)

        # stochastic depth
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[: self.num_enc_layers]))]
        conv_dpr = [drop_path_rate] * depths[4]
        dec_dpr = enc_dpr[::-1]

        # build layers

        # Input/Output
        self.input_proj = InputProj(in_channel=in_chans, out_channel=embed_dim, kernel_size=3, stride=1)
        self.output_proj = OutputProj(in_channel=2 * embed_dim, out_channel=out_chans, kernel_size=3, stride=1)

        # Encoder
        self.encoderlayer_0 = BasicUformerLayer(
            dim=embed_dim,
            output_dim=embed_dim,
            input_resolution=(img_size, img_size),
            depth=depths[0],
            num_heads=num_heads[0],
            win_size=win_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=enc_dpr[sum(depths[:0]) : sum(depths[:1])],
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
            token_projection=token_projection,
            token_mlp=token_mlp,
            se_layer=se_layer,
        )
        self.dowsample_0 = dowsample(embed_dim, embed_dim * 2)
        self.encoderlayer_1 = BasicUformerLayer(
            dim=embed_dim * 2,
            output_dim=embed_dim * 2,
            input_resolution=(img_size // 2, img_size // 2),
            depth=depths[1],
            num_heads=num_heads[1],
            win_size=win_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=enc_dpr[sum(depths[:1]) : sum(depths[:2])],
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
            token_projection=token_projection,
            token_mlp=token_mlp,
            se_layer=se_layer,
        )
        self.dowsample_1 = dowsample(embed_dim * 2, embed_dim * 4)
        self.encoderlayer_2 = BasicUformerLayer(
            dim=embed_dim * 4,
            output_dim=embed_dim * 4,
            input_resolution=(img_size // (2**2), img_size // (2**2)),
            depth=depths[2],
            num_heads=num_heads[2],
            win_size=win_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=enc_dpr[sum(depths[:2]) : sum(depths[:3])],
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
            token_projection=token_projection,
            token_mlp=token_mlp,
            se_layer=se_layer,
        )
        self.dowsample_2 = dowsample(embed_dim * 4, embed_dim * 8)
        self.encoderlayer_3 = BasicUformerLayer(
            dim=embed_dim * 8,
            output_dim=embed_dim * 8,
            input_resolution=(img_size // (2**3), img_size // (2**3)),
            depth=depths[3],
            num_heads=num_heads[3],
            win_size=win_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=enc_dpr[sum(depths[:3]) : sum(depths[:4])],
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
            token_projection=token_projection,
            token_mlp=token_mlp,
            se_layer=se_layer,
        )
        self.dowsample_3 = dowsample(embed_dim * 8, embed_dim * 16)

        # Bottleneck
        self.conv = BasicUformerLayer(
            dim=embed_dim * 16,
            output_dim=embed_dim * 16,
            input_resolution=(img_size // (2**4), img_size // (2**4)),
            depth=depths[4],
            num_heads=num_heads[4],
            win_size=win_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=conv_dpr,
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
            token_projection=token_projection,
            token_mlp=token_mlp,
            se_layer=se_layer,
        )

        # Decoder
        self.upsample_0 = upsample(embed_dim * 16, embed_dim * 8)
        self.decoderlayer_0 = BasicUformerLayer(
            dim=embed_dim * 16,
            output_dim=embed_dim * 16,
            input_resolution=(img_size // (2**3), img_size // (2**3)),
            depth=depths[5],
            num_heads=num_heads[5],
            win_size=win_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dec_dpr[: depths[5]],
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
            token_projection=token_projection,
            token_mlp=token_mlp,
            se_layer=se_layer,
        )
        self.upsample_1 = upsample(embed_dim * 16, embed_dim * 4)
        self.decoderlayer_1 = BasicUformerLayer(
            dim=embed_dim * 8,
            output_dim=embed_dim * 8,
            input_resolution=(img_size // (2**2), img_size // (2**2)),
            depth=depths[6],
            num_heads=num_heads[6],
            win_size=win_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dec_dpr[sum(depths[5:6]) : sum(depths[5:7])],
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
            token_projection=token_projection,
            token_mlp=token_mlp,
            se_layer=se_layer,
        )
        self.upsample_2 = upsample(embed_dim * 8, embed_dim * 2)
        self.decoderlayer_2 = BasicUformerLayer(
            dim=embed_dim * 4,
            output_dim=embed_dim * 4,
            input_resolution=(img_size // 2, img_size // 2),
            depth=depths[7],
            num_heads=num_heads[7],
            win_size=win_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dec_dpr[sum(depths[5:7]) : sum(depths[5:8])],
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
            token_projection=token_projection,
            token_mlp=token_mlp,
            se_layer=se_layer,
        )
        self.upsample_3 = upsample(embed_dim * 4, embed_dim)
        self.decoderlayer_3 = BasicUformerLayer(
            dim=embed_dim * 2,
            output_dim=embed_dim * 2,
            input_resolution=(img_size, img_size),
            depth=depths[8],
            num_heads=num_heads[8],
            win_size=win_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dec_dpr[sum(depths[5:8]) : sum(depths[5:9])],
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
            token_projection=token_projection,
            token_mlp=token_mlp,
            se_layer=se_layer,
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, token_projection={self.token_projection}, token_mlp={self.mlp}, win_size={self.win_size}"

    def forward(self, x, *args, mask=None, **kwargs):
        # Input Projection
        y = self.input_proj(x)
        y = self.pos_drop(y)
        # Encoder
        conv0 = self.encoderlayer_0(y, mask=mask)
        pool0 = self.dowsample_0(conv0)
        conv1 = self.encoderlayer_1(pool0, mask=mask)
        pool1 = self.dowsample_1(conv1)
        conv2 = self.encoderlayer_2(pool1, mask=mask)
        pool2 = self.dowsample_2(conv2)
        conv3 = self.encoderlayer_3(pool2, mask=mask)
        pool3 = self.dowsample_3(conv3)

        # Bottleneck
        conv4 = self.conv(pool3, mask=mask)

        # Decoder
        up0 = self.upsample_0(conv4)
        deconv0 = torch.cat([up0, conv3], -1)
        deconv0 = self.decoderlayer_0(deconv0, mask=mask)

        up1 = self.upsample_1(deconv0)
        deconv1 = torch.cat([up1, conv2], -1)
        deconv1 = self.decoderlayer_1(deconv1, mask=mask)

        up2 = self.upsample_2(deconv1)
        deconv2 = torch.cat([up2, conv1], -1)
        deconv2 = self.decoderlayer_2(deconv2, mask=mask)

        up3 = self.upsample_3(deconv2)
        deconv3 = torch.cat([up3, conv0], -1)
        deconv3 = self.decoderlayer_3(deconv3, mask=mask)

        # Output Projection
        y = self.output_proj(deconv3)

        return y + x
