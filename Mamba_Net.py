import torch
import torch.nn as nn
from mamba_ssm import Mamba
import torch.nn.functional as F
import numpy as np
from Deform_conv.network_architecture.synapse.deform_conv import DeformConvPack


# todo 可变形卷积
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding."""
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv1x1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, padding=2, dilation=2, stride=stride, bias=False)


def make_res_layer(inplanes, planes, blocks, stride=1): #  self.blocks, stride=2
    downsample = nn.Sequential(
        conv1x1x1(inplanes, planes, stride),  #
        nn.BatchNorm3d(planes),
    )
    layers = []
    layers.append(BasicBlock(inplanes, planes, stride, downsample))
    downsample_1 = nn.Sequential(
        conv1x1x1(planes, planes, stride=1),  #
    )
    for _ in range(1, blocks):
        layers.append(BasicBlock(planes, planes, stride=1, downsample=downsample_1))
    return nn.Sequential(*layers)  # 第一个有downsample


class MambaLayer(nn.Module):
    def __init__(self,  dim, depth=2, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.Sequential()
        for i in range(depth):
            layer = Mamba(
                d_model=dim,  # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,  # Local convolution width  # 局部卷积宽度
                expand=expand  # Block expansion factor
            )
            self.layers.append(layer)

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=dim,out_channels=dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PReLU(),
        )
        self.scale = nn.Parameter(torch.ones(1))

        def init_weights(m):
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.conv1.apply(init_weights)

    def forward(self, x):
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.layers(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        scale = F.softplus(self.scale)
        out = out + (scale * self.conv1(x))
        return out


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=int(kernel_size / 2)),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, dilation=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        return self.conv(input)

class SingleConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(SingleConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv3d(in_ch, out_ch, 3, padding=1), nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True))

    def forward(self, input):
        return self.conv(input)

class Mamba_UNet(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel, blocks):
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel
        self.blocks = blocks
        bias_opt = True
        super(Mamba_UNet, self).__init__()

        self.eninput = self.encoder(self.in_channel, self.start_channel, bias=bias_opt)
        self.ec1 = self.encoder(self.start_channel, self.start_channel, bias=bias_opt)

        self.ec2 = make_res_layer(self.start_channel, self.start_channel * 2, self.blocks, stride=2)
        self.ec3 = MambaLayer(self.start_channel * 2, depth=1) # self.encoder(self.start_channel*2, self.start_channel*2, bias=bias_opt)#

        self.ec4 = make_res_layer(self.start_channel*2, self.start_channel * 4, self.blocks, stride=2)
        self.ec5 = MambaLayer(self.start_channel * 4, depth=1)

        self.ec6 = make_res_layer(self.start_channel* 4, self.start_channel * 8, self.blocks, stride=2)
        self.ec7 = MambaLayer(self.start_channel * 8, depth=1)

        self.ec8 = make_res_layer(self.start_channel* 8, self.start_channel * 8, self.blocks, stride=2)
        self.ec9 = MambaLayer(self.start_channel * 8, depth=2)

        self.dc1 = self.encoder(self.start_channel * 8 + self.start_channel * 8, self.start_channel * 8, kernel_size=3,
                                stride=1, bias=bias_opt)
        self.dc2 = self.encoder(self.start_channel * 8, self.start_channel * 4, kernel_size=3, stride=1, bias=bias_opt)
        self.dc3 = self.encoder(self.start_channel * 4 + self.start_channel * 4, self.start_channel * 4, kernel_size=3,
                                stride=1, bias=bias_opt)
        self.dc4 = self.encoder(self.start_channel * 4, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)
        self.dc5 = self.encoder(self.start_channel * 2 + self.start_channel * 2, self.start_channel * 4, kernel_size=3,
                                stride=1, bias=bias_opt)
        self.dc6 = self.encoder(self.start_channel * 4, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)
        self.dc7 = self.encoder(self.start_channel * 2 + self.start_channel * 1, self.start_channel * 2, kernel_size=3,
                                stride=1, bias=bias_opt)
        self.dc8 = self.encoder(self.start_channel * 2, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)
        self.dc9 = self.outputs(self.start_channel * 2, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc10 = self.outputs(self.start_channel * 2, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)

        self.up1 = self.decoder(self.start_channel * 8, self.start_channel * 8)
        self.up2 = self.decoder(self.start_channel * 4, self.start_channel * 4)
        self.up3 = self.decoder(self.start_channel * 2, self.start_channel * 2)
        self.up4 = self.decoder(self.start_channel * 2, self.start_channel * 2)

        self.transform = SpatialTransform().cuda()
        self.diff_transform = DiffeomorphicTransform(time_step=7).cuda()

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.PReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.PReLU())
        return layer

    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.PReLU())
        return layer

    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.Tanh())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.Softsign())
        return layer

    def forward(self, x, y):
        x_in = torch.cat([x, y], dim=1)
        e0 = self.eninput(x_in)
        e0 = self.ec1(e0)

        e1 = self.ec2(e0)
        e1 = self.ec3(e1)

        e2 = self.ec4(e1)
        e2 = self.ec5(e2)

        e3 = self.ec6(e2)
        e3 = self.ec7(e3)

        e4 = self.ec8(e3)
        e4 = self.ec9(e4)

        d0 = torch.cat((self.up1(e4), e3), 1)

        d0 = self.dc1(d0)
        d0 = self.dc2(d0)

        d1 = torch.cat((self.up2(d0), e2), 1)

        d1 = self.dc3(d1)
        d1 = self.dc4(d1)

        d2 = torch.cat((self.up3(d1), e1), 1)

        d2 = self.dc5(d2)
        d2 = self.dc6(d2)

        d3 = torch.cat((self.up4(d2), e0), 1)
        d3 = self.dc7(d3)
        d3 = self.dc8(d3)

        f_xy = self.dc9(d3)
        D_f_xy =f_xy # self.diff_transform(f_xy)
        #x_ed = self.transform(x, D_f_xy.permute(0, 2, 3, 4, 1))

        return f_xy


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding."""
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        # todo
        self.conv2 = DeformConvPack(in_channels=planes, out_channels=planes, kernel_size=(3, 3, 3), stride=1, padding=1)#conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride  # 两个卷积

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class DoubleConv(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=int(kernel_size / 2)),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, dilation=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        return self.conv(input)


class SingleConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(SingleConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv3d(in_ch, out_ch, 3, padding=1), nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True))

    def forward(self, input):
        return self.conv(input)


class nnMambaSeg(nn.Module):
    def __init__(self, in_ch=1, channels=32, blocks=3, number_classes=6):
        super(nnMambaSeg, self).__init__()
        self.in_conv = DoubleConv(in_ch, channels, stride=2, kernel_size=3)
        # self.mamba_layer_stem = MambaLayer(channels)

        self.layer1 = make_res_layer(channels, channels * 2, blocks, stride=2)  # 卷积，下采样
        self.mamba_layer_1 = MambaLayer(channels*2)

        self.layer2 = make_res_layer(channels * 2, channels * 4, blocks, stride=2)
        self.mamba_layer_2 = MambaLayer(channels*4)

        self.layer3 = make_res_layer(channels * 4, channels * 8, blocks, stride=2)
        self.mamba_layer_3 = MambaLayer(channels*8)

        self.up5 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv5 = DoubleConv(channels * 12, channels * 4)
        self.up6 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv6 = DoubleConv(channels * 6, channels * 2)
        self.up7 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv7 = DoubleConv(channels * 3, channels)
        self.up8 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv8 = DoubleConv(channels, number_classes)

    def forward(self, x):
        c1 = self.in_conv(x)
        # c1_s = self.mamba_layer_stem(c1) + c1
        c2 = self.layer1(c1)
        c2_s = self.mamba_layer_1(c2) + c2
        c3 = self.layer2(c2_s)
        c3_s = self.mamba_layer_2(c3) + c3
        c4 = self.layer3(c3_s)
        c4_s = self.mamba_layer_3(c4) + c4

        up_5 = self.up5(c4_s)
        merge5 = torch.cat([up_5, c3], dim=1)
        c5 = self.conv5(merge5)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c2], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c1], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        c8 = self.conv8(up_8)
        return c8

class nnMambaLand(nn.Module):
    def __init__(self, in_ch=1, channels=32, blocks=3, number_classes=6):
        super(nnMambaLand, self).__init__()
        self.in_conv = DoubleConv(in_ch, channels, stride=2, kernel_size=3)
        self.mamba_layer_stem = MambaLayer(channels)

        self.layer1 = make_res_layer(channels, channels * 2, blocks, stride=2)
        self.layer2 = make_res_layer(channels * 2, channels * 4, blocks, stride=2)
        self.layer3 = make_res_layer(channels * 4, channels * 8, blocks, stride=2)

        self.up5 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv5 = DoubleConv(channels * 12, channels * 4)
        self.up6 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv6 = DoubleConv(channels * 6, channels * 2)
        self.up7 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv7 = DoubleConv(channels * 3, channels)
        self.up8 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv8 = DoubleConv(channels, number_classes)

    def forward(self, x):
        c1 = self.in_conv(x)
        c1_s = self.mamba_layer_stem(c1) + c1
        c2 = self.layer1(c1_s)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        up_5 = self.up5(c4)
        merge5 = torch.cat([up_5, c3], dim=1)
        c5 = self.conv5(merge5)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c2], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c1], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        c8 = self.conv8(up_8)
        return c8


class nnMambaEncoder(nn.Module):
    def __init__(self, in_ch=1, channels=32, blocks=3, number_classes=6):
        super(nnMambaEncoder, self).__init__()
        self.in_conv = DoubleConv(in_ch, channels, stride=2, kernel_size=3)
        self.mamba_layer_stem = MambaLayer(channels)

        self.layer1 = make_res_layer(channels, channels * 2, blocks, stride=2)
        self.mamba_layer_1 = MambaLayer(channels*2)

        self.layer2 = make_res_layer(channels * 2, channels * 4, blocks, stride=2)
        self.mamba_layer_2 = MambaLayer(channels*4)

        self.layer3 = make_res_layer(channels * 4, channels * 8, blocks, stride=2)
        self.mamba_layer_3 = MambaLayer(channels*8)

        self.pooling = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.mlp = nn.Sequential(nn.Linear(channels*8, channels), nn.ReLU(), nn.Linear(channels, number_classes))

    def forward(self, x):  # 分类
        c1 = self.in_conv(x)
        c1_s = self.mamba_layer_stem(c1) + c1
        c2 = self.layer1(c1_s)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.pooling(c4).view(c4.shape[0], -1)
        c5 = self.mlp(c5)
        return c5


class SpatialTransform(nn.Module):
    def __init__(self):
        super(SpatialTransform, self).__init__()

    def forward(self, mov_image, flow, mod='bilinear'):
        d2, h2, w2 = mov_image.shape[-3:]
        grid_d, grid_h, grid_w = torch.meshgrid(
            [torch.linspace(-1, 1, d2), torch.linspace(-1, 1, h2), torch.linspace(-1, 1, w2)])
        grid_h = grid_h.cuda().float()
        grid_d = grid_d.cuda().float()
        grid_w = grid_w.cuda().float()
        grid_d = nn.Parameter(grid_d, requires_grad=False)
        grid_w = nn.Parameter(grid_w, requires_grad=False)
        grid_h = nn.Parameter(grid_h, requires_grad=False)
        flow_d = flow[:, :, :, :, 0]
        flow_h = flow[:, :, :, :, 1]
        flow_w = flow[:, :, :, :, 2]
        # Remove Channel Dimension
        disp_d = (grid_d + (flow_d)).squeeze(1)
        disp_h = (grid_h + (flow_h)).squeeze(1)
        disp_w = (grid_w + (flow_w)).squeeze(1)
        sample_grid = torch.stack((disp_w, disp_h, disp_d), 4)  # shape (N, D, H, W, 3)
        warped = torch.nn.functional.grid_sample(mov_image, sample_grid, mode=mod, align_corners=True)

        return warped


class DiffeomorphicTransform(nn.Module):
    def __init__(self, time_step=7):
        super(DiffeomorphicTransform, self).__init__()
        self.time_step = time_step

    def forward(self, flow):
        # print(flow.shape)
        d2, h2, w2 = flow.shape[-3:]
        grid_d, grid_h, grid_w = torch.meshgrid(
            [torch.linspace(-1, 1, d2), torch.linspace(-1, 1, h2), torch.linspace(-1, 1, w2)])
        grid_h = grid_h.cuda().float()
        grid_d = grid_d.cuda().float()
        grid_w = grid_w.cuda().float()
        grid_d = nn.Parameter(grid_d, requires_grad=False)
        grid_w = nn.Parameter(grid_w, requires_grad=False)
        grid_h = nn.Parameter(grid_h, requires_grad=False)
        flow = flow / (2 ** self.time_step)

        for i in range(self.time_step):
            flow_d = flow[:, 0, :, :, :]
            flow_h = flow[:, 1, :, :, :]
            flow_w = flow[:, 2, :, :, :]
            disp_d = (grid_d + flow_d).squeeze(1)
            disp_h = (grid_h + flow_h).squeeze(1)
            disp_w = (grid_w + flow_w).squeeze(1)

            deformation = torch.stack((disp_w, disp_h, disp_d), 4)  # shape (N, D, H, W, 3)
            flow = flow + torch.nn.functional.grid_sample(flow, deformation, mode='bilinear', padding_mode="border",
                                                          align_corners=True)
        return flow


def smoothloss(y_pred):
    d2, h2, w2 = y_pred.shape[-3:]
    dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :]) / 2 * d2
    dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :]) / 2 * h2
    dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1]) / 2 * w2
    return (torch.mean(dx * dx) + torch.mean(dy * dy) + torch.mean(dz * dz)) / 3.0


"""
Normalized local cross-correlation function in Pytorch. Modified from https://github.com/voxelmorph/voxelmorph.
"""


class NCC(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """

    def __init__(self, win=9, eps=1e-5):
        super(NCC, self).__init__()
        self.win_raw = win
        self.eps = eps
        self.win = win

    def forward(self, I, J):
        ndims = 3
        win_size = self.win_raw
        self.win = [self.win_raw] * ndims

        weight_win_size = self.win_raw
        weight = torch.ones((1, 1, weight_win_size, weight_win_size, weight_win_size), device=I.device,
                            requires_grad=False)
        conv_fn = F.conv3d

        # compute CC squares
        I2 = I * I
        J2 = J * J
        IJ = I * J

        # compute filters
        # compute local sums via convolution
        I_sum = conv_fn(I, weight, padding=int(win_size / 2))
        J_sum = conv_fn(J, weight, padding=int(win_size / 2))
        I2_sum = conv_fn(I2, weight, padding=int(win_size / 2))
        J2_sum = conv_fn(J2, weight, padding=int(win_size / 2))
        IJ_sum = conv_fn(IJ, weight, padding=int(win_size / 2))

        # compute cross correlation
        win_size = np.prod(self.win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + self.eps)

        # return negative cc.
        return -1.0 * torch.mean(cc)


class DiceLoss(nn.Module):
    """Dice and Xentropy loss"""

    def __init__(self, num_class=36):
        super().__init__()
        self.num_class = num_class

    def forward(self, y_pred, y_true):
        # y_pred = models.round(y_pred)
        # y_pred = nn.functional.one_hot(models.round(y_pred).long(), num_classes=7)
        # y_pred = models.squeeze(y_pred, 1)
        # y_pred = y_pred.permute(0, 4, 1, 2, 3).contiguous()
        y_true = nn.functional.one_hot(y_true, num_classes=self.num_class)
        y_true = torch.squeeze(y_true, 1)
        y_true = y_true.permute(0, 4, 1, 2, 3).contiguous()
        intersection = y_pred * y_true
        intersection = intersection.sum(dim=[2, 3, 4])
        union = torch.pow(y_pred, 2).sum(dim=[2, 3, 4]) + torch.pow(y_true, 2).sum(dim=[2, 3, 4])
        dsc = (2. * intersection) / (union + 1e-5)
        dsc = (1 - torch.mean(dsc))
        return dsc


class Dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice

class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)

class SAD:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean(torch.abs(y_true - y_pred))


if __name__ == "__main__":
    # model = nnMambaSeg().cuda()
    # model = nnMambaLand().cuda()
    model = nnMambaEncoder().cuda()

    input = torch.zeros((8, 1, 128, 128, 128)).cuda()
    output = model(input)
    print(output.shape)