import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from util.rotation import euler2matrix

from .model_util import initialize_weights
from .projection import Project2Dto3D, Project3Dto2D
from .unet_3d import UNet3D_Base


def _get_Rt(euler):
    R = euler2matrix(euler[[0, 2, 1]], order="YZX", deg=True)
    t = np.array([0, 0, 1])[:, None]
    return torch.Tensor(np.hstack((R, t)))


class RepresentObjectsMultiview(nn.Module):
    def __init__(
        self,
        in_chan,
        out_chan,
        rot_rep,
        feat_dim,
        voxel_dim,
        sample_mode="fusion",
        small_decoder=False,
        no_refinement=False,
    ):
        super(RepresentObjectsMultiview, self).__init__()

        in_feat_dim = feat_dim
        batch_norm = True
        n_samples = 64
        im_dim = 256  # image sptial dimension
        feat2d_dim = im_dim // 8  # Features spatial dimensions

        self.small_decoder = small_decoder
        self.in_chan = in_chan
        self.rot_rep = rot_rep
        self.voxel_dim = voxel_dim
        self.feat_dim = feat_dim  # Number of features
        self.in_feat_dim = in_feat_dim  # Number of feature inputs
        self.bins = [18, 9, 9]
        self.sample_mode = sample_mode

        self.permeability = nn.Sequential(
            nn.Linear(in_feat_dim, 1), nn.Sigmoid()
        )

        self.unproject = Project2Dto3D(voxel_dim)
        self.refine3d = UNet3D_Base(2 * in_feat_dim, feat_dim, batch_norm)
        self.project = Project3Dto2D(feat2d_dim, samples=n_samples)

        self.depth_fusion = nn.Parameter(torch.randn(1, 1, 1, 1, n_samples))
        self.depth_conv = nn.Conv2d(n_samples * in_feat_dim, feat_dim, 1)

        self.encode2d = Encode2D(in_chan, feat_dim, BN=batch_norm)
        self.decode2d = Decode2D(feat_dim, out_chan, BN=batch_norm)
        self.to_refine = not no_refinement

        self.generate_euler()

        # initialize
        initialize_weights(self)

    def fusion(self, feat):
        if self.sample_mode == "fusion":
            feat = (feat * self.depth_fusion.softmax(dim=4)).sum(dim=4)
        elif self.sample_mode == "conv":
            B, F, H, W, Z = feat.shape
            feat = feat.permute(0, 1, 4, 2, 3)  # B x F x Z x H x W
            feat = feat.view(B, F * Z, H, W)  # B x FZ x H x W
            feat = self.depth_conv(feat)
        return feat

    def permeability_prob(self, x):
        x = x.permute(0, 2, 3, 4, 1)
        x = self.permeability(x)
        x = x.permute(0, 4, 1, 2, 3)
        return x

    def consistency_loss(self, feat, feat_proj):
        # -- l1 loss
        return F.l1_loss(feat, feat_proj, reduction="none").mean(dim=(1, 2, 3))

    def ray_consistency_cost(self, tensor, sdf):
        """ Given a tensor with permeability probabilities (x f'n from the MVC
        paper), and a depth map, calculate the ray consistency cost.
        By assuming an orthographic projection; depth slices from Karr's LSM
        paper become equivalent to ray traces.
        Input:  tensor  permeability tensor         B x 1 x H x W x Z-samples
                sdf     signed-distance function    B x 1 x H x W x Z-samples
        """
        # q^d(z_p = i) = (1-x_i) *  cum_prod(x_j; 0 <= j < i)
        #              = (1-x_i) * (cumprod(x_j; 0 <= j <= i)/x_i ) for x_i > 0
        # we use cumprod and divide by current x_i to keep code clean;
        # clamping is done to avoid edge case of x_i = 0;
        # cumprod would return 0 for q_i
        x_p = self.permeability_prob(tensor)
        q_d = (
            (1.0 - x_p)
            * torch.cumprod(x_p.clamp(min=1e-9), dim=1)
            / x_p.clamp(min=1e-9)
        )

        # compute event cost -- could define a mask version, but low priority.
        psi = sdf.abs()

        # return ray consistency loss per instance -- mean across pixels and ray
        ray_cost = psi * q_d
        ray_cost = ray_cost.mean(dim=(1, 2, 3, 4))
        return ray_cost

    def get_canonical_refined(self, rgb, depth, vp, vp_decanon):
        # All inputs are batch (x #Views x *) -- where * depends on the input type
        def _2Dto3D(img, index):
            feat_2d = self.encode2d(rgb[:, index])
            if depth is None:
                feat_3d = self.unproject(feat_2d, vp[:, index], None)
                return feat_3d, feat_2d, None
            else:
                feat_3d, sdf_3d = self.unproject(
                    feat_2d, vp[:, index], depth[:, index], return_sdf=True
                )
                return feat_3d, feat_2d, sdf_3d

        def _3Dto2D(tensor, index):
            proj_2d = self.project(tensor, vp[:, index])
            return self.fusion(proj_2d), proj_2d

        # -- encode 2d
        in1_3d, in1_2d, sdf1_3d = _2Dto3D(rgb, 0)
        in2_3d, in2_2d, sdf2_3d = _2Dto3D(rgb, 1)

        # -- combine multiviews and refine
        if self.to_refine:
            in_3d = torch.cat((in1_3d, in2_3d), dim=1)  # batch x 2F x D x H x W
            _, ref_3d = self.refine3d(in_3d)
        else:
            ref_3d = in1_3d + in2_3d

        ref_3d = self.project(ref_3d, vp_decanon)
        return ref_3d

    def get_refined(self, rgb, depth, vp):
        # All inputs are batch (x #Views x *) -- where * depends on the input type
        def _2Dto3D(img, index):
            feat_2d = self.encode2d(rgb[:, index])
            if depth is None:
                feat_3d = self.unproject(feat_2d, vp[:, index], None)
                return feat_3d, feat_2d, None
            else:
                feat_3d, sdf_3d = self.unproject(
                    feat_2d, vp[:, index], depth[:, index], return_sdf=True
                )
                return feat_3d, feat_2d, sdf_3d

        def _3Dto2D(tensor, index):
            proj_2d = self.project(tensor, vp[:, index])
            return self.fusion(proj_2d), proj_2d

        # -- encode 2d
        in1_3d, in1_2d, sdf1_3d = _2Dto3D(rgb, 0)
        in2_3d, in2_2d, sdf2_3d = _2Dto3D(rgb, 1)

        # -- combine multiviews and refine
        if self.to_refine:
            in_3d = torch.cat((in1_3d, in2_3d), dim=1)  # batch x 2F x D x H x W
            _, ref_3d = self.refine3d(in_3d)
        else:
            ref_3d = in1_3d + in2_3d

        return ref_3d

    def forward(self, rgb, depth, vp):
        # All inputs are batch (x #Views x *) -- where * depends on the input type
        def _2Dto3D(img, index):
            feat_2d = self.encode2d(rgb[:, index])
            if depth is None:
                feat_3d = self.unproject(feat_2d, vp[:, index], None)
                return feat_3d, feat_2d, None
            else:
                feat_3d, sdf_3d = self.unproject(
                    feat_2d, vp[:, index], depth[:, index], return_sdf=True
                )
                return feat_3d, feat_2d, sdf_3d

        def _3Dto2D(tensor, index):
            proj_2d = self.project(tensor, vp[:, index])
            return self.fusion(proj_2d), proj_2d

        # -- encode 2d
        in1_3d, in1_2d, sdf1_3d = _2Dto3D(rgb, 0)
        in2_3d, in2_2d, sdf2_3d = _2Dto3D(rgb, 1)

        # -- combine multiviews and refine
        if self.to_refine:
            in_3d = torch.cat((in1_3d, in2_3d), dim=1)  # batch x 2F x D x H x W
            _, ref_3d = self.refine3d(in_3d)
        else:
            ref_3d = in1_3d + in2_3d

        # -- extract 2d features
        in1_2d_p, in1_2d_zp = _3Dto2D(ref_3d, 0)
        in2_2d_p, in2_2d_zp = _3Dto2D(ref_3d, 1)
        out_2d_p, _ = _3Dto2D(ref_3d, 2)

        # -- consistency losses -- using raytracing if depth is provided
        if depth is not None:
            sdf1_3d_p = self.project(sdf1_3d, vp[:, 0])[
                :, 0:1
            ]  # ends up appending coord
            sdf2_3d_p = self.project(sdf2_3d, vp[:, 1])[
                :, 0:1
            ]  # ends up appending coord
            ray_loss_1 = self.ray_consistency_cost(in1_2d_zp, sdf1_3d_p)
            ray_loss_2 = self.ray_consistency_cost(in2_2d_zp, sdf2_3d_p)
            cons_loss = 0.5 * (ray_loss_1 + ray_loss_2)
        else:
            cons_loss_1 = self.consistency_loss(in1_2d, in1_2d_p)
            cons_loss_2 = self.consistency_loss(in2_2d, in2_2d_p)
            cons_loss = 0.5 * (cons_loss_1 + cons_loss_2)

        # -- decode projected 2d
        in1_rgb_p = self.decode2d(in1_2d_p)
        in2_rgb_p = self.decode2d(in2_2d_p)
        out_rgb_p = self.decode2d(out_2d_p)

        return {
            "reconstruction_1": in1_rgb_p,
            "reconstruction_2": in2_rgb_p,
            "reconstruction_projected": out_rgb_p,
            "2DConsistency_loss": cons_loss,
        }

    def generate_euler(self):
        bins = self.bins
        # generate rotation angles
        euler_1 = torch.linspace(-170, 170, bins[0])
        euler_2 = torch.linspace(-80, 80, bins[1])
        euler_3 = torch.linspace(-170, 170, bins[2])
        euler = torch.meshgrid((euler_1, euler_2, euler_3))
        euler = torch.stack(euler, dim=3)  # bin1 x bin2 x bin3 x 3
        self.euler = euler.view(-1, 3)

        # generate rotation matrices
        self.rot_matrices = torch.stack([_get_Rt(e) for e in self.euler], dim=0)


class Encode2D(nn.Module):
    def __init__(self, in_ch, out_ch, BN):
        super(Encode2D, self).__init__()
        self.conv = nn.Sequential(
            double_conv(in_ch, out_ch // 4, BN),
            nn.MaxPool2d(2),
            double_conv(out_ch // 4, out_ch // 2, BN),
            nn.MaxPool2d(2),
            double_conv(out_ch // 2, out_ch, BN),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Decode2D(nn.Module):
    def __init__(self, in_ch, out_ch, BN):
        super(Decode2D, self).__init__()
        conv_layer = double_conv

        self.conv = nn.Sequential(
            conv_layer(in_ch, in_ch // 2, BN),
            nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            ),
            conv_layer(in_ch // 2, in_ch // 4, BN),
            nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            ),
            conv_layer(in_ch // 4, in_ch // 8, BN),
            nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            ),
            conv_layer(in_ch // 8, out_ch, BN),
            outconv(out_ch, out_ch),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


# ----- UNet 2D Utils -------
class single_conv(nn.Module):
    """(conv => BN => LeakyReLU) """

    def __init__(self, in_ch, out_ch, BN=True):
        super(single_conv, self).__init__()
        in_ch = max(in_ch, 1)
        out_ch = max(out_ch, 1)
        if BN:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.LeakyReLU(inplace=True),
            )

    def forward(self, x):
        x = self.conv(x)
        return x


class double_conv(nn.Module):
    """(conv => BN => LeakyReLU) * 2"""

    def __init__(self, in_ch, out_ch, BN=True):
        super(double_conv, self).__init__()
        in_ch = max(in_ch, 1)
        out_ch = max(out_ch, 1)
        if BN:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.LeakyReLU(inplace=True),
            )

    def forward(self, x):
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        in_ch = max(in_ch, 1)
        out_ch = max(out_ch, 1)
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
