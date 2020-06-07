import numpy as np

import torch
import torch.nn as nn
from util.rotation import euler2matrix

from .model_util import initialize_weights


def _get_Rt(euler):
    R = euler2matrix(euler[[0, 2, 1]], order="YZX", deg=True)
    t = np.array([0, 0, 1])[:, None]
    return torch.Tensor(np.hstack((R, t)))


class MultiviewRealism(nn.Module):
    def __init__(self, model_3d, no_refinement):
        super(MultiviewRealism, self).__init__()
        BN = True
        feat_dim = (
            (2 * model_3d.in_feat_dim) if no_refinement else model_3d.feat_dim
        )
        self.realism_check = RealismCheck(feat_dim, model_3d.voxel_dim, BN)
        self.model_3d = model_3d
        self.refine = not no_refinement

        self.euler = self.model_3d.euler
        self.rot_matrices = self.model_3d.rot_matrices

    def fusion(self, feat):
        B, F, H, W, Z = feat.shape
        if self.sample_mode == "fusion":
            feat = (feat * self.depth_fusion.softmax(dim=4)).sum(dim=4)
        elif self.sample_mode == "conv":
            feat = feat.permute(0, 1, 4, 2, 3)  # B x F x Z x H x W
            feat = feat.view(B, F * Z, H, W)  # B x FZ x H x W
            feat = self.depth_conv(feat)
        elif self.sample_mode == "mean":
            feat = feat.mean(dim=4)
        elif self.sample_mode == "center":
            feat = feat[:, :, :, :, Z // 2]
        return feat

    def singleview2tensor(self, rgb, depth, vp, index):
        feat_2d = self.model_3d.encode2d(rgb[:, index])
        _depth = None if depth is None else depth[:, index]
        feat_3d = self.model_3d.unproject(feat_2d, vp[:, index], _depth)
        return feat_3d

    def fuse_and_refine(self, in1_3d, in2_3d):
        in_3d = torch.cat((in1_3d, in2_3d), dim=1)  # batch x 2F x D x H x W
        ref_3d = self.model_3d.refine3d(in_3d)[1] if self.refine else in_3d
        return ref_3d

    def multiview2tensor(self, rgb, depth, vp):
        in1_3d = self.singleview2tensor(rgb, depth, vp, 0)
        in2_3d = self.singleview2tensor(rgb, depth, vp, 1)

        # -- combine multiviews and refine
        ref_3d = self.fuse_and_refine(in1_3d, in2_3d)
        return ref_3d

    def forward(self, rgb, depth, vp):
        tensor_3d = self.multiview2tensor(rgb, depth, vp)
        # -- condense tensor to something manageable and output consistent tensor or not
        mag, eul = self.realism_check(tensor_3d)
        return {"realism_magnitude": mag, "realism_euler": eul}

    def get_viewpoint(self, rgb, depth, vp, batch_size=16):
        with torch.no_grad():
            euler, rot_matrices = (
                self.model_3d.euler,
                self.model_3d.rot_matrices,
            )
            rot_matrices = rot_matrices.cuda()
            score = torch.zeros(euler.shape[0])

            anchor_3d = self.singleview2tensor(rgb, depth, vp, 0)
            target_2d = self.model_3d.encode2d(rgb[:, 1])

            # repeat relevant things
            anchor_3d = anchor_3d.repeat(batch_size, 1, 1, 1, 1)
            target_2d = target_2d.repeat(batch_size, 1, 1, 1)
            target_dep = (
                None
                if depth is None
                else depth[:, 1].repeat(batch_size, 1, 1, 1)
            )

            # project and compare
            num_views = euler.shape[0]
            for i_l in range(0, num_views, batch_size):
                # get indices
                i_u = min(euler.shape[0], i_l + batch_size)
                _bs = i_u - i_l  # current batch size

                # get target_3d

                if target_dep is None:
                    target_3d = self.model_3d.unproject(
                        target_2d[:_bs], rot_matrices[i_l:i_u], None
                    )
                else:
                    target_3d = self.model_3d.unproject(
                        target_2d[:_bs], rot_matrices[i_l:i_u], target_dep[:_bs]
                    )

                joint_3d = self.fuse_and_refine(
                    anchor_3d[:_bs], target_3d[:_bs]
                )
                score[i_l:i_u] = self.realism_check(joint_3d)[0].squeeze(dim=1)

            # get maximum similarity -- add extra dim since it should be batch
            guess = _get_Rt(euler[score.argmin(0)])

            # extra dimension since output should be batched
            bins = self.model_3d.bins
            return {
                "viewpoint": guess[None, :],
                "viewpoint_distribution": score.view(
                    1, bins[0], bins[1], bins[2]
                ),
            }


class RealismCheck(nn.Module):
    """ Inception-inspired 3D network module """

    def __init__(self, feat_dim, voxel_dim, BN):
        super(RealismCheck, self).__init__()

        self.incept_a = inception_3d(feat_dim, feat_dim // 2, BN)
        self.avg_pool = nn.AvgPool3d(2)
        self.incept_b = inception_3d(feat_dim // 2, 1, BN)
        self.fc_reduce = nn.Sequential(
            nn.Linear(int((voxel_dim / 2) ** 3), 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
        )
        self.fc_magnitude = nn.Linear(512, 1)
        self.fc_euler = nn.Linear(512, 3)
        initialize_weights(self)

    def forward(self, x):
        x = self.incept_a(x)
        x = self.avg_pool(x)
        x = self.incept_b(x)
        x = x.view(x.shape[0], -1)
        x = self.fc_reduce(x)
        mag = self.fc_magnitude(x)
        eul = self.fc_euler(x)
        return mag, eul


class inception_3d(nn.Module):
    """ Inception-inspired 3D network module """

    def __init__(self, in_ch, out_ch, BN):
        super(inception_3d, self).__init__()
        in_ch = max(1, in_ch)
        out_ch = max(1, out_ch)
        self.conv_1 = nn.Conv3d(in_ch, in_ch, 1, padding=0)
        self.conv_3 = nn.Conv3d(in_ch, in_ch, 3, padding=1)
        self.conv_5 = nn.Conv3d(in_ch, in_ch, 5, padding=2)
        self.max_3 = nn.MaxPool3d(3, stride=1, padding=1)

        self.out_conv = nn.Conv3d(in_ch * 4, out_ch, 1)

        self.norm_1 = nn.BatchNorm3d(in_ch * 4) if BN else nn.Identity()
        self.norm_2 = nn.BatchNorm3d(out_ch) if BN else nn.Identity()
        self.activ = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        c1 = self.conv_1(x)
        c3 = self.conv_3(x)
        c5 = self.conv_5(x)
        m3 = self.max_3(x)
        x = torch.cat((c1, c3, c5, m3), dim=1)
        x = self.activ(self.norm_1(x))

        x = self.out_conv(x)
        x = self.activ(self.norm_2(x))
        return x
