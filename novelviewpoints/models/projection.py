"""
    Projection layer:
    - adapted from code by Abhishek Karr for Learned Stereo Machine
        link:

    This layer projects 2D feature map onto a 3D feature tensor.
    Input:
        - features: 4D tensor of image features (Batch x W x H x F)
        - KRcam:    3D tensor of the camera coordinates (Batch x 3 x 4)

    Output:
        - grid      image features projected into a grid

"""
import torch
import torch.nn.functional as F
from torch import nn


def get_camera_params():
    fx = fy = 1  # -420
    x0 = y0 = 0  # im_dim/2

    R = torch.Tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1]])
    K = torch.Tensor([[fx, 0, x0], [0, fy, y0], [0, 0, 1]])
    return R, K


class Project2Dto3D(nn.Module):
    def __init__(self, res):
        super(Project2Dto3D, self).__init__()
        self.res = res
        self.nVox = self.res ** 3
        self.grid = self.generate_grid()

        self.R, self.K = get_camera_params()  # Projection Parameters

    def generate_grid(self):
        g_range = torch.linspace(-0.5, 0.5, self.res)
        grid = torch.stack(torch.meshgrid(g_range, g_range, g_range))
        grid = grid.view(3, -1)
        # convert to homogeneous coordinates
        grid = torch.cat((grid, torch.ones(1, self.nVox)), dim=0)
        return grid

    def forward(self, features, Rcam=None, depth=None, return_sdf=False):
        n_batch = features.shape[0]
        self.K, self.grid = self.K.cuda(), self.grid.cuda()

        # if no RCam, use an arbitary one for no rotation
        if Rcam is None:
            Rcam = self.R[None, :, :].repeat(n_batch, 1, 1).cuda()

        # Rotate grid using KRcam
        KRcam = self.K.matmul(Rcam)  # batch x 3 x 4
        b_grid = KRcam.matmul(self.grid)  # batch x 3 x XYZ

        if depth is not None:
            # calculate depth grid
            d_grid = (
                b_grid[:, 2:] - 1
            )  # scale from 0.5 to 1.5 instead of 1.5-2.5:
            d_grid = d_grid.view(
                -1, 1, self.res, self.res, self.res
            )  # batch x 1 x X x Y x Z
            d_grid = d_grid.permute(0, 1, 4, 3, 2)  # batch x 1 x Z x Y x X

        # remap grid to [-1, 1] range
        # z = 0 for 3D interplation of a 2D array
        b_grid[:, 2] = b_grid[:, 2] * 0.0
        b_grid = 2 * b_grid

        # remap from XYZ to DxHxW; and move xyz dim to last
        b_grid = b_grid.view(
            -1, 3, self.res, self.res, self.res
        )  # batch x 3 x X x Y x Z
        b_grid = b_grid.permute(0, 4, 3, 2, 1)  # batch x Z x Y x X x 3

        # bilinear using grid_sample
        features = features[:, :, None, :, :]
        f_bil = F.grid_sample(
            features,
            b_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True
        )  # batch x F x Z x Y x X

        if depth is not None:
            depth = depth[:, :, None, :, :]
            d_bil = F.grid_sample(
                depth, b_grid, mode="bilinear",
                padding_mode="border",
                align_corners=True
            )  # batch x 1 x Z x Y x X

            # rescale depth according to Z valuei
            sdf = d_grid - (d_bil - 1.0)  # signed distance function for depth
            weighting = (sdf >= 0).float()
            f_bil = weighting * f_bil

        if return_sdf:
            return f_bil, sdf
        else:
            return f_bil


class Project3Dto2D(nn.Module):
    def __init__(self, res, samples=64):
        super(Project3Dto2D, self).__init__()
        self.res = res
        self.nPix = self.res ** 2
        self.S = samples

        # generate 2D grid
        self.grid = self.generate_img_grid()
        self.R, self.K = get_camera_params()
        self._generate_Xc()

    def generate_img_grid(self):
        g_range = torch.linspace(-0.5, 0.5, self.res)
        z_range = torch.linspace(-0.5, 0.5, self.S)
        grid = torch.stack(torch.meshgrid(g_range, g_range, z_range))
        grid = grid.view(3, -1)
        return grid

    def _generate_Xc(self):
        b_grid = self.grid[None, :, :]  # 1 x 3 x X*Y*Z
        b_grid = b_grid.view(1, 3, self.res ** 2, self.S)  # 1 x 3 x XY x Z
        b_grid = b_grid.permute(0, 3, 1, 2)  # 1 x S x 3 x XY

        # make into homogeneous coordinates
        hom_coord = torch.ones(1, self.S, 1, self.nPix)  # 1 x S x 1 x X*Y
        self.Xc = torch.cat((b_grid, hom_coord), dim=2)  # 1 x S x 4 x X*Y

    def forward(self, f_tensor, R=None, interp="bilinear"):
        n_batch, g_h, g_w, g_d, g_f = f_tensor.shape

        # -- Image2Cam -- precomputed to self.Xc
        Xc = self.Xc.cuda()  # 1 x S x 4 x X*Y

        # -- Cam2World --
        # Use arbitary R if no R is provided
        if R is None:
            R = self.R[None, :, :].cuda().repeat(n_batch, 1, 1)

        # construct {R^T | -R^Tt]
        tr = R[:, :, 3:4] * 0  # batch x 3 x 1
        Rt = R[:, :, :3].transpose(1, 2)  # batch x 3 x 3
        Rt = torch.cat((Rt, tr), dim=2)  # batch x 3 x 4

        Rt = Rt[:, None, ::].repeat(1, self.S, 1, 1)  # batch x S x 3 x 4
        Xw = Rt.matmul(Xc)  # batch x S x 3 x X*Y
        Xw = Xw.view(-1, self.S, 3, self.res, self.res)  # batch x S x 3 x X x Y
        Xw = Xw.permute(0, 1, 4, 3, 2)  # batch x S x Y x X x 3

        # -- Sample 3D -> dim of g_val batch x F x D x H x W
        Xw = Xw * 2  # range from [-0.5, 0.5] to [-1, 1]
        g_val = F.grid_sample(f_tensor, Xw, mode=interp, padding_mode="zeros", align_corners=True)

        # move Z to samples
        g_val = g_val.permute(0, 1, 3, 4, 2)  # batch x F x Y x X x S
        return g_val
