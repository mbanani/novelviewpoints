import copy
import os

import numpy as np

import cv2
import torch
import util.binvox_rw as binvox_rw
from PIL import Image


"""
Abstract class that takes care of a lot of the boiler plate stuff that I have for all datasets
"""


class AbstractDataset(torch.utils.data.Dataset):
    def __init__(self, name, data_root, img_dim):
        # dataset parameters
        self.name = name
        self.root = data_root
        self.img_dim = img_dim

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass

    def get_rgba(self, path, bbox=None):
        path = os.path.join(self.root, path)
        with open(path, "rb") as f:
            with Image.open(f) as img:
                img = img.convert("RGB")
                if bbox is not None:
                    img = img.crop(box=bbox)
                return img

    def get_rgb(self, path, bbox=None):
        path = os.path.join(self.root, path)
        with open(path, "rb") as f:
            with Image.open(f) as img:
                img = img.convert("RGB")
                if bbox is not None:
                    img = img.crop(box=bbox)
                return img

    def get_alpha(self, path, bbox=None):
        path = os.path.join(self.root, path)
        with open(path, "rb") as f:
            with Image.open(f) as img:
                r, g, b, a = img.split()
                if bbox is not None:
                    a = a.crop(box=bbox)
                a = (np.array(a) > 0).astype(dtype=np.float)
        return a

    def get_exr(self, path, bbox=None, ndim=1):
        """
        loads an .exr file as a numpy array
        """
        # get absolute path
        path = os.path.join(self.root, path)

        depth = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        depth = depth[:, :, 0]  # all channels are the same

        # crop -- bbox is  (left, upper, right, lower)-tuple.
        if bbox is not None:
            l, u, r, d = bbox
            depth[l:r, u:d]

        return depth

    def get_vox_bb(self, vox):
        d0 = np.any(vox, axis=(1, 2))
        d1 = np.any(vox, axis=(0, 2))
        d2 = np.any(vox, axis=(0, 1))
        min1, max1 = np.where(d0)[0][[0, -1]]
        min2, max2 = np.where(d1)[0][[0, -1]]
        min3, max3 = np.where(d2)[0][[0, -1]]

        return np.array([min1, min2, min3, max1, max2, max3], dtype=int)

    def get_voxel(self, path):
        # loads an .binvox array
        if self.get_canonicalized == False:
            return -1

        # get absolute path
        path = os.path.join(self.root, path)
        try:
            with open(path, "rb") as f:
                voxel = binvox_rw.read_as_3d_array(f).data
        except:
            return 0

        bb = self.get_vox_bb(voxel)
        bb_range = bb[3:] - bb[0:3]
        s = (128 - bb_range) // 2
        e = s + bb_range
        new_vox = np.zeros((128, 128, 128), dtype=bool)
        new_vox[s[0] : e[0], s[1] : e[1], s[2] : e[2]] = voxel[
            bb[0] : bb[3], bb[1] : bb[4], bb[2] : bb[5]
        ]
        new_vox = copy.deepcopy(new_vox[:, ::-1, :])

        new_vox = torch.tensor(new_vox).float()
        new_vox = (
            torch.max_pool3d(new_vox[None, None, :], kernel_size=4, stride=4)[
                0, 0
            ]
            > 0
        )
        new_vox = new_vox.permute(1, 2, 0)  # place in the same frame as images
        return new_vox
