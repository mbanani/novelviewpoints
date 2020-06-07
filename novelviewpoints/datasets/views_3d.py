import itertools

import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from util.rotation import (
    euler2matrix,
    matrix2euler,
    matrix_angle,
    relative_matrix,
)

from .dataset_util import AbstractDataset


class Views3DDataset(AbstractDataset):
    def __init__(
        self,
        name,
        data_root,
        split,
        data_dict,
        rot_rep,
        rgb_transform,
        n_views,
        corrupt_vp,
    ):
        super(Views3DDataset, self).__init__(name, data_root, img_dim=256)
        assert n_views > 1  # cannot have relative for single view problems
        if corrupt_vp:
            assert n_views == 2 and rot_rep in [
                "Rt",
                "matrix",
            ]  # corruption only supported for 2 views with rot matrices

        # Load instance data from csv-file
        self.data_dict = data_dict
        self.rot_rep = rot_rep
        self.n_views = n_views
        self.corrupt_vp = corrupt_vp
        self.split = split

        self.num_classes = len(data_dict)
        self.instances = self.dict_to_instances(self.data_dict)

        # transform functions
        self.transform_rgb = rgb_transform
        self.transform_depth = transforms.Compose([transforms.ToTensor()])
        self.get_canonicalized = False

        # Print out dataset stats
        print("================================")
        print("Stats for {} Viewpoint Dataset".format(self.name))
        print("\tDataset size     : ", len(self.instances))
        print("---------------------------------")

    """
        Retuns the Length of the dataset
    """

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        cls_num, cls_id, m_id, ids = self.instances[index]
        m_instance = self.data_dict[cls_id][m_id]
        vox_path = m_instance["voxel_path"]
        voxel = self.get_voxel(vox_path)
        if self.corrupt_vp:
            # 2 negatives per positive
            corrupt_r = np.random.randint(3)
            corrupt = 1.0 if corrupt_r > 0 else 0.0
        else:
            corrupt = 0.0
        corrupt_angle = 0.0
        corrupt_euler = np.array([0.0, 0.0, 0.0])

        def _get_view(_id):
            # load and crop images
            instance = m_instance["views"][_id]
            bbox = instance["bbox"] if "bbox" in instance else None
            view = instance["viewpoint"]
            image = self.get_rgb(instance["image_path"], bbox)
            depth = self.get_exr(instance["depth_path"], bbox, ndim=1)

            view = np.array(view, dtype=float)
            view = self.transform_view(view)
            view = torch.Tensor(view)  # convert to torch tensor
            image = self.transform_rgb(image)
            depth = depth.clip(max=10)  # assuming the world ends in 10 meters
            depth = self.transform_depth(depth)
            fg = (depth < 10).float()

            return image, view, fg, depth

        if self.n_views == 1:
            img, view, fg, depth = _get_view(ids[0])
        else:
            img, view, fg, depth = [], [], [], []
            _, view_ref, _, _ = _get_view(ids[0])
            for _id in ids:
                _img, _view, _fg, _depth = _get_view(_id)
                _view = self.relative_view(_view, view_ref)
                if self.corrupt_vp and corrupt and len(view) == 1:
                    _euler = m_instance["views"][_id]["viewpoint"][:3]
                    _view, corrupt_angle, corrupt_euler = self.corrupt_view(
                        _view, corrupt_r, _euler
                    )

                img.append(_img.unsqueeze(0))
                view.append(_view.unsqueeze(0))
                fg.append(_fg.unsqueeze(0))
                depth.append(_depth.unsqueeze(0))

            # concatenate all images
            img = torch.cat(tuple(img), dim=0)
            view = torch.cat(tuple(view), dim=0)
            fg = torch.cat(tuple(fg), dim=0)
            depth = torch.cat(tuple(depth), dim=0)

        u_id = "{}_{}_{}".format(
            cls_id, m_id, "-".join([str(_id) for _id in ids])
        )
        if self.get_canonicalized:
            canon_vp = torch.cat(
                (view_ref[:, :3].t(), view_ref[:, 3:4]),
                dim=1
            )
            return (
                img,
                depth,
                fg,
                view,
                voxel,
                cls_num,
                corrupt,
                corrupt_angle,
                corrupt_euler,
                canon_vp,
                u_id,
            )
        else:
            return (
                img,
                depth,
                fg,
                view,
                voxel,
                cls_num,
                corrupt,
                corrupt_angle,
                corrupt_euler,
                u_id,
            )

    def corrupt_view(self, view, corrupt_r, euler, min_e=10, max_e=80):
        if False:  # corrupt_r == 1:
            # input is view
            e1 = np.random.randint(min_e, max_e * 2) * (
                -(1 ** np.random.randint(2))
            )  # larger range for azim
            e2 = np.random.randint(min_e, max_e) * (
                -(1 ** np.random.randint(2))
            )
            e3 = np.random.randint(min_e, max_e * 2) * (
                -(1 ** np.random.randint(2))
            )  # larger range for tilt
            e = np.array([e1, e2, e3]).astype(
                np.float
            )  # ordered in the same way as matrix
        elif corrupt_r in [1, 2]:
            e = -1 * np.array(euler)
        else:
            raise ValueError(
                "Corrupt_r should be 1 or 2, but got {}.".format(corrupt_r)
            )
        Rc = euler2matrix(e[[0, 2, 1]], order="YZX", deg=True)
        geo = matrix_angle(Rc)
        view[:, :3] = torch.Tensor(Rc).t() @ view[:, :3]
        e = e * np.pi / 180.0
        return view, geo, e

    def transform_view(self, view):
        if "euler" in self.rot_rep:
            view = view[0:3]
        elif "R" == self.rot_rep:  # ORDER IS YZX for AZIM, TILT, ELEV
            view = euler2matrix(view[:3], order="ZYX", deg=True)
        elif "Rt" == self.rot_rep:  # ORDER IS YZX for AZIM, TILT, ELEV
            R = euler2matrix(view[[0, 2, 1]], order="YZX", deg=True)
            t = np.array([0, 0, view[3]])[:, None]
            view = np.hstack((R, t))  # second vector is t
        else:
            print("Views3d has no rot_rep {}. exiting.".format(self.rot_rep))
            exit()
        return view

    def relative_view(self, view_t, view_r):
        if "R" == self.rot_rep:
            view = torch.Tensor(relative_matrix(view_t, view_r))
        elif "Rt" in self.rot_rep:
            view_t[:, :3] = torch.Tensor(
                relative_matrix(view_t[:, :3], view_r[:, :3])
            )
            view = view_t
        elif "euler" in self.rot_rep:
            R_r = euler2matrix(view_r, order="ZYX", deg=True)
            R_t = euler2matrix(view_t, order="ZYX", deg=True)

            R_rel = relative_matrix(R_t, R_r)
            e_rel = matrix2euler(R_rel, order="ZYX") * 180.0 / np.pi
            view = torch.Tensor(e_rel).float()
            # get to all positive range
            view[0] = min(view[0] + 180, 359)
            view[1] = min(view[1] + 90, 179)
            view[2] = min(view[2] + 180, 359)
        else:
            raise ValueError(
                "Views3d's relative view cannot handle rot_rep {}.".format(
                    self.rot_rep
                )
            )

        return view

    def dict_to_instances(self, data_dict):
        """
        converts the data dictionary into a list of instances
        :return: list containing instances information
        TODO made a bit faster, but still slow given what it's doing
        """
        inst_len = -1
        instances = []
        for cls_num, cls_id in enumerate(tqdm(data_dict)):
            for m_id in data_dict[cls_id]:
                _insts = list(data_dict[cls_id][m_id]["views"].keys())
                if self.name == "shapenet":
                    _insts = _insts[:10]
                if inst_len != len(_insts):
                    inst_len = len(_insts)
                    combs = itertools.permutations(_insts, self.n_views)

                    # just slice -- got randomizations from random viewpoints
                    n_insts = len(_insts)
                    n_combs = n_insts * self.n_views
                    if self.name == "shapenet" and self.split in [
                        "valid",
                        "test",
                    ]:
                        n_combs = int(n_combs / 5)

                    total = np.cumprod(
                        np.arange(n_insts - self.n_views + 1, n_insts + 1)
                    )[-1]
                    step = int(total // n_combs)
                    perm_subset = list(
                        itertools.islice(combs, 0, step * n_combs, step)
                    )

                for c in perm_subset:
                    instances.append([cls_num, cls_id, m_id, c])

        return instances
