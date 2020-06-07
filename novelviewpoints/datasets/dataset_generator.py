import copy
import os
import pickle

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from .views_3d import Views3DDataset

DATA_ROOT = None

def _pkl_to_dict(pkl_file):
    with open(pkl_file, "rb") as f:
        dict_ = pickle.load(f)
        return dict_


def get_loaders(
    name,
    batch_size,
    num_workers,
    split,
    rot_rep,
    n_views,
    corrupt_vp,
):

    # Use ImageNet mean/std
    rgb_mean = [0.485, 0.456, 0.406]
    rgb_std = [0.229, 0.224, 0.225]

    rgb_t_fn = transforms.Compose(
        [
            transforms.Resize((256, 256), interpolation=Image.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize(mean=rgb_mean, std=rgb_std),
        ]
    )

    if split is None:
        train_set = get_dataset(
            name, rgb_t_fn, "train", rot_rep, n_views, corrupt_vp,
        )
        valid_set = get_dataset(
            name, rgb_t_fn, "valid", rot_rep, n_views, corrupt_vp,
        )

        train_loader = DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
            drop_last=True,
        )

        valid_loader = DataLoader(
            dataset=valid_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

        return train_loader, valid_loader

    elif split in ["train", "valid", "test", "all"]:
        _set = get_dataset(name, rgb_t_fn, split, rot_rep, n_views, corrupt_vp,)

        _loader = DataLoader(
            dataset=_set,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            drop_last=split == "train",
        )
        return _loader
    else:
        print("Data split can be None, train, valid, or test. exiting.")
        exit()


def get_dataset(
    name, img_transform, split, rot_rep, n_views, corrupt_vp,
):
    if "shapenet" in name:
        data_root = os.path.join(DATA_ROOT, "ShapeNet55_render_background")
        data_dict = {}
        for s in ["train", "valid", "test"]:
            data_dict[s] = _pkl_to_dict("data/shapenet55_{}.pkl".format(s))

    elif "pix3d" in name:
        data_root = os.path.join(DATA_ROOT, "pix3d_render_background")
        data_dict = {}
        for s in ["train", "valid", "test", "all"]:
            data_dict[s] = _pkl_to_dict("data/pix3d_{}.pkl".format(s))

    elif "thingi10k" in name:
        data_root = os.path.join(DATA_ROOT, "Thingi10k_render_background")
        data_dict = {}
        for s in ["train", "valid", "test", "all"]:
            data_dict[s] = _pkl_to_dict("data/thingi10k_{}.pkl".format(s))

    elif "shapenet_vox" in name:
        data_root = os.path.join(DATA_ROOT, "ShapeNet55_render_background")
        data_dict = {}
        for s in ["train", "valid", "test"]:
            data_dict[s] = _pkl_to_dict("data/shapenet55_{}.pkl".format(s))
            # Only using airplane class
            data_dict[s] = {"02691156": data_dict[s]["02691156"]}

    else:
        raise ValueError("Unknown dataset ({})".format(name))

    assert split in data_dict

    _dataset = Views3DDataset(
        name=name,
        data_root=data_root,
        split=split,
        data_dict=data_dict[split],
        rot_rep=rot_rep,
        rgb_transform=img_transform,
        n_views=n_views,
        corrupt_vp=corrupt_vp,
    )

    return _dataset
