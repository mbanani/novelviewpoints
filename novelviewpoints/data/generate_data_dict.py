"""
This converts the view dictionary used for rendering into a data dictionary used
for data loading.
"""
import argparse
import json
import os
import pickle
import random

import numpy as np
from IPython import embed


def real_viewparams2dict(view_params):
    new_dict = {}
    for synset in view_params:
        syn_d = {}
        for mod in view_params[synset]:
            pref = os.path.join(synset, mod)
            mod_id = mod.split("_")[-1]
            mod_d = {
                "views": dict(),
            }
            mod_instances = list(view_params[synset][mod].keys())
            mod_instances.sort()
            for i in range(len(mod_instances)):
                ins = mod_instances[i]
                ins_name = ins.split("_pose")[0]
                ins_actual = "_".join(ins_name.split("_")[-2:])
                if synset == "soda_can":
                    print(ins_actual)
                mod_d["views"][ins_actual] = {
                    "viewpoint": np.array(view_params[synset][mod][ins]),
                    "bbox": None,
                    "image_path": os.path.join(pref, "{}.png".format(ins_name)),
                    "depth_path": os.path.join(
                        pref, "{}_depth.png".format(ins_name)
                    ),
                    "mask_path": os.path.join(
                        pref, "{}_mask.png".format(ins_name)
                    ),
                }
            syn_d[mod_id] = mod_d
        new_dict[synset] = syn_d
    return new_dict


def viewparams2dict(view_params):
    new_dict = {}
    for synset in view_params:
        syn_d = {}
        for mod in view_params[synset]:
            pref = os.path.join(synset, mod)
            mod_d = {
                "views": dict(),
                "voxel_path": os.path.join(pref, "model.binvox"),
            }
            mod_instances = list(view_params[synset][mod].keys())
            mod_instances.sort()
            for i in range(len(mod_instances)):
                ins = mod_instances[i]
                mod_d["views"][ins] = {
                    "viewpoint": np.array(view_params[synset][mod][ins]),
                    "bbox": None,
                    "image_path": os.path.join(pref, "{}_img.png".format(ins)),
                    "depth_path": os.path.join(
                        pref, "{}_depth.exr".format(ins)
                    ),
                }
            syn_d[mod] = mod_d
        new_dict[synset] = syn_d
    return new_dict


def split_dict(data_dict, ratio):
    assert ratio < 1.0 and ratio > 0.0

    subset_dict = {}

    for synset in data_dict:
        subset_dict[synset] = {}

        valid_subset = list(data_dict[synset].keys())
        valid_subset = random.sample(
            valid_subset, int(ratio * len(valid_subset)) + 1
        )

        for valid_model in valid_subset:
            subset_dict[synset][valid_model] = data_dict[synset][valid_model]
            del data_dict[synset][valid_model]

    return data_dict, subset_dict


def pix3d_json2viewparams(dataset_root):
    json_path = os.path.join(dataset_root, "pix3d.json")
    with open(json_path) as f:
        pix3d_json = json.load(f)

    for instance in pix3d_json:
        cls_name = instance["category"]
        model_name = None


def load_pickle(path):
    with open(path, "rb") as f:
        data_dict = pickle.load(f)
    return data_dict


def save_pickle(data_dict, path):
    with open(path, "wb") as f:
        pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


def count_models(data_dict):
    models = 0
    for s in data_dict:
        models += len(data_dict[s])
    return models


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dict_path", type=str, default=None)
    args = parser.parse_args()

    path_prefix = args.dict_path[:-4]
    data_dict = load_pickle(args.dict_path)
    data_dict = viewparams2dict(data_dict)
    print(
        "Number of models in {} split is {}".format(
            "All", count_models(data_dict)
        )
    )
    save_pickle(data_dict, path_prefix + "_all.pkl")

    # train/valid/test 0.8/0.1/0.1
    train_dict, test_dict = split_dict(data_dict, 0.1)
    train_dict, valid_dict = split_dict(train_dict, 1.0 / 9)

    print(
        "Number of models in {} split is {}".format(
            "train", count_models(train_dict)
        )
    )
    print(
        "Number of models in {} split is {}".format(
            "valid", count_models(valid_dict)
        )
    )
    print(
        "Number of models in {} split is {}".format(
            "test", count_models(test_dict)
        )
    )

    save_pickle(train_dict, path_prefix + "_train.pkl")
    save_pickle(valid_dict, path_prefix + "_valid.pkl")
    save_pickle(test_dict, path_prefix + "_test.pkl")
