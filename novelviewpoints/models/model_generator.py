from IPython import embed

import torch

from .model_3d_mv import RepresentObjectsMultiview
from .model_3d_realism import MultiviewRealism


def get_model(
    model_name,
    rot_rep,
    in_rep,
    out_rep=None,
    pretrained=None,
    no_refinement=False,
):

    _reps = {"rgb": 3, "rgbd": 4, "depth": 1, "mask": 1}

    def get_c(rep, output=False):
        return 2 * _reps[rep[1:]] if rep[0] == "2" else _reps[rep]

    voxel_dim = 32
    feat_dim = 32
    sample_mode = "conv"
    small_decoder = False
    coord_conv = False
    certainty_fusion = "none"
    single = in_rep[0] != "2"
    in_c = get_c(in_rep)
    out_c = get_c(out_rep, True) if out_rep else None

    if pretrained:
        pretrained_model = get_pretrained(pretrained)

    if model_name == "mvRealism":
        model = MultiviewRealism(pretrained_model, no_refinement=no_refinement)
    elif model_name == "3dRepMV":
        model = RepresentObjectsMultiview(
            in_c,
            out_c,
            rot_rep,
            feat_dim,
            voxel_dim,
            sample_mode,
            small_decoder,
            no_refinement,
        )
    else:
        print("Error: Model {} undefined.".format(model_name))
        exit()

    return model


def get_pretrained(pretrained_path):
    checkpoint = torch.load(pretrained_path)
    ck_args = checkpoint["args"]
    model = get_model(
        ck_args.model,
        ck_args.rot_rep,
        ck_args.model_input,
        ck_args.unet_output,
        ck_args.pretrained,
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model
