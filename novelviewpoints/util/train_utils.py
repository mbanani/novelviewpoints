import torch
import numpy as np
from util.metrics import Metrics
from util.rotation import extract_viewpoint
import torch.nn.functional as F


def reconstruction_loss(pred, gt, out_rep):
    if out_rep[0] == "2":
        p_s = pred.shape[1] // 2
        g_s = gt.shape[1] // 2
        l1, m_l1 = reconstruction_loss(pred[:, :p_s], gt[:, :g_s], out_rep[1:])
        l2, m_l2 = reconstruction_loss(pred[:, p_s:], gt[:, g_s:], out_rep[1:])

        return l1 + l2, m_l1 + m_l2

    if out_rep == "depth":
        loss_inst = F.mse_loss(pred, gt, reduction="none")
    elif out_rep == "mask":
        loss_inst = F.binary_cross_entropy_with_logits(
            pred, gt, reduction="none"
        )

    mean_dims = tuple(range(1, len(loss_inst.shape)))
    loss_inst = loss_inst.mean(dim=mean_dims)
    return loss_inst.mean(), loss_inst.detach().cpu().numpy()


def get_flags(parser, evaluate=False):
    # general parameters
    parser.add_argument("--exp_name", type=str, default="test_test")
    parser.add_argument("--num_workers", type=int, default=6)

    # training parameters
    if not evaluate:
        parser.add_argument("--num_epochs", type=int, default=10)
        parser.add_argument("--save_epoch", type=int, default=1)
        parser.add_argument("--eval_epoch", type=int, default=6)
        parser.add_argument("--eval_step", type=int, default=1100)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--optimizer", type=str, default="RAdam")
        parser.add_argument("--scheduler", type=str, default=None)
        parser.add_argument("--resume", type=str, default=None)

    # evaluation details
    # Since the viewpoint inference is VERY slow; you can use p_id and total_p
    # to split your evaluation to P splits, and then merge the resulting dicts
    # to parallelize it. Not the most elegant solution, but it works.
    if evaluate:
        parser.add_argument("--checkpoint", type=str, default=None)
        parser.add_argument("--split", type=str, default="test")
        parser.add_argument("--p_id", type=int, default=None)
        parser.add_argument("--total_p", type=int, default=9)

    # dataset parameters
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_views", type=int, default=2)
    parser.add_argument("--rot_rep", type=str, default="Rt")
    parser.add_argument("--overfit", default=False, action="store_true")

    parser.add_argument("--viewpoint", default=False, action="store_true")
    parser.add_argument("--reconstruction", default=False, action="store_true")
    parser.add_argument("--realism_check", default=False, action="store_true")

    # model parameters
    parser.add_argument("--model", type=str, default="3dRep")
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--model_input", type=str, default="rgb")
    parser.add_argument("--unet_output", type=str, default="rgb")
    parser.add_argument("--depth_sculpt", default=False, action="store_true")
    parser.add_argument("--consistency_loss", default=False, action="store_true")
    parser.add_argument("--no_refinement", default=False, action="store_true")

    args = parser.parse_args()

    assert (
        args.realism_check == (args.model == "mvRealism")
    ) or not args.realism_check
    args.corrupt_vp = True if args.realism_check else False
    args.best_loss = 1e9

    return args


def get_metrics(reconstruction, viewpoint, realism_check, out_rep, header=""):
    logable_metrics, printable_metrics = [], []

    if reconstruction and out_rep[0] == "2":
        l1, p1 = get_metrics(reconstruction, False, out_rep[1:], "t_")
        l2, p2 = get_metrics(reconstruction, viewpoint, out_rep[1:], "r_")
        logable = l1 + l2
        printable = p1 + p2

        logable.remove("t_Recon_loss")
        logable.remove("r_Recon_loss")
        printable.remove("t_Recon_loss")
        printable.remove("r_Recon_loss")
        logable.append("Recon_loss")
        printable.append("Recon_loss")
        return logable, printable

    if realism_check:
        logable_metrics += [
            "RC Accuracy",
            "RC Azim Error",
            "RC Elev Error",
            "RC Tilt Error",
            "RC Euler Loss",
            "RC Mag Error",
            "RC Mag Loss",
        ]
        printable_metrics += [
            "RC Accuracy",
            "RC Azim Error",
            "RC Elev Error",
            "RC Tilt Error",
            "RC Euler Loss",
            "RC Mag Error",
            "RC Mag Loss",
        ]

    if reconstruction:
        if out_rep in ["mutual_vis", "mask"]:
            logable_metrics += [
                "proj_IoU",
                "proj_Recall",
                "proj_Precision",
                "proj_F1",
                "proj_PixAcc",
                header + "IoU",
                header + "Recall",
                header + "Precision",
                header + "F1",
                header + "PixAcc",
                header + "Proj_loss",
                header + "Recon_loss",
                header + "2DConsistency_loss",
            ]
            printable_metrics += [
                "proj_IoU",
                "proj_F1",
                header + "IoU",
                header + "F1",
                header + "Recon_loss",
                header + "Proj_loss",
                header + "2DConsistency_loss",
            ]
        elif out_rep in [
            "depth",
        ]:
            logable_metrics += [
                "proj_thresh_1.25",
                "proj_abs_rel_diff",
                "proj_rms_linear",
                "proj_rms_log",
                "proj_L1_diff",
                header + "thresh_1.25",
                header + "abs_rel_diff",
                header + "rms_linear",
                header + "rms_log",
                header + "L1_diff",
                header + "Recon_loss",
                header + "2DConsistency_loss",
                header + "Proj_loss",
            ]
            printable_metrics += [
                "proj_thresh_1.25",
                "proj_abs_rel_diff",
                "proj_rms_linear",
                "proj_rms_log",
                "proj_L1_diff",
                header + "Proj_loss",
                header + "thresh_1.25",
                header + "abs_rel_diff",
                header + "rms_linear",
                header + "rms_log",
                header + "L1_diff",
                header + "Recon_loss",
                header + "2DConsistency_loss",
            ]
        else:
            logable_metrics = [out_rep + "-L1 Loss"]
            printable_metrics = [out_rep + "-L1 Loss"]

    if viewpoint:
        printable_metrics += ["VP_loss", "VP_Error", "VP_Accuracy30"]
        logable_metrics += ["VP_loss", "VP_Error", "VP_Accuracy30"]

    return logable_metrics, printable_metrics


"""
    Logs the losses and metrics; for a batch or an entire data split
"""


def depth_estimation_metrics(pred, gt, h=""):
    # Using metrics from https://arxiv.org/pdf/1805.01328.pdf
    metrics = {}
    metrics[h + "thresh_1.25"] = Metrics.pixel_treshold(pred, gt, 1.25)
    metrics[h + "abs_rel_diff"] = Metrics.absolute_rel_diff(pred, gt)
    metrics[h + "rms_linear"] = Metrics.rms(pred, gt)
    metrics[h + "rms_log"] = Metrics.log_rms(pred, gt)

    l1_loss = torch.nn.functional.l1_loss(pred, gt, reduction="none")
    mean_dims = tuple(range(1, len(l1_loss.shape)))
    l1_loss = l1_loss.mean(dim=mean_dims)
    l1_loss = _detach(l1_loss)
    metrics[h + "L1_diff"] = l1_loss
    return metrics


def mask_estimation_metrics(pred, gt, h=""):
    metrics = {}
    _iou, _recall, _precision, _accuracy = Metrics.roc(pred, gt)
    metrics[h + "IoU"] = _iou
    metrics[h + "Recall"] = _recall
    metrics[h + "Precision"] = _precision
    metrics[h + "F1"] = (2 * _precision * _recall) / (
        _precision + _recall
    ).clip(min=1e-6)
    metrics[h + "PixAcc"] = _accuracy
    return metrics


def _detach(x):
    return x.detach().cpu().numpy()


def reconstruction_metrics(pred, gt, recon_rep, header=""):
    if recon_rep == "depth":
        return depth_estimation_metrics(pred, gt, header)
    elif recon_rep == "mask":
        pred = pred.sigmoid()
        return mask_estimation_metrics(pred, gt, header)


def get_dense_rep(data_batch, dense_rep):
    def _half_channels(_img):
        return _img[:, 0 : _img.shape[1] / 2]

    if "rgb" in dense_rep:
        rgb_gt = data_batch[0].cuda()
        return rgb_gt
    elif "depth" in dense_rep:
        d_gt = data_batch[1].cuda()
        return d_gt


"""
    Given prediction and ground truth, calculate the losses

    @return loss    tensor of size 1 with the loss to be backproped
            losses  dictionary with all the losses calculated
"""


def evaluate_batch(pred, labels, args):
    # -- calculate losses --
    metrics = {}

    if args.reconstruction:
        if "reconstruction" in labels:
            recon_gt = labels["reconstruction"].cuda()
            recon_pr = pred["reconstruction"]
        else:
            recon_gt1 = labels["reconstruction_1"].cuda()
            recon_pr1 = pred["reconstruction_1"]
            recon_gt2 = labels["reconstruction_2"].cuda()
            recon_pr2 = pred["reconstruction_2"]
            recon_pr = torch.cat((recon_pr1, recon_pr2), dim=1)
            recon_gt = torch.cat((recon_gt1, recon_gt2), dim=1)

        if "reconstruction_projected" in labels:
            proj_gt = labels["reconstruction_projected"].cuda()
            proj_pr = pred["reconstruction_projected"]
        else:
            proj_gt = recon_gt
            proj_pr = recon_pr

        p_loss, p_loss_inst = reconstruction_loss(
            proj_pr, proj_gt, args.unet_output
        )
        r_loss, r_loss_inst = reconstruction_loss(
            recon_pr, recon_gt, args.unet_output
        )
        metrics.update(
            {"Recon_loss": r_loss_inst, "Proj_loss": p_loss_inst}
        )
        loss = p_loss

        # metrics
        r_metrics = reconstruction_metrics(recon_pr, recon_gt, args.unet_output)
        metrics.update(r_metrics)
        p_metrics = reconstruction_metrics(
            proj_pr, proj_gt, args.unet_output, "proj_"
        )
        metrics.update(p_metrics)

    if args.viewpoint:
        vp_pr = pred["viewpoint"]
        vp_gt = labels["viewpoint"]
        vp_pr_ext = extract_viewpoint(vp_pr, args.rot_rep)
        vp_error = get_vp_error(vp_pr_ext, vp_gt, args.rot_rep)

        # no loss -- just a placeholder
        loss = pred["viewpoint"].sum() * 0.0
        batch_size = pred["viewpoint"].shape[0]
        vp_loss_inst = np.zeros(batch_size)
        vp_loss = 0

        if "viewpoint_reference" in pred:
            vp_pr_1 = pred["viewpoint_reference"]
            vp_gt_1 = labels["viewpoint_reference"]
            vp_pr_1_ext = extract_viewpoint(vp_pr_1, args.rot_rep)
            vp_loss_1, vp_loss_inst_1 = viewpoint_loss(
                vp_pr_1, vp_gt_1, rot_rep=args.rot_rep
            )
            vp_error_1 = get_vp_error(vp_pr_1_ext, vp_gt_1, args.rot_rep)

            vp_loss = 0.5 * (vp_loss + vp_loss_1)
            vp_error = 0.5 * (vp_error + vp_error_1)
            vp_loss_inst = 0.5 * (vp_loss_inst + vp_loss_inst_1)

        metrics["VP_loss"] = vp_loss_inst
        metrics["VP_Error"] = vp_error
        metrics["VP_Accuracy30"] = vp_error <= 30.0
        loss = loss + vp_loss if args.reconstruction else vp_loss

    if args.realism_check:
        pr_mag = pred["realism_magnitude"].squeeze(dim=1)
        gt_mag = labels["realism_magnitude"].float().cuda()
        pr_eul = pred["realism_euler"]
        gt_eul = labels["realism_euler"].float().cuda()

        # corruption magnitude error and loss
        m_loss_inst = F.mse_loss(pr_mag, gt_mag, reduction="none")
        m_loss_inst, m_loss = _detach(m_loss_inst), m_loss_inst.mean()

        metrics["RC Mag Loss"] = m_loss_inst
        metrics["RC Mag Error"] = _detach((pr_mag - gt_mag).abs())

        # euler prediction -- error and loss
        e_loss_inst = F.l1_loss(pr_eul, gt_eul, reduction="none")
        e_loss_inst = _detach(e_loss_inst)

        metrics["RC Euler Loss"] = e_loss_inst.mean(axis=1)
        metrics["RC Azim Error"] = e_loss_inst[:, 0]
        metrics["RC Elev Error"] = e_loss_inst[:, 1]
        metrics["RC Tilt Error"] = e_loss_inst[:, 2]

        # Feel good metric -- accuracy (not reported in paper!)
        pr_np = _detach(pr_mag) > 0.3
        gt_np = _detach(gt_mag) > 0.0
        metrics["RC Accuracy"] = (pr_np == gt_np).astype(float)

        loss = m_loss

    for _key in pred.keys():
        if "loss" in _key:
            metrics[_key] = _detach(pred[_key])
            if args.consistency_loss:
                loss = loss + pred[_key].mean()

    # aggreagte mean
    metrics_mean = {}
    for m in metrics:
        metrics_mean[m] = metrics[m].mean()

    return loss, metrics_mean, metrics


def infer(model, data_batch, args):
    PR, GT = {}, {}
    # img, depth, fg, view, voxel, cls_num, u_id = instance
    if "3dRep" in args.model:
        assert args.reconstruction and not args.viewpoint
        if args.model_input == "rgb":
            _i_id = 0
        if args.model_input == "mask":
            _i_id = 2
        elif args.model_input == "depth":
            _i_id = 1

        if args.unet_output == "depth":
            _r_id = 1
        elif args.unet_output == "mask":
            _r_id = 2

        if args.n_views in [2, 3]:
            model_in = data_batch[_i_id].cuda()
            model_out = data_batch[_r_id]
            euler = data_batch[3].cuda()
            depth = data_batch[1].cuda()
        else:
            raise ValueError("Cannot handle multi views")

        if not args.depth_sculpt:
            depth = None

        PR = model(model_in, depth, euler)
        if args.n_views == 3:
            GT["reconstruction_1"] = model_out[:, 0]
            GT["reconstruction_2"] = model_out[:, 1]
        else:
            GT["reconstruction"] = model_out[:, 0]
        GT["reconstruction_projected"] = model_out[:, -1]
    elif "mvRealism" in args.model:
        assert args.n_views == 2
        if args.model_input == "rgb":
            _i_id = 0
        elif args.model_input == "mask":
            _i_id = 2
        elif args.model_input == "depth":
            _i_id = 1

        model_in = data_batch[_i_id].cuda()
        euler = data_batch[3].cuda()
        depth = data_batch[1].cuda() if args.depth_sculpt else None

        PR = model(model_in, depth, euler)
        GT["realism_magnitude"] = data_batch[7]
        GT["realism_euler"] = data_batch[8]
    return PR, GT
