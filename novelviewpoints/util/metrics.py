import numpy as np

import torch


def _detach(t):
    return t.detach().cpu().numpy()


# -- depth measures
class Metrics:
    def pixel_treshold(pred, gt, threshold):
        # clip so that nothing is 0
        ratio = torch.cat((pred / gt.clamp(1e-9), gt / pred.clamp(1e-9)), 1)
        ratio = ratio.max(dim=1)[0] < threshold
        ratio = ratio.float().mean(dim=(1, 2))
        return _detach(ratio)

    def absolute_rel_diff(pred, gt):
        diff = ((pred - gt).abs() / gt).mean(dim=(1, 2, 3))
        diff = _detach(diff)
        if np.isnan(diff).any():
            raise ValueError()
        return diff

    def square_rel_diff(pred, gt):
        diff = ((pred - gt).pow(2) / gt).mean(dim=(1, 2, 3))
        return _detach(diff)

    def rms(pred, gt):
        diff = (pred - gt).pow(2).mean(dim=(1, 2, 3))
        diff = diff.sqrt()
        return _detach(diff)

    def log_rms(pred, gt):
        return Metrics.rms(pred.clamp(1e-9).log(), gt.clamp(1e-9).log())

    # -- mask methods
    #   thresh is the value for determining if classification is true; good for ROC
    def roc(pred, gt, thresh=0.5):
        pred = (pred.float() >= thresh).byte()
        gt = gt.byte()

        acc = (pred == gt).float().mean(dim=(1, 2, 3))
        inter = (pred & gt).float().sum(dim=(1, 2, 3)).float()
        union = (pred | gt).float().sum(dim=(1, 2, 3)).float()
        gt_sum = gt.sum(dim=(1, 2, 3)).float()
        pred_sum = pred.sum(dim=(1, 2, 3)).float()

        use_variation = False
        if use_variation:
            min_val = 100
            inter[inter <= min_val] = 0.0
            union[union <= min_val] = 0.0
            gt_sum[gt_sum <= min_val] = 0.0
            pred_sum[pred_sum <= min_val] = 0.0

        recall = inter.clamp(min=1e-9) / gt_sum.clamp(min=1e-9)
        precision = inter.clamp(min=1e-9) / pred_sum.clamp(min=1e-9)
        iou = inter.clamp(min=1e-9) / union.clamp(min=1e-9)

        return _detach(iou), _detach(recall), _detach(precision), _detach(acc)
