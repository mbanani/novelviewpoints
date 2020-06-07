import argparse

from tqdm import tqdm

import torch
from datasets import get_loaders
from models import get_model
from util.util import Checkpoint
from util.train_utils import get_metrics, get_flags, infer, evaluate_batch
from util.result_dict import ResultDict

parser = argparse.ArgumentParser()
args = get_flags(parser, evaluate=True)


def main(args):
    global step, epoch, result_dict

    if args.checkpoint is not None:
        print("#############  Get Old Args   ##############")
        model, epoch, step, old_args = Checkpoint.restore(args.checkpoint)

        args.model = old_args.model
        args.model_input = old_args.model_input
        args.viewpoint = old_args.viewpoint
        args.rot_rep = old_args.rot_rep
        args.feat_dim = old_args.feat_dim
        args.voxel_dim = old_args.voxel_dim
        args.small_decoder = old_args.small_decoder
        args.reconstruction = old_args.reconstruction
        args.unet_output = old_args.unet_output
        args.depth_sculpt = old_args.depth_sculpt
        args.best_loss = old_args.best_loss
    else:
        model = get_model(
            args.model,
            args.rot_rep,
            args.model_input,
            args.unet_output,
            args.pretrained,
            args.no_refinement
        )
    print(args)

    loader = get_loaders(
        name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split=args.split,
        rot_rep=args.rot_rep,
        n_views=args.n_views,
        corrupt_vp=args.corrupt_vp,
    )

    loader.dataset.__getitem__(0)
    # initialize result dictionaries
    logable_metrics, printable_metrics = get_metrics(
        args.reconstruction, args.viewpoint, args.realism_check, args.unet_output,
    )
    result_dict = ResultDict(loader.dataset, logable_metrics, printable_metrics)

    # Train on GPU
    model.cuda()

    print("#############  Start Evaluation   ##############")
    eval_step(model, loader, args.split)

def eval_step(model, loader, split):
    global result_dict
    model.eval()
    result_dict.reset(0)
    epoch_loss = 0.0

    with torch.no_grad():
        for i, data_batch in enumerate(tqdm(loader)):

            preds, labels = infer(model, data_batch, args)
            loss, _, batch_metrics = evaluate_batch(preds, labels, args)
            epoch_loss += loss.item()
            result_dict.update(preds, data_batch[-1], batch_metrics, labels)

    # log results
    result_dict.calculate_summary_statistics()
    result_dict.log_performance(split, display=True)
    epoch_loss = epoch_loss / len(loader)

    return epoch_loss


if __name__ == "__main__":
    print(args)
    main(args)
