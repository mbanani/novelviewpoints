import os

import torch
from models import get_model


def makedir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except:
            print("Could not create path below as it already exists")
            print("\t", path)


def get_gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1.0 / 2)
    return total_norm


class Checkpoint:
    def save(experiment_path, model, curr_epoch, curr_loss, args, step=None):
        """
        Saves a checkpoint and updates the best loss and best weighted accuracy
        """
        model_state = (
            model.module.cpu().state_dict()
            if len(args.gpu) > 2
            else model.cpu().state_dict()
        )

        state = {
            "epoch": curr_epoch,
            "step": step,
            "curr_loss": curr_loss,
            "state_dict": model_state,
            "args": args,
        }

        model.cuda()
        if step:
            path = os.path.join(
                experiment_path,
                "checkpoint@epoch{:03d}step{}.pkl".format(curr_epoch, step),
            )
        else:
            path = os.path.join(
                experiment_path,
                "checkpoint@epoch{:03d}.pkl".format(curr_epoch)
            )
        torch.save(state, path)
        print("Checkpoint saved at {}".format(path))

        if curr_loss < args.best_loss:
            path = os.path.join(experiment_path, "best_loss.pkl")
            torch.save(state, path)

    def restore(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        epoch = checkpoint["epoch"]
        step = checkpoint["step"]
        curr_loss = checkpoint["curr_loss"]
        args = checkpoint["args"]

        print("Loading a model :", args.model)

        model = get_model(
            args.model,
            args.rot_rep,
            args.model_input,
            args.unet_output,
            args.pretrained,
            args.no_refinement,
        )

        try:
            model.load_state_dict(checkpoint["state_dict"])
        except RuntimeError as R:
            print("Missing Keys? \n {}".format(R))
            model.load_state_dict(checkpoint["state_dict"], strict=False)

        if len(args.gpu) > 2:
            model = torch.nn.DataParallel(model)
        print(
            "Loaded pretrained model from:\n\t path: {} \n\t loss: {}".format(
                checkpoint_path, curr_loss
            )
        )

        return model, epoch, step, args
