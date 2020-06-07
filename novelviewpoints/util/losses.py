import torch.nn.functional as F


def depth_variance_loss(pred, gt):
    diff = pred - gt.log()  # batch x 1 x N x N

    # Var(d) = E(d^2) - E(d)^2 -- all dims are (batch,  )
    squared_expectation = (diff ** 2).mean(dim=(1, 2, 3))  # E(d^2)
    expectation_squared = diff.mean(dim=(1, 2, 3)) ** 2  # E(d) ^ 2
    variance = squared_expectation - expectation_squared

    # Depth gradients
    diff_x = diff[:, :, :, 1:] - diff[:, :, :, :-1]
    diff_y = diff[:, :, :, 1:] - diff[:, :, :, :-1]

    diff_grad_2 = (diff_x ** 2) + (diff_y ** 2)
    diff_grad_2 = diff_grad_2.mean(dim=(1, 2, 3))

    # loss
    loss = variance + diff_grad_2
    return loss.mean(), loss.detach().cpu().numpy()


