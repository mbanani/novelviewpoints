import torch.nn as nn


# CNN based models
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


# weigh initialization from torchvision/models/vgg.py
def initialize_weights(network):
    for m in network.modules():
        # add deconv to instances?
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                m.bias.data.zero_()
        elif (
            isinstance(m, nn.BatchNorm2d)
            or isinstance(m, nn.BatchNorm3d)
            or isinstance(m, nn.BatchNorm1d)
        ):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
            m.bias.data.zero_()
