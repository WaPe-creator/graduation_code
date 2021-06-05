import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50, Bottleneck

num_classes = 751  # change this depend on your dataset


class resnet_mlf(nn.Module):
    def __init__(self):
        super(resnet_mlf, self).__init__()

        feats = 256
        resnet = resnet50(pretrained=True)

        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        self.avgpool_g = nn.AvgPool2d(kernel_size=(12, 4))
        self.reduction_g  = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats), nn.ReLU())
        self._init_reduction(self.reduction_g)
        self.fc_g = nn.Linear(feats, num_classes)
        self._init_fc(self.fc_g)

    @staticmethod
    def _init_reduction(reduction):
        # conv
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')
        # nn.init.constant_(reduction[0].bias, 0.)

        # bn
        nn.init.normal_(reduction[1].weight, mean=1., std=0.02)
        nn.init.constant_(reduction[1].bias, 0.)

    @staticmethod
    def _init_fc(fc):
        nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        # nn.init.normal_(fc.weight, std=0.001)
        nn.init.constant_(fc.bias, 0.)



    def forward(self, x):
        x = self.backbone(x)
 
        p1 = self.avgpool_g(x)
        p1 = self.reduction_g(p1).squeeze(dim=3).squeeze(dim=2)
        f1 = self.fc_g(p1)

        return p1, p1, f1
