import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50, Bottleneck
from attn import MultiLayerFusionBlock

num_classes = 702  # change this depend on your dataset


class resnet_mlf(nn.Module):
    def __init__(self):
        super(resnet_mlf, self).__init__()

        # feats = 256
        resnet = resnet50(pretrained=True)

        self.backbone1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
        )

        self.backbone2 = resnet.layer2
        self.backbone3 = resnet.layer3
        self.backbone4 = resnet.layer4

        self.mlf1_2 = MultiLayerFusionBlock(256, 512)
        self.mlf2_3 = MultiLayerFusionBlock(512, 1024)
        self.mlf3_4 = MultiLayerFusionBlock(1024, 2048)

        self.avgpool = nn.AvgPool2d(kernel_size=(12, 4))

        self.reduction_g  = nn.Sequential(nn.Conv2d(2048, 256, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU())
        self._init_reduction(self.reduction_g)
        self.fc_g = nn.Linear(256, num_classes)
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
        x1 = self.backbone1(x)
        x2 = self.backbone2(x1)
        m1_2 = self.mlf1_2(x1, x2)
        
        x3 = self.backbone3(m1_2)
        m2_3 = self.mlf2_3(m1_2, x3)
        
        x4 = self.backbone4(m2_3)
        m3_4 = self.mlf3_4(m2_3, x4)

        x = self.avgpool(m3_4)    

        p1 = self.reduction_g(x).squeeze(dim=3).squeeze(dim=2)
        f1 = self.fc_g(p1)

        return p1, p1, f1
