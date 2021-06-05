import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50, Bottleneck
# from attn import MultiLayerFusionBlock

num_classes = 751  # change this depend on your dataset


class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()

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
        )

        # self.backbone2 = resnet.layer2
        # self.backbone3 = resnet.layer3

        res_conv5 = resnet.layer4
        
        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        res_p_conv5.load_state_dict(resnet.layer4.state_dict())

        # branch: part1->global, part2->local_2, part3->local_3
        self.part1 = copy.deepcopy(res_conv5)
        self.part2 = copy.deepcopy(res_p_conv5)
        self.part3 = copy.deepcopy(res_p_conv5)

        # global feature
        self.avgpool_g = nn.AvgPool2d(kernel_size=(12, 4))
        self.reduction_g  = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats), nn.ReLU())
        self._init_reduction(self.reduction_g)
        self.fc_g = nn.Linear(feats, num_classes)
        self._init_fc(self.fc_g)
        
        # local features
        self.maxpool_2 = nn.MaxPool2d(kernel_size=(12, 8))
        self.maxpool_3 = nn.MaxPool2d(kernel_size=(8, 8))

        self.reduction_2_1 = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats), nn.ReLU())
        self.reduction_2_2 = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats), nn.ReLU())
        self.reduction_3_1 = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats), nn.ReLU())
        self.reduction_3_2 = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats), nn.ReLU())
        self.reduction_3_3 = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats), nn.ReLU())
        self._init_reduction(self.reduction_2_1)
        self._init_reduction(self.reduction_2_2)
        self._init_reduction(self.reduction_3_1)
        self._init_reduction(self.reduction_3_2)
        self._init_reduction(self.reduction_3_3)

        self.fc_2_1 = nn.Linear(feats, num_classes)
        self.fc_2_2 = nn.Linear(feats, num_classes)
        self.fc_3_1 = nn.Linear(feats, num_classes)
        self.fc_3_2 = nn.Linear(feats, num_classes)
        self.fc_3_3 = nn.Linear(feats, num_classes)
        self._init_fc(self.fc_2_1)
        self._init_fc(self.fc_2_2)
        self._init_fc(self.fc_3_1)
        self._init_fc(self.fc_3_2)
        self._init_fc(self.fc_3_3)


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

        # global feature
        xg = self.part1(x)
        pg = self.avgpool_g(xg)
        pg = self.reduction_g(pg).squeeze(dim=3).squeeze(dim=2)
        fg = self.fc_g(pg)


        # local features

        xp2 = self.part2(x)
        xp2 = self.maxpool_2(xp2)
        p2_1 = xp2[:,:,0:1,:]
        p2_2 = xp2[:,:,1:2,:]
        p2_1 = self.reduction_2_1(p2_1).squeeze(dim=3).squeeze(dim=2)
        p2_2 = self.reduction_2_2(p2_2).squeeze(dim=3).squeeze(dim=2)
        f2_1 = self.fc_2_1(p2_1)
        f2_2 = self.fc_2_2(p2_2)


        xp3 = self.part3(x)
        xp3 = self.maxpool_3(xp3)
        p3_1 = xp3[:,:,0:1,:]
        p3_2 = xp3[:,:,1:2,:]
        p3_3 = xp3[:,:,2:3,:]
        p3_1 = self.reduction_3_1(p3_1).squeeze(dim=3).squeeze(dim=2)
        p3_2 = self.reduction_3_2(p3_2).squeeze(dim=3).squeeze(dim=2)
        p3_3 = self.reduction_3_3(p3_3).squeeze(dim=3).squeeze(dim=2)
        f3_1 = self.fc_3_1(p3_1)
        f3_2 = self.fc_3_2(p3_2)
        f3_3 = self.fc_3_3(p3_3)


        pr2 = torch.cat([p2_1, p2_2], dim=1)
        pr3 = torch.cat([p3_1, p3_2, p3_3], dim=1)

        predict = torch.cat([pg, pr2, pr3], dim=1)

        return predict, pg, pr2, pr3, fg, f2_1, f2_2, f3_1, f3_2, f3_3
