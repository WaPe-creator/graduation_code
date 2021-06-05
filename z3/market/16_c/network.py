import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50, Bottleneck
# from attn import MultiLayerFusionBlock

num_classes = 751  # change this depend on your dataset


class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()

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

        # branch: part1->global, part2->local
        self.part1 = copy.deepcopy(res_conv5)
        self.part2 = copy.deepcopy(res_p_conv5)

        # global feature
        self.avgpool_g = nn.AvgPool2d(kernel_size=(12, 4))
        self.reduction_g  = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats), nn.ReLU())
        self._init_reduction(self.reduction_g)
        self.fc_g = nn.Linear(feats, num_classes)
        self._init_fc(self.fc_g)

        
        # local features
        self.maxpool_p = nn.MaxPool2d(kernel_size=(4, 8))

        self.reduction_1 = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats), nn.ReLU())
        self.reduction_2 = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats), nn.ReLU())
        self.reduction_3 = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats), nn.ReLU())
        self.reduction_4 = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats), nn.ReLU())
        self.reduction_5 = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats), nn.ReLU())
        self.reduction_6 = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats), nn.ReLU())
        self._init_reduction(self.reduction_1)
        self._init_reduction(self.reduction_2)
        self._init_reduction(self.reduction_3)
        self._init_reduction(self.reduction_4)
        self._init_reduction(self.reduction_5)
        self._init_reduction(self.reduction_6)

        self.fc_1 = nn.Linear(feats, num_classes)
        self.fc_2 = nn.Linear(feats, num_classes)
        self.fc_3 = nn.Linear(feats, num_classes)
        self.fc_4 = nn.Linear(feats, num_classes)
        self.fc_5 = nn.Linear(feats, num_classes)
        self.fc_6 = nn.Linear(feats, num_classes)
        self._init_fc(self.fc_1)
        self._init_fc(self.fc_2)
        self._init_fc(self.fc_3)
        self._init_fc(self.fc_4)
        self._init_fc(self.fc_5)
        self._init_fc(self.fc_6)

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
        xp = self.part2(x)
        pt = self.maxpool_p(xp)
        p1 = pt[:,:,0:1,:]
        p2 = pt[:,:,1:2,:]
        p3 = pt[:,:,2:3,:]
        p4 = pt[:,:,3:4,:]
        p5 = pt[:,:,4:5,:]
        p6 = pt[:,:,5:,:]

        p_1 = self.reduction_1(p1).squeeze(dim=3).squeeze(dim=2)
        p_2 = self.reduction_2(p2).squeeze(dim=3).squeeze(dim=2)
        p_3 = self.reduction_3(p3).squeeze(dim=3).squeeze(dim=2)
        p_4 = self.reduction_4(p4).squeeze(dim=3).squeeze(dim=2)
        p_5 = self.reduction_5(p5).squeeze(dim=3).squeeze(dim=2)
        p_6 = self.reduction_6(p6).squeeze(dim=3).squeeze(dim=2)

        f_1 = self.fc_1(p_1)
        f_2 = self.fc_2(p_2)
        f_3 = self.fc_3(p_3)
        f_4 = self.fc_4(p_4)
        f_5 = self.fc_5(p_5)
        f_6 = self.fc_6(p_6)
        
        pr = torch.cat([p_1, p_2, p_3, p_4, p_5, p_6], dim=1)

        predict = torch.cat([pg, pr], dim=1)

        return predict, pg, pr, fg, f_1, f_2, f_3, f_4, f_5, f_6
