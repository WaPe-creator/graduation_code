import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50, Bottleneck
from attn import MSA

num_classes = 751  # change this depend on your dataset


class AttNet(nn.Module):
    def __init__(self):
        super(AttNet, self).__init__()

        feats = 256
        resnet = resnet50(pretrained=True)

        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            #resnet.layer3,
        )

        res_conv4 = resnet.layer3
        res_conv5 = resnet.layer4
        
        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        res_p_conv5.load_state_dict(resnet.layer4.state_dict())

        self.msa = MSA(512)

        self.part_nl = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_conv5))
        self.part_sp = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_conv5))

        self.part2 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        self.part3 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))

        self.avgpool_nl = nn.AvgPool2d(kernel_size=(12, 4))
        self.avgpool_sp = nn.AvgPool2d(kernel_size=(12, 4))

        self.reduction_g0 = nn.Sequential(nn.Conv2d(4096, 2048, 1, bias=False), nn.BatchNorm2d(2048), nn.ReLU())
        self._init_reduction(self.reduction_g0)
        self.reduction_g1 = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats), nn.ReLU())
        self._init_reduction(self.reduction_g1)
        self.fc_g = nn.Linear(feats, num_classes)
        self._init_fc(self.fc_g)
        
        self.reduction_nl = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats), nn.ReLU())
        self._init_reduction(self.reduction_nl)
        self.fc_nl = nn.Linear(feats, num_classes)
        self._init_fc(self.fc_nl)
        
        self.reduction_sp = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats), nn.ReLU())
        self._init_reduction(self.reduction_sp)
        self.fc_sp = nn.Linear(feats, num_classes)
        self._init_fc(self.fc_sp)

        
        #==================== local region feature ====================
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

        nl_x, sp_x = self.msa(x)

        xnl = self.part_nl(nl_x)
        xsp = self.part_sp(sp_x)

        xnl = self.avgpool_nl(xnl)
        xsp = self.avgpool_sp(xsp)
        
        pnl = self.reduction_nl(xnl).squeeze(dim=3).squeeze(dim=2)
        fnl = self.fc_nl(pnl)
        
        psp = self.reduction_sp(xsp).squeeze(dim=3).squeeze(dim=2)
        fsp = self.fc_sp(psp)

        xg = torch.cat([xnl, xsp], dim=1)
        pg = self.reduction_g1(self.reduction_g0(xg)).squeeze(dim=3).squeeze(dim=2)
        fg = self.fc_g(pg)

        #====================  ====================

        xp2 = self.part2(sp_x)
        xp2 = self.maxpool_2(xp2)
        p2_1 = xp2[:,:,0:1,:]
        p2_2 = xp2[:,:,1:, :]

        xp3 = self.part3(sp_x)
        xp3 = self.maxpool_3(xp3)
        p3_1 = xp3[:,:,0:1,:]
        p3_2 = xp3[:,:,1:2,:]
        p3_3 = xp3[:,:,2:, :]
    
        p2_1 = self.reduction_2_1(p2_1).squeeze(dim=3).squeeze(dim=2)
        p2_2 = self.reduction_2_2(p2_2).squeeze(dim=3).squeeze(dim=2)
        p3_1 = self.reduction_3_1(p3_1).squeeze(dim=3).squeeze(dim=2)
        p3_2 = self.reduction_3_2(p3_2).squeeze(dim=3).squeeze(dim=2)
        p3_3 = self.reduction_3_3(p3_3).squeeze(dim=3).squeeze(dim=2)

        f2_1 = self.fc_2_1(p2_1)
        f2_2 = self.fc_2_2(p2_2)
        f3_1 = self.fc_3_1(p3_1)
        f3_2 = self.fc_3_2(p3_2)
        f3_3 = self.fc_3_3(p3_3)

        pr2 = torch.cat([p2_1, p2_2], dim=1)
        pr3 = torch.cat([p3_1, p3_2, p3_3], dim=1)

        predict = torch.cat([pnl, pg, psp, pr2, pr3], dim=1)

        return predict, pnl, pg, psp, pr2, pr3, fnl, fg, fsp, f2_1, f2_2, f3_1, f3_2, f3_3
