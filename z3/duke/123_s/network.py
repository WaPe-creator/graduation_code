import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50, Bottleneck

num_classes = 702  # change this depend on your dataset


class M2N(nn.Module):
    def __init__(self):
        super(M2N, self).__init__()

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

        res_conv5 = resnet.layer4
        
        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        res_p_conv5.load_state_dict(resnet.layer4.state_dict())

        self.part1 = copy.deepcopy(res_conv5)
        self.part2 = copy.deepcopy(res_p_conv5)
        self.part3 = copy.deepcopy(res_p_conv5)
        
        self.avgpool_g = nn.AvgPool2d(kernel_size=(12, 4))
        self.reduction_g  = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats), nn.ReLU())
        self._init_reduction(self.reduction_g)
        self.fc_g = nn.Linear(feats, num_classes)
        self._init_fc(self.fc_g)

        #==================== block STN_2 ====================
        self.localization_2 = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(256, 16, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.fc_loc_2 = nn.Linear(16 * 3 * 1, 1)

        self.fc_loc_2.weight.data.zero_()
        self.fc_loc_2.bias.data.zero_()

        self.scale_factors_2 = []
        self.scale_factors_2.append(torch.tensor([[1, 0], [0, 1/2]], dtype=torch.float))
        self.scale_factors_2.append(torch.tensor([[1, 0], [0, 1/2]], dtype=torch.float))

        #==================== block STN_3 ====================
        self.localization_3 = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(256, 16, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.fc_loc_3 = nn.Linear(16 * 3 * 1, 1)

        self.fc_loc_3.weight.data.zero_()
        self.fc_loc_3.bias.data.zero_()

        self.scale_factors_3 = []
        self.scale_factors_3.append(torch.tensor([[1, 0], [0, 1/3]], dtype=torch.float))
        self.scale_factors_3.append(torch.tensor([[1, 0], [0, 1/3]], dtype=torch.float))
        self.scale_factors_3.append(torch.tensor([[1, 0], [0, 1/3]], dtype=torch.float))
        
        #==================== local region feature ====================
        self.maxpool_2_1 = nn.MaxPool2d(kernel_size=(12, 8))
        self.maxpool_2_2 = nn.MaxPool2d(kernel_size=(12, 8))
        self.maxpool_3_1 = nn.MaxPool2d(kernel_size=(8, 8))
        self.maxpool_3_2 = nn.MaxPool2d(kernel_size=(8, 8))
        self.maxpool_3_3 = nn.MaxPool2d(kernel_size=(8, 8))

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


    # Spatial transformer network forward function
    #==================== STN ====================
    def transform_theta_2(self, theta_i, region):
        scale_factors = self.scale_factors_2[region]
        theta = torch.zeros(theta_i.size(0), 2, 3)
        theta[:,:,:2] = scale_factors
        theta[:,-1,-1] = theta_i.squeeze(1)
        for n in range(theta_i.size(0)):
            if theta[n,-1,-1]+theta[n,-1,-2]>1:
                theta[n,-1,-1] = 1-theta[n,-1,-2]
            if theta[n,-1,-1]-theta[n,-1,-2]<-1:
                theta[n,-1,-1] = -1+theta[n,-1,-2]
        theta = theta.cuda()
        return theta

    def transform_theta_3(self, theta_i, region):
        scale_factors = self.scale_factors_3[region]
        theta = torch.zeros(theta_i.size(0), 2, 3)
        theta[:,:,:2] = scale_factors
        theta[:,-1,-1] = theta_i.squeeze(1)
        for n in range(theta_i.size(0)):
            if theta[n,-1,-1]+theta[n,-1,-2]>1:
                theta[n,-1,-1] = 1-theta[n,-1,-2]
            if theta[n,-1,-1]-theta[n,-1,-2]<-1:
                theta[n,-1,-1] = -1+theta[n,-1,-2]
        theta = theta.cuda()
        return theta  

    def stn(self, x, theta, num):
        h = x.size(2)
        if num==2:
            s = x[:,:,0:int(h/2),:].size()
        if num==3:
            s = x[:,:,0:int(h/3),:].size()
        grid = F.affine_grid(theta, s)
        x = F.grid_sample(x, grid)
        return x


    def forward(self, x):
        x = self.backbone(x)

        xg = self.part1(x)
        pg = self.avgpool_g(xg)
        pg = self.reduction_g(pg).squeeze(dim=3).squeeze(dim=2)
        fg = self.fc_g(pg)


        xp2 = self.part2(x)
        #==================== block STN_2 ====================
        region_2 = []
        xr2 = self.localization_2(xp2)
        xr2 = xr2.view(-1, 16 * 3 * 1)
        theta2 = self.fc_loc_2(xr2)
        for region in range(2):
            if region == 0:
                theta2[:,].add_(-1/2)
                theta2_i = self.transform_theta_2(theta2, region)
            if region ==1:
                theta2[:,].add_(1)
                theta2_i = self.transform_theta_2(theta2, region)

            p2_i = self.stn(xp2, theta2_i, 2)
            region_2.append(p2_i)

        xp3 = self.part3(x)
        #==================== block STN_3 ====================
        region_3 = []
        xr3 = self.localization_3(xp3)
        xr3 = xr3.view(-1, 16 * 3 * 1)
        theta3 = self.fc_loc_3(xr3)
        for region in range(3):
            if region == 0:
                theta3[:,].add_(-2/3)
                theta3_i = self.transform_theta_3(theta3, region)
            if region ==1:
                theta3[:,].add_(2/3)
                theta3_i = self.transform_theta_3(theta3, region)
            if region == 2:
                theta3[:,].add_(2/3)
                theta3_i = self.transform_theta_3(theta3, region)

            p3_i = self.stn(xp3, theta3_i, 3)
            region_3.append(p3_i)


        #==================== local region feature ====================
        region_2[0] = self.maxpool_2_1(region_2[0])
        region_2[1] = self.maxpool_2_2(region_2[1])
        region_3[0] = self.maxpool_3_1(region_3[0])
        region_3[1] = self.maxpool_3_2(region_3[1])
        region_3[2] = self.maxpool_3_3(region_3[2])

        region_2[0] = self.reduction_2_1(region_2[0]).squeeze(dim=3).squeeze(dim=2)
        region_2[1] = self.reduction_2_2(region_2[1]).squeeze(dim=3).squeeze(dim=2)
        region_3[0] = self.reduction_3_1(region_3[0]).squeeze(dim=3).squeeze(dim=2)
        region_3[1] = self.reduction_3_2(region_3[1]).squeeze(dim=3).squeeze(dim=2)
        region_3[2] = self.reduction_3_3(region_3[2]).squeeze(dim=3).squeeze(dim=2)

        f2_1 = self.fc_2_1(region_2[0])
        f2_2 = self.fc_2_2(region_2[1])
        f3_1 = self.fc_3_1(region_3[0])
        f3_2 = self.fc_3_2(region_3[1])
        f3_3 = self.fc_3_3(region_3[2])

        pr2 = torch.cat([region_2[0], region_2[1]], dim=1)
        pr3 = torch.cat([region_3[0], region_3[1], region_3[2]], dim=1)

        predict = torch.cat([pg, pr2, pr3], dim=1)

        return predict, pg, fg, f2_1, f2_2, f3_1, f3_2, f3_3
