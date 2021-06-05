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
        self.localization = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(256, 16, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.fc_loc = nn.Linear(16 * 3 * 1, 1)

        self.fc_loc.weight.data.zero_()
        self.fc_loc.bias.data.zero_()

        self.scale_factors = []
        self.scale_factors.append(torch.tensor([[1, 0], [0, 1/6]], dtype=torch.float))
        self.scale_factors.append(torch.tensor([[1, 0], [0, 1/6]], dtype=torch.float))
        self.scale_factors.append(torch.tensor([[1, 0], [0, 1/6]], dtype=torch.float))
        self.scale_factors.append(torch.tensor([[1, 0], [0, 1/6]], dtype=torch.float))
        self.scale_factors.append(torch.tensor([[1, 0], [0, 1/6]], dtype=torch.float))
        self.scale_factors.append(torch.tensor([[1, 0], [0, 1/6]], dtype=torch.float))
        
        #==================== local region feature ====================
        self.maxpool_1 = nn.MaxPool2d(kernel_size=(4, 8))
        self.maxpool_2 = nn.MaxPool2d(kernel_size=(4, 8))
        self.maxpool_3 = nn.MaxPool2d(kernel_size=(4, 8))
        self.maxpool_4 = nn.MaxPool2d(kernel_size=(4, 8))
        self.maxpool_5 = nn.MaxPool2d(kernel_size=(4, 8))
        self.maxpool_6 = nn.MaxPool2d(kernel_size=(4, 8))

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


    # Spatial transformer network forward function
    #==================== STN ====================
    def transform_theta(self, theta_i, region):
        scale_factors = self.scale_factors[region]
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
    
    def transform_theta_(self, theta_i, region):
        scale_factors = self.scale_factors[region]
        theta = torch.zeros(theta_i.size(0), 2, 3)
        theta[:,:,:2] = scale_factors
        theta[:,-1,-1] = theta_i.squeeze(1)
        theta = theta.cuda()
        return theta 
 

    def stn(self, x, theta):
        h = x.size(2)
        s = x[:,:,0:int(h/6),:].size()
        grid = F.affine_grid(theta, s)
        x = F.grid_sample(x, grid)
        return x  


    def forward(self, x):
        x = self.backbone(x)

        # global feature
        xg = self.part1(x)
        pg = self.avgpool_g(xg)
        pg = self.reduction_g(pg).squeeze(dim=3).squeeze(dim=2)
        fg = self.fc_g(pg)


        # align partition

        xp = self.part2(x)
        #==================== block STN_2 ====================
        region_ = []
        xr = self.localization(xp)
        xr = xr.view(-1, 16 * 3 * 1)
        theta = self.fc_loc(xr)
        for region in range(6):
            if region == 0:
                theta[:,].add_(-5/6)
                theta_i = self.transform_theta(theta, region)
            if region ==1:
                theta[:,].add_(2/6)
                theta_i = self.transform_theta(theta, region)
            if region ==2:
                theta[:,].add_(2/6)
                theta_i = self.transform_theta(theta, region)
            if region ==3:
                theta[:,].add_(2/6)
                theta_i = self.transform_theta(theta, region)
            if region ==4:
                theta[:,].add_(2/6)
                theta_i = self.transform_theta(theta, region)
            if region ==5:
                theta[:,].add_(2/6)
                theta_i = self.transform_theta(theta, region)

            p_i = self.stn(xp, theta_i)
            region_.append(p_i)


        # local features
        region_[0] = self.maxpool_1(region_[0])
        region_[1] = self.maxpool_2(region_[1])
        region_[2] = self.maxpool_3(region_[2])
        region_[3] = self.maxpool_4(region_[3])
        region_[4] = self.maxpool_5(region_[4])
        region_[5] = self.maxpool_6(region_[5])

        region_[0] = self.reduction_1(region_[0]).squeeze(dim=3).squeeze(dim=2)
        region_[1] = self.reduction_2(region_[1]).squeeze(dim=3).squeeze(dim=2)
        region_[2] = self.reduction_3(region_[2]).squeeze(dim=3).squeeze(dim=2)
        region_[3] = self.reduction_4(region_[3]).squeeze(dim=3).squeeze(dim=2)
        region_[4] = self.reduction_5(region_[4]).squeeze(dim=3).squeeze(dim=2)
        region_[5] = self.reduction_6(region_[5]).squeeze(dim=3).squeeze(dim=2)

        f_1 = self.fc_1(region_[0])
        f_2 = self.fc_2(region_[1])
        f_3 = self.fc_3(region_[2])
        f_4 = self.fc_4(region_[3])
        f_5 = self.fc_5(region_[4])
        f_6 = self.fc_6(region_[5])

        pr = torch.cat([region_[0], region_[1], region_[2], region_[3], region_[4], region_[5]], dim=1)

        predict = torch.cat([pg, pr], dim=1)

        return predict, pg, pr, fg, f_1, f_2, f_3, f_4, f_5, f_6
