import torch
from torch import nn
from torch.nn import functional as F


class MultiLayerFusionBlock(nn.Module):
    def __init__(self, trans_channels, in_channels, inter_channels=None, sub_sample=False, bn_layer=True):
        super(MultiLayerFusionBlock, self).__init__()

        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.conv1x1 = nn.Sequential(nn.Conv2d(trans_channels, in_channels, 1, bias=False), nn.BatchNorm2d(in_channels), nn.ReLU())

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,kernel_size=1, stride=1, padding=0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, nn.MaxPool2d(kernel_size=(2, 2)))
            self.phi = nn.Sequential(self.phi, nn.MaxPool2d(kernel_size=(2, 2)))

        if bn_layer:
            self.W = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)



    def forward(self, x, x_end):
        '''
        :param x: (n, c, h, w)
        :return:
        '''

        batch_size = x_end.size(0)

        x = self.conv1x1(x)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)    # (n, c, M)
        g_x = g_x.permute(0, 2, 1)                                   # (n, M, c)

        theta_x = self.theta(x_end).view(batch_size, self.inter_channels, -1)   # (n, c, N)
        theta_x = theta_x.permute(0, 2, 1)                                  # (n, N, c)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)       # (n, c, M)
        f = torch.matmul(theta_x, phi_x)                                # (n, N, M)
        f_div = F.softmax(f, dim=-1)

        y = torch.matmul(f_div, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x_end.size()[2:])
        W_y = self.W(y)
        # z = W_y + x

        return W_y