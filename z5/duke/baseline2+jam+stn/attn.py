import torch
from torch import nn
from torch.nn import functional as F
import math


class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NonLocalBlock, self).__init__()

        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

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

    def forward(self, x):
        '''
        :param x: (n, c, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)    # (n, c, h*w), if sub_sample: (n, c, h/2*w/2)
        g_x = g_x.permute(0, 2, 1)                                   # (n, h*w, c), if sub_sample: (n, h/2*w/2, c)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)   # (n, c, h*w)
        theta_x = theta_x.permute(0, 2, 1)                                  # (n, h*w, c)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)       # (n, c, h*w), if sub_sample: (n, c, h/2*w/2)
        f = torch.matmul(theta_x, phi_x)                                # (n, h*w, h*w), if sub_sample: (n, h*w, h/2*w/2)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

class ChannelAttention(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        #self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        #self.relu1 = nn.ReLU()
        #self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        
        #t = int(abs((math.log(channels, 2) + b) / gamma))
        #k_size = t if t % 2 else t + 1
        k_size = 7
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.conv(self.avg_pool(x).squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        max_out = self.conv(self.max_pool(x).squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class JAM(nn.Module):
    def __init__(self, in_channels):
        super(JAM, self).__init__()
        self.nonlocal_attn = NonLocalBlock(in_channels)
        self.channel_attn = ChannelAttention(in_channels)
        self.spatial_attn = SpatialAttention()
        self.conv = nn.Conv2d(in_channels, in_channels, 1, 1, 0)

    def forward(self, x):

        ch_x = self.channel_attn(x)

        x = x * ch_x

        nl_x = self.nonlocal_attn(x)

        sp_x = self.spatial_attn(x) * x

        return nl_x, sp_x