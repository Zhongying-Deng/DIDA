import torch
import torch.nn as nn
import torch.nn.functional as F


class DIDAModule(nn.Module):
    def __init__(self, k, inplanes, planes, m=4, padding=None, stride=1):
        """The input argument m is the channel reduction rate"""
        super(DIDAModule, self).__init__()
        self.k = k
        self.channel = inplanes 
        self.group = self.channel // m
        self.conv = nn.Conv2d(self.channel, self.channel // m, 1, padding=0, bias=True)
        self.pad = padding
        self.stride = stride
        
        self.conv_k = nn.Conv2d(1, 1, 1, padding=0, bias=True)
        self.conv_k2 = nn.Conv2d(1, 1, 1, padding=0, bias=True)
        self.conv_kernel = nn.Conv2d(1, k*k, 1, padding=0, bias=True)
        self.conv_kernel2 = nn.Conv2d(1, 3*3, 1, padding=0, bias=True)
        self.fuse = nn.Conv2d(self.channel // m, planes // 2, 1, padding=0, bias=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        N, C, H, W = x.shape
        # [N * C/m * H * W]
        f = F.relu(self.conv(x))
        # [N * C/m * 1 * 1]
        g = F.relu(self.conv(self.avg_pool(x))) 
        # [N * 1 * C/m * 1]
        g_perm = g.permute(0, 2, 1, 3).contiguous()
        # [N * k^2 * C/m * 1]
        kernel = self.conv_kernel(g_perm)
        # [N * 1 * C/m * k^2]
        kernel = kernel.permute(0, 3, 2, 1)
        # [N * 3^2 * C/m * 1]
        kernel_atrous = self.conv_kernel2(g_perm)
        # [N * 1 * C/m * 3^2]
        kernel_atrous = kernel_atrous.permute(0, 3, 2, 1)

        # [1 * C/m * H * W]
        f_list = torch.split(f, 1, 0)
        g_list = torch.split(kernel, 1, 0)
        g_list_atrous = torch.split(kernel_atrous, 1, 0)

        out = []
        out_atrous = []
        for i in range(N):
            # [1* C/m * H * W]
            f_one = f_list[i]
            # [1 * 1 * C/m * k^2]
            g_one = g_list[i]
            g_one_atrous  = g_list_atrous[i]

            # [1 * 1 * C/m * k^2]
            g_k = self.conv_k(g_one)
            # [C/m * 1 * k * k]
            g_k = g_k.reshape(g_k.size(2), g_k.size(1), self.k, self.k)
            g_k_atrous = self.conv_k2(g_one_atrous)
            # [C/m * 1 * 3 * 3]
            g_k_atrous = g_k_atrous.reshape(g_k_atrous.size(2), g_k_atrous.size(1), 3, 3)

            # [1* C/m * H * W]
            if self.pad is None:
                padding = ((self.k-1) // 2, (self.k-1) // 2, (self.k-1) // 2, (self.k-1) // 2)
            else:
                padding = (self.pad, self.pad, self.pad, self.pad)
            x = F.pad(input=f_one, pad=padding, mode='constant', value=0)
            o = F.conv2d(input=x, weight=g_k, stride=self.stride, groups=self.group)
            out.append(o)
            x = F.pad(input=f_one, pad=(2,2,2,2), mode='constant', value=0)
            o_atrous = F.conv2d(input=x, weight=g_k_atrous, stride=self.stride, dilation=2, groups=self.group)
            out_atrous.append(o_atrous)

        # [N * C/m * H * W]
        y = torch.cat(out, dim=0)
        y = self.fuse(y)
        y_atrous = torch.cat(out_atrous, dim=0)
        # [N * 2C/m * H * W]
        y_atrous = self.fuse(y_atrous)
        y = torch.cat([y, y_atrous], dim=1) 
        return y
