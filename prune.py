import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init
from functools import partial
import math
from thop import profile
from ptflops import get_model_complexity_info




class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out_forward = 0.5 * torch.sign(x) + 0.5
        mask1 = x < -0.5
        mask2 = x < 0
        mask3 = x < 0.5
        out1 = 0 * mask1.type(torch.float32) + (2 * x * x + 2 * x + 0.5) * (1 - mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-2 * x * x + 2 * x + 0.5) * (1 - mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1 - mask3.type(torch.float32))
        # out3 = self.sigmoid(x)
        out = out_forward.detach() - out3.detach() + out3
        return out


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def downsample_basic_block(x, planes):
    x = nn.AvgPool2d(2,2)(x)
    zero_pads = torch.Tensor(
        x.size(0), planes - x.size(1), x.size(2), x.size(3)).zero_()
    if isinstance(x.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()
    out = Variable(torch.cat([x.data, zero_pads], dim=1))
    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, cfg, stride=1, downsample=None, mask_index=None):
        # cfg should be a number in this case
        super(BasicBlock, self).__init__()
        self.main_network_pretrain = False
        self.mask_index = mask_index
        self.conv1 = conv3x3(in_planes, cfg, stride)
        self.bn1 = nn.BatchNorm2d(cfg)
        self.conv2 = conv3x3(cfg, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        # channel select block
        self.WW = nn.Linear(cfg, 1)
        self.binary_layer = BinaryActivation()

        self.downsample = downsample
        self.stride = stride

    def mask_mult(self, x, y):
        shape2 = y.size()
        return x.mul(y.view(1, shape2[0], shape2[1], 1))

    def forward(self, x):
        mask_index = []
        mask = torch.zeros(1)
        mask = mask.to(x.device)
        flops = torch.zeros(1)
        flops = flops.to(x.device)

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out_conv_weights = self.conv1.weight.data.clone()
        out_conv_weights = out_conv_weights.view(self.conv1.weight.data.size(0), -1)
        in_conv_weights = self.conv2.weight.data.clone()
        in_conv_weights = in_conv_weights.permute(1, 0, 2, 3)
        in_conv_weights = in_conv_weights.reshape(self.conv2.weight.data.size(1), -1)
        out_in_conv_weights = torch.cat((out_conv_weights, in_conv_weights), dim = 1)
        filter_similarity_matrix = F.cosine_similarity(out_in_conv_weights.unsqueeze(1), out_in_conv_weights.unsqueeze(0), dim=2)
        real_value_mask = self.WW(filter_similarity_matrix)
        binary_mask = self.binary_layer(real_value_mask)
        mask_index.append(binary_mask)
        mask += torch.sum(binary_mask)
        if self.mask_index is not None:
            out = self.mask_mult(out, self.mask_index[0])
        elif not self.main_network_pretrain:
            out = self.mask_mult(out, binary_mask)
        flops += conv_flops_compute(self.conv1.weight.size(), out.size()[2:4], out_mask = mask)

        out = self.conv2(out)
        out = self.bn2(out)
        flops += conv_flops_compute(self.conv2.weight.size(), out.size()[2:4], in_mask = mask)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out, mask_index, mask, flops


class Net(nn.Module):
    def __init__(self, depth, dataset='Div2k', cfg=None, mask_index=None):
        super(Net, self).__init__()
        # Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        block = BasicBlock

        if cfg == None:
            cfg = [[16] * n, [32] * n, [64] * n]
            cfg = [item for sub_list in cfg for item in sub_list]

        self.inplanes = 16
        self.n = n

        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.mask_index = mask_index
        if self.mask_index is not None:
            self.layer1 = self._make_layer(block, 16, n, cfg=cfg[0:n], mask_index=self.mask_index[0:n])
            self.layer2 = self._make_layer(block, 32, n, cfg=cfg[n:2 * n], stride=2,
                                           mask_index=self.mask_index[n:2 * n])
            self.layer3 = self._make_layer(block, 64, n, cfg=cfg[2 * n:3 * n], stride=2,
                                           mask_index=self.mask_index[2 * n:3 * n])
        else:
            self.layer1 = self._make_layer(block, 16, n, cfg=cfg[0:n])
            self.layer2 = self._make_layer(block, 32, n, cfg=cfg[n:2 * n], stride=2)
            self.layer3 = self._make_layer(block, 64, n, cfg=cfg[2 * n:3 * n], stride=2)

        self.avgpool = nn.AvgPool2d(8)

        num_classes = 10
        
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, block_num, cfg, stride=1, mask_index=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = partial(downsample_basic_block, planes=planes * block.expansion)

        layers = []
        if mask_index is not None:
            layers.append(block(self.inplanes, planes, cfg[0], stride, downsample, mask_index[0]))
            self.inplanes = planes * block.expansion
            for i in range(1, block_num):
                layers.append(block(self.inplanes, planes, cfg[i], mask_index[i]))
        else:
            layers.append(block(self.inplanes, planes, cfg[0], stride, downsample))
            self.inplanes = planes * block.expansion
            for i in range(1, block_num):
                layers.append(block(self.inplanes, planes, cfg[i]))

        return nn.Sequential(*layers)

    def forward(self, x):
        mask_index = []
        mask = torch.zeros(3 * self.n)
        mask = mask.to(x.device)
        k = 0
        flops = torch.zeros(1)
        flops = flops.to(x.device)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # 32x32
        # flops += conv_flops_compute(self.conv1.weight.size(), x.size()[2:4])

        for i in range(0, self.n):
            x, mask_index0, mask[k], flops0 = self.layer1[i](x)
            mask_index.append(mask_index0)
            flops += flops0
            k = k + 1  # 32x32

        for i in range(0, self.n):
            x, mask_index0, mask[k], flops0 = self.layer2[i](x)
            mask_index.append(mask_index0)
            flops += flops0
            k = k + 1  # 16x16

        for i in range(0, self.n):
            x, mask_index0, mask[k], flops0 = self.layer3[i](x)
            mask_index.append(mask_index0)
            flops += flops0
            k = k + 1  # 8x8

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # flops += linear_flops_compute(self.fc.weight.size())

        return x, mask_index, mask, flops