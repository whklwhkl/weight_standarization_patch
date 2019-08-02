from resnet import Bottleneck

import torch
import torch.nn as nn
import torch.nn.functional as F


def fuse(conv, bn):
    w = conv.weight
    mean = bn.running_mean
    var_sqrt = torch.sqrt(bn.running_var + bn.eps)
    beta = bn.weight
    gamma = bn.bias
    if conv.bias is not None:
        b = conv.bias
    else:
        b = mean.new_zeros(mean.shape)
    w = w * (beta / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
    b = (b - mean)/var_sqrt * beta + gamma
    fused_conv = nn.Conv2d(conv.in_channels,
                           conv.out_channels,
                           conv.kernel_size,
                           conv.stride,
                           conv.padding,
                           conv.dilation,
                           conv.groups,
                           bias=True)
    fused_conv.weight = nn.Parameter(w)
    fused_conv.bias = nn.Parameter(b)
    return fused_conv


def merge_conv_bn(net):
    """
    replace CNN+BN with CNN to speed up
    """
    previous = None
    has_seen_cnn = False
    conv_replace_queue = []
    bn_replace_queue = []
    for s in net.children():
        if has_seen_cnn and isinstance(s, nn.BatchNorm2d):
            conv_replace_queue.append(previous)
            bn_replace_queue += [s]
        if isinstance(s, nn.Conv2d):
            has_seen_cnn = True
        else:
            has_seen_cnn = False
        previous = s
    if len(conv_replace_queue):
        if isinstance(net, nn.Sequential):
            for i, sub in enumerate(net):
                if isinstance(sub, nn.Conv2d) and sub in conv_replace_queue:
                    idx = conv_replace_queue.index(sub)
                    bn = bn_replace_queue[idx]
                    new_conv = fuse(sub, bn)
                    net[i] = new_conv
                    net[i + 1] = nn.Identity()
        else:
            for n in dir(net):
                sub = getattr(net, n)
                if isinstance(sub, nn.Conv2d) and sub in conv_replace_queue:
                    idx = conv_replace_queue.index(sub)
                    bn = bn_replace_queue[idx]
                    new_conv = fuse(sub, bn)
                    setattr(net, n, new_conv)
            for n in dir(net):
                sub = getattr(net, n)
                if isinstance(sub, nn.BatchNorm2d) and sub in bn_replace_queue:
                    setattr(net, n, nn.Identity())


def cvt_ws_cnn(net):
    previous = None
    has_seen_cnn = False
    replace_queue = set()
    for s in net.children():
        if has_seen_cnn and isinstance(s, nn.BatchNorm2d):
            replace_queue.add(previous)
        if isinstance(s, nn.Conv2d):
            has_seen_cnn = True
        else:
            has_seen_cnn = False
        previous = s
    if len(replace_queue):
        for n in dir(net):
            sub = getattr(net, n)
            if isinstance(sub, nn.Conv2d) and sub in replace_queue:
                new_conv = Conv2d(sub.in_channels,
                                  sub.out_channels,
                                  sub.kernel_size,
                                  sub.stride,
                                  sub.padding,
                                  sub.dilation,
                                  sub.groups,
                                  sub.bias is None)
                new_conv.weight = sub.weight
                # todo: bn weights, bias
        # if isinstance(subnet, BasicConv2d):
        #     subnet.conv = fuse(subnet.conv, subnet.bn)
        #     del subnet.bn
        #     subnet.forward = _relu.__get__(subnet, BasicConv2d)  # monkey patch


class Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


if __name__ == '__main__':
    import sys
    model_path = sys.argv[1]
    m = torch.load(model_path).eval().apply(merge_conv_bn)
    torch.save(m, model_path.replace('.pth', '_{}.pth'.format('fused')))
