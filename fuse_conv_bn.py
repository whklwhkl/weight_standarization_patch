import torch
import torch.nn as nn


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


if __name__ == '__main__':
    import sys
    model_path = sys.argv[1]
    m = torch.load(model_path).eval().apply(merge_conv_bn)
    torch.save(m, model_path.replace('.pth', '_{}.pth'.format('fused')))
