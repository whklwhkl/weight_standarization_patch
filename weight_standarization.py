import torch
import torch.nn as nn
import torch.nn.functional as F


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
    m = torch.load(model_path).apply(cvt_ws_cnn)
    torch.save(m, model_path.replace('.pth', '_{}.pth'.format('fused')))
