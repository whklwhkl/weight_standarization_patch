# Fuse Conv2d and BatchNorm2d
Useful function to speed up networks.

## speed up
```python
from fuse_conv_bn import merge_conv_bn

import torch, torchvision

model = torchvision.models.resnet50(1000)
x = torch.rand(1, 3, 224, 224)

y = model(x)  # 49.6 ms ± 657 µs
model.apply(merge_conv_bn)
y = model(x)  # 42.7 ms ± 547 µs
```
