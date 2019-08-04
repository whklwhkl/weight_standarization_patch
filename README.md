# Fuse Conv2d and BatchNorm2d
Useful function to speed up networks in `pytorch`.

## install
```bash
git clone https://github.com/whklwhkl/weight_standarization_patch
cd weight_standarization_patch
python setup.py install
```

## Usage

### call in python scripts
```python
from fuse_conv_bn import merge_conv_bn

import torch, torchvision

model = torchvision.models.resnet50(1000)
x = torch.rand(1, 3, 224, 224)

y = model(x)  # 61.9 ms ± 510 µs
model.apply(merge_conv_bn)
y = model(x)  # 40.1 ms ± 206 µs
```
### command line tool
```bash
python fuse_conv_bn.py ${YOUR_MODEL}.pth  # saved as ${YOUR_MODEL_fused}.pth
```
