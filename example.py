import torch

from aoa_torch.main import AoA

x = torch.randn(1, 10, 512)
model = AoA(512, 8, 64, 0.1)
out = model(x)
print(out.shape)
