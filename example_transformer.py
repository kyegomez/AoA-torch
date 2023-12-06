import torch 
from aoa.main import AoATransformer


x = torch.randint(0, 100, (1, 10))
model = AoATransformer(512, 1, 100)
out = model(x)
print(out.shape)
