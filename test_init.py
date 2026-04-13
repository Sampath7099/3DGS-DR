import torch
from scene.gaussian_model import GaussianModel
import os

model = GaussianModel(0)
model.create_from_pcd(type('PCD', (object,), {'points': [[0,0,0]], 'colors': [[0,0,0]]})(), 1.0, 128, 1)

print("Global map initial test:")
rays = torch.rand(100, 3).cuda()
out = model.env_maps[0](rays)
print(out.mean().item(), out.std().item())

from arguments import OptimizationParams
model.training_setup(OptimizationParams(None))
model.split_env_maps(2, 0.01)

print("Local map initial test:")
out_local = model.env_maps[1](rays)
print(out_local.mean().item(), out_local.std().item())
