import os

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
import torch.distributed as dist
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_backend
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.runtime as xr


def _mp_fn(index):
  device = xm.xla_device()
# -  dist.init_process_group('xla', rank=xm.get_ordinal(), world_size=xm.xrt_world_size())
  dist.init_process_group('xla', init_method='xla://')

  torch.manual_seed(42)
  model = nn.Linear(128, 10).to(device)

  # Optional for TPU v4 and GPU
  xm.broadcast_master_param(model)
  model = DDP(model, gradient_as_bucket_view=True)

  loss_fn = nn.MSELoss()
  optimizer = optim.SGD(model.parameters(), lr=.001)

  for i in range(10):
    data, target = torch.randn((128, 128), device=device), torch.randn((128, 10), device=device)

    optimizer.zero_grad()
    output = model(data)
    loss = loss_fn(output, target)
    loss.backward()

    optimizer.step()
    xm.mark_step()

  # Print mean parameters so we can confirm they're the same across replicas
  print([p.mean() for p in model.parameters()])

if __name__ == '__main__':
# -  os.environ['XRT_TPU_CONFIG'] = 'localservice;0;localhost:51011'
# -  os.environ['MASTER_ADDR'] = 'localhost'
# -  os.environ['MASTER_PORT'] = '12355'

  # Recommended: set PJRT_DEVICE to your local device type
  os.environ['PJRT_DEVICE'] = 'TPU'

  xmp.spawn(_mp_fn)