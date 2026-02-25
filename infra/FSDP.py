import torch.nn as nn

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard



def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
):
    for layer_id, transformer_block in model.layers.items():
        fully_shard(
            transformer_block,
            mesh = dp_mesh
        )
    fully_shard(model, mesh=dp_mesh)