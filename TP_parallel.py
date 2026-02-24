import torch
import torch.nn as nn
from torch.distributed.tensor.parallel import ParallelStyle
from functools import partial
from torch.distributed.device_mesh import DeviceMesh



# 标准导入 DTensor API（PyTorch 2.2+）
from torch.distributed.tensor import (
    distribute_tensor,
    distribute_module,
    DTensor,
    Replicate,
    Shard,
)


class ColwiseParallel(ParallelStyle):
    def __init__(self, gather_output: bool = False):
        super().__init__()
        self.input_layouts = (Replicate(),)
        self.output_layouts = (Shard(-1),)
        self.desired_input_layouts = (Replicate(),)
        self.use_local_output = True
        self.src_data_rank = 0
        if gather_output:
            self.output_layouts = (Replicate(),)
            self.use_local_output = False

    @staticmethod
    def _prepare_input_fn(input_layouts, desired_input_layouts, mod, inputs, device_mesh):
        input_tensor = inputs[0]
        if isinstance(input_tensor, DTensor):
            if input_tensor.placements != desired_input_layouts:
                input_tensor = input_tensor.redistribute(
                    placements=desired_input_layouts, async_op=True
                )
            return input_tensor
        input_tensor = DTensor.from_local(
            input_tensor, device_mesh, input_layouts, run_check=False
        )
        if input_layouts != desired_input_layouts:
            input_tensor = input_tensor.redistribute(
                placements=desired_input_layouts, async_op=True
            )
        return input_tensor

    def _partition_linear_fn(self, name, module, device_mesh):
        for name, param in module.named_parameters():
            dist_param = nn.Parameter(
                distribute_tensor(
                    param,
                    device_mesh,
                    [Shard(0)],
                )
            )
            module.register_parameter(name, dist_param)

    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        if outputs.placements != output_layouts:
            outputs = outputs.redistribute(placements=output_layouts, async_op=True)
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            self._partition_linear_fn,
            partial(
                self._prepare_input_fn, self.input_layouts, self.desired_input_layouts
            ),
            partial(
                self._prepare_output_fn, self.output_layouts, self.use_local_output
            ),
        )


class RowwiseParallel(ParallelStyle):
    def __init__(self):
        super().__init__()
        self.input_layouts = (Shard(-1),)
        # 保持 DTensor 输出（use_local_output=False），避免 to_local() 破坏 autograd graph
        # 输出是 Replicate() 的 DTensor，可以在 loss 计算时转换为普通 Tensor
        self.output_layouts = (Replicate(),)
        self.use_local_output = False  # 保持 DTensor，避免 autograd graph 被破坏
        self.src_data_rank = 0

    @staticmethod
    def _prepare_input_fn(input_layouts, desired_input_layouts, mod, inputs, device_mesh):
        input_tensor = inputs[0]
        input_tensor = DTensor.from_local(
            input_tensor, device_mesh, input_layouts, run_check=False
        )
        if input_layouts != desired_input_layouts:
            input_tensor = input_tensor.redistribute(
                placements=desired_input_layouts, async_op=True
            )
        return input_tensor

    def _partition_linear_fn(self, name, module, device_mesh):
        module.register_parameter(
            "weight",
            nn.Parameter(
                distribute_tensor(
                    module.weight,
                    device_mesh,
                    [Shard(1)],
                )
            ),
        )

    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        if outputs.placements != output_layouts:
            outputs = outputs.redistribute(placements=output_layouts, async_op=True)
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        self.desired_input_layouts = (Shard(-1),)
        return distribute_module(
            module,
            device_mesh,
            self._partition_linear_fn,
            partial(
                self._prepare_input_fn, self.input_layouts, self.desired_input_layouts
            ),
            partial(
                self._prepare_output_fn, self.output_layouts, self.use_local_output
            ),
        )
