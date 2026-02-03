import sys
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.distributed_c10d import get_global_rank


class DDP(nn.Module):
    """
    

    用法示例（和官方 DDP 类似）:
        model = ToyModel().to(device)
        model = DDP(model)  # 替代 torch.nn.parallel.DistributedDataParallel
        ...
        loss.backward()     # 会触发 grad hook 做 all_reduce
        optimizer.step()
    """

    def __init__(self, module: nn.Module, process_group: dist.ProcessGroup | None = None) -> None:
        super().__init__()
        if not dist.is_initialized():
            raise RuntimeError("DDP: torch.distributed 未初始化，请先调用 dist.init_process_group(...)。")

        self.module = module
        self.process_group = process_group if process_group is not None else dist.group.WORLD

        # 记录 world_size / rank
        self.world_size = dist.get_world_size(self.process_group)
        self.rank = dist.get_rank(self.process_group)

        # 处理 TP + DDP 混合并行：将 DTensor 参数转换为本地 Tensor
        from torch.distributed.tensor.parallel.ddp import _pre_dp_module_transform
        _pre_dp_module_transform(self.module)

        # 做一次参数广播：保证所有 rank 上模型参数一致（以 DP group 的 rank0 为准）
        self._broadcast_parameters()

        # 构建 Reducer（使用官方 C++ Reducer，等效于官方 DDP 的梯度同步）
        self._build_reducer()

    # --------- 核心公开接口 ----------
    def forward(self, *args, **kwargs):
        """
        前向传播：直接调用原始 module 的 forward。
        """
        return self.module(*args, **kwargs)

    # --------- 内部辅助函数 ----------
    def _broadcast_parameters(self) -> None:
        """
        使用 dist.broadcast 同步 DP group 中 rank0 的参数/缓冲区到其他 rank。
        
        - 遍历所有参数
        - 逐个 broadcast
        - 只同步参数，不处理 buffers
        - 使用 DP 组内 rank 0 作为源，但需要先转换成全局 rank
        """
        params_list = []
        for name, param in self.module.named_parameters():
            if not isinstance(param, torch.Tensor):
                continue
            params_list.append((name, param))
        
        # DP 组内 src rank（组内 rank0），转换成全局 rank
        src_global_rank = get_global_rank(self.process_group, 0)

        for name, param in params_list:
            dist.broadcast(param, src=src_global_rank, group=self.process_group)

    def _build_reducer(self) -> None:
        """
        使用官方的 C++ Reducer，实现梯度同步（与官方 DDP 相同的核心逻辑）。
        """
        params = [p for p in self.module.parameters() if p.requires_grad]
        if len(params) == 0:
            return

        # 单一 bucket，包含全部参数
        bucket_indices = [list(range(len(params)))]
        per_bucket_size_limits = [sys.maxsize]
        # 创建 Reducer（对应 distributed.py 第1222行）
        # 仅保留必需参数和最小配置，其他使用默认值
        self.reducer = dist.Reducer(
            params,
            bucket_indices,
            per_bucket_size_limits,
            self.process_group,
        )


