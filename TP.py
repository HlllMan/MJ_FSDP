import os

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.tensor import DTensor
import time

from DDP import DDP
from MyTrainDataset import MyTrainDataset
from Model.model import ToyModel
from torch.distributed.device_mesh import init_device_mesh
from TP_parallel import (
    ColwiseParallel,
    RowwiseParallel,
)
from parallelize_module import parallelize_module

def set_seed(seed=42):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 尽量保证“统计意义上可复现/接近”
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def main(total_epochs: int):

    # 在 main 函数入口调用
    set_seed(42)
    _local_rank = int(os.environ["LOCAL_RANK"])
    _global_rank = int(os.environ["RANK"])
    _world_size = int(os.environ["WORLD_SIZE"])

    # 每卡 batch 固定 256，避免多卡时 batch 过小导致 all-reduce 次数过多、通信占主导
    # 1 卡: 256/epoch; 8 卡: 每卡 256，global=2048，batch 数减少 8 倍 → all-reduce 次数减少
    batch_size = 256
    
    
    torch.cuda.set_device(_local_rank)
    
    if not dist.is_initialized():
        # 明确指定 device_id 以避免警告和潜在的设备映射错误
        dist.init_process_group(backend="nccl", device_id=_local_rank)

    try:
        # 当前容器中 torch 没有 torch.accelerator，直接固定用 CUDA
        device_type = "cuda"
        
        # 1 卡：不做 TP/DP；多卡：默认 TP=2，其余作为 DP
        if _world_size == 1:
            tp_size = 1
        else:
            tp_size = 2

        dp_size = _world_size // tp_size

        assert _world_size % tp_size == 0, f"World size {_world_size} must be divisible by TP size {tp_size}"
        
        device_mesh = init_device_mesh(
            device_type=device_type,
            mesh_shape=(dp_size, tp_size),
            mesh_dim_names=("dp", "tp")
        )
        
        tp_mesh = device_mesh["tp"]
        dp_mesh = device_mesh["dp"]
        
        dp_rank = dp_mesh.get_local_rank()
        
        model = ToyModel().to(device_type)
        
        model = parallelize_module(
            module=model,
            device_mesh=tp_mesh,
            parallelize_plan={
                "in_proj": ColwiseParallel(),
                "out_proj": RowwiseParallel(),
            },
        )
        
        # 使用自定义简化版 DDP：仅在 DP 维度上同步梯度
        model = DDP(
            model,
            process_group=dp_mesh.get_group(),  # 只在 DP 维度上同步
        )
        
        # 确保模型处于训练模式
        model.train()
        
        dataset = MyTrainDataset(204800)
        # 使用 dp_mesh 的 size  and rank 配置 sampler
        train_data = DataLoader(
            dataset,
            batch_size,
            pin_memory=True,
            shuffle=False,
            sampler=DistributedSampler(
                dataset,
                num_replicas=dp_size,
                rank=dp_rank
            )
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.25, foreach=True)
        # 确保所有进程都完成初始化后再开始训练
        # 重要：需要全局 barrier，不只是 DP 组内的 barrier
        dist.barrier()
        
        # 添加调试输出，确认所有节点都到达训练循环
        if _global_rank == 0:
            print(f"[Rank {_global_rank}] 开始训练，总 epoch: {total_epochs}")
        elif _global_rank == 8:
            print(f"[Rank {_global_rank}] Worker 节点已就绪，开始训练")
        
        torch.cuda.synchronize()
        t0 = time.perf_counter()


        for epoch in range(total_epochs):
            train_data.sampler.set_epoch(epoch)
            
            for batch_idx, (source, targets) in enumerate(train_data):
                optimizer.zero_grad()
                
                source = source.to(device_type)
                targets = targets.to(device_type)  # 不再需要 .long()，因为标签是连续值
                
                output = model(source)

                # 处理可能的 AsyncCollectiveTensor（需要先 wait 才能使用）
                from torch.distributed._functional_collectives import AsyncCollectiveTensor
                if isinstance(output, AsyncCollectiveTensor):
                    output = output.wait()
                
                # RowwiseParallel 输出是 Replicate() 的 DTensor（use_local_output=False）
                # 需要转换为普通 Tensor 才能使用 MSE loss，但要确保 autograd graph 完整
                if isinstance(output, DTensor):
                    # 使用 .to_local() 转换，但确保 autograd graph 完整
                    # 注意：to_local() 本身是 autograd Function，应该保持 grad_fn
                    output = output.to_local()
                
                # 现在 output 是普通 Tensor，可以直接使用 MSE loss
                # output: [batch, 5], targets: [batch, 5] - shape 完全匹配
                loss = F.mse_loss(output, targets)
                loss.backward()
                # source 0.7797 targets 0.9832 output -0.0543 loss 0.2909
                optimizer.step()
                # 输出 loss：每个节点的 local rank 0 打印，避免同节点重复
                if _local_rank == 0 and batch_idx  == 0:
                    print(f"[LocalRank {_local_rank}] Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        if _global_rank ==0 :
            total = t1 - t0
            print(f"total time: {total}sec ")

    finally:
        # 确保无论是否发生异常都清理进程组
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main(total_epochs=10)