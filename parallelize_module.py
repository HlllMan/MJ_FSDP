import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel.style import ParallelStyle


def parallelize_module(
    module: nn.Module,
    device_mesh: DeviceMesh,
    parallelize_plan: dict[str, ParallelStyle],
    *,
    src_data_rank: int = 0,
) -> nn.Module:
    """
    parallelize_module，只保留 MVP 需要的功能。
    
    - 只支持字典形式的 parallelize_plan
    - 只支持直接子模块名（如 "in_proj", "out_proj"），不支持嵌套路径
    - 直接通过 module.get_submodule() 或 module.named_children() 查找子模块
    - 对每个匹配的子模块调用 parallelize_style._apply()
    
    Args:
        module: 要并行化的模块
        device_mesh: 设备网格（必须是 1D）
        parallelize_plan: 并行化计划字典，键是子模块名，值是对应的 ParallelStyle
        src_data_rank: 源数据 rank，默认 0
    
    Returns:
        并行化后的模块（原地修改，返回原模块引用）
    """
    for module_name, parallelize_style in parallelize_plan.items():
        # 设置 src_data_rank
        parallelize_style.src_data_rank = src_data_rank
        
        # 查找子模块（只支持直接子模块，不支持嵌套路径）
        # 对应原版 api.py 第99-105行：使用 module.named_children() 查找子模块
        # 原版使用 fnmatch 支持通配符，我们简化为精确匹配
        submodule = None
        for name, submod in module.named_children():
            if name == module_name:
                submodule = submod
                break
        
        # 应用并行化风格
        parallelize_style._apply(submodule, device_mesh)
    
    return module
