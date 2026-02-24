import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel.style import ParallelStyle

from fnmatch import fnmatch


def parallelize_module(
    module: nn.Module,
    device_mesh: DeviceMesh,
    parallelize_plan: dict[str, ParallelStyle],
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
    for module_path, parallelize_style in parallelize_plan.items():
        
        path_splits = module_path.split(".")

        while path_splits:
            atom = path_splits.pop(0)
            matched_children = filter(
                lambda t: fnmatch(t[0], atom), module.named_children()
            )
            for _, submodule in matched_children:
                if path_splits:
                    leaf_path = ".".join(path_splits)
                    parallelize_module(submodule, device_mesh, {leaf_path: parallelize_style})
                else:
                    parallelize_style._apply(submodule, device_mesh)
    
    return module
