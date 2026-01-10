import torch
import numpy as np
import random
import os
import time
from contextlib import contextmanager
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
# ... 下面是原来的代码 ...
# ==========================================
# 1. 设备选择 (Device Selection)
# ==========================================
def get_device():
    """自动选择设备，优先使用 CUDA"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        # 打印显卡信息，确认没跑在 CPU 上
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("⚠️ Using CPU")
    return device

# ==========================================
# 2. Seed 固定 (Reproducibility)
# ==========================================
def setup_seed(seed=277527):
    """一键固定所有随机种子，确保实验可复现"""
    torch.manual_seed(seed)#cpu范围内的seed
    torch.cuda.manual_seed_all(seed)#gpu范围内的seed
    np.random.seed(seed)#numpy范围内的seed
    random.seed(seed)#python范围内的seed
    # 为了保证绝对一致性，可能会牺牲一点点速度（可选）
    torch.backends.cudnn.deterministic = True#确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False#关闭自动寻找最优卷积算法
    print(f"🔒 Random seed set to: {seed}")

# ==========================================
# 3. 计时器 (Timer)
# ==========================================
@contextmanager#这是一个装饰器，将一个生成器函数转换为上下文管理器
#写了这个以后可以用with语句调用这个函数
def time_block(label="Block"):
    """上下文管理器计时器
    用法:
    with time_block("Matrix Mul"):
        output = a @ b
    """
    start = time.perf_counter()
    try:
        yield#用yield移交“控制权”，让with语句块内的代码运行。类似于中断点
    finally:
        end = time.perf_counter()
        print(f"⏱️  {label} time: {end - start:.6f} sec")
    #try finally结构确保无论代码块内是否抛出异常，计时器都能正确结束并打印时间
# ==========================================
# 4. 常用打印 (Tensor Inspector)
# ==========================================
def inspect(tensor, name="Tensor"):
    """
    打印张量的关键信息：shape, dtype, device, grad
    防止眼花缭乱，只看核心属性
    """
    if not isinstance(tensor, torch.Tensor):
        print(f"❌ {name} is not a Tensor (Type: {type(tensor)})")
        return

    info = f"🔎 {name}: \n" \
           f"   Shape:  {tuple(tensor.shape)}\n" \
           f"   Dtype:  {tensor.dtype}\n" \
           f"   Device: {tensor.device}\n" \
           f"   Grad:   {tensor.requires_grad} (Grad Fn: {tensor.grad_fn is not None})"
    
    # 如果是标量（比如 Loss），打印数值
    if tensor.numel() == 1:
        info += f"\n   Value:  {tensor.item():.4f}"
    
    print(info)
    print("-" * 30)

# ==========================================
# 初始化默认环境
# ==========================================
DEVICE = get_device()
setup_seed()

if __name__ == "__main__":
    # 测试一下我们的工具
    print("\n--- Testing Playground ---")
    
    # 1. 测试张量和 inspect
    x = torch.randn(3, 4, requires_grad=True, device=DEVICE)
    inspect(x, "Test Tensor x")
    
    # 2. 测试计时器
    with time_block("Sleep Test"):
        y = x.sum()
        time.sleep(0.1)
    
    inspect(y, "Sum Result y")