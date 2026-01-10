# 🛠️ PyTorch Debug Checklist
> 遇到 Bug 先别慌，按顺序灵魂拷问自己一遍。

## 1. 维度与形状 (Shape Mismatch)
- [ ] **Shape 是否符合预期？**
    - 尤其注意 LLM 中的 (Batch, Time, Channel) vs (Batch, Channel, Time)。
    - 卷积层的输入是否需要 (N, C, H, W)？
- [ ] **广播 (Broadcasting) 是否隐式发生了？**
    - 比如 (3, 1) + (3,) 会变成 (3, 3)，而不是你以为的对应相加。
- [ ] **Reshape/View 之后数据是否错乱？**
    - 如果使用了 `permute` 或 `transpose`，后续是否接着用了 `view`？(必须先 `.contiguous()` 再 `view`，或者直接用 `reshape`)。

## 2. 数据类型 (Dtype)
- [ ] **Input 是否为 Float32？**
    - 神经网络输入通常是 `float32`，不是 `float64` (double) 也不是 `long`。
- [ ] **Target/Label 是否为 Long？**
    - `CrossEntropyLoss` 的 target 必须是 `long` (int64)，且不能是 One-hot（除非自定义 loss）。
- [ ] **混合精度导致溢出？**
    - 如果用了 FP16，Loss 是否变成了 NaN？

## 3. 设备 (Device)
- [ ] **所有 Tensor 都在同一个设备上吗？**
    - 常见报错：`Expected all tensors to be on the same device`。
    - 检查：Model, Input, Label 是否都 `.to(device)` 了。

## 4. 梯度与训练 (Gradient & Training)
- [ ] **Loss 是标量 (Scalar) 吗？**
    - 最终 backward 的 loss 必须是一个数，不能是向量。检查是否漏了 `.mean()` 或 `.sum()`。
- [ ] **是否忘了清空梯度？**
    - 训练循环里写了 `optimizer.zero_grad()` 吗？
- [ ] **train/eval 模式切换了吗？**
    - 验证/测试时是否写了 `model.eval()`？(影响 Dropout 和 BatchNorm)。
    - 训练时是否写了 `model.train()`？
- [ ] **Tensor 是否意外切断了梯度？**
    - 中间变量是否不小心用了 `.detach()` 或 `.item()` 导致计算图断裂？

## 5. 常见 NaN/Inf 原因
- [ ] 学习率 (LR) 是否太大？
- [ ] 除数是否可能为 0？(加个 epsilon: `x / (y + 1e-8)`)
- [ ] Log 的输入是否 <= 0？
