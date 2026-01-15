import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sklearn
from sklearn.model_selection import train_test_split# 用于划分训练集和验证集
from utils import torch_playground
from mlp import SimpleMLP
import copy # 用于深度拷贝最佳模型

# === 超参数配置 ===
DEVICE = torch_playground.get_device()
NUM_EPOCHS = 150        # 轮数可以多一点，因为有 Early Stopping 机制
LR = 0.002              # 初始学习率
BATCH_SIZE = 64
HIDDEN_DIM = 256        # 加宽网络
WEIGHT_DECAY = 1e-4     # 正则化力度稍大一点

def main():
    # 1. 读入数据
    print('>>> 正在加载数据...')
    df_train = pd.read_csv('dataset/train_processed.csv')
    df_test = pd.read_csv('dataset/test_processed.csv')

   # ... 上面的代码不变 ...

    # 准备特征和标签 (训练集有 Transported，所以要 drop)
    X_all = df_train.drop(['Transported', 'PassengerId'], axis=1).values
    y_all = df_train['Transported'].values
    
    # 准备测试集数据 (测试集没有 Transported，所以只 drop PassengerId)
    # -----------------------------------------------------------
    X_sub = df_test.drop(['PassengerId'], axis=1).values 
    # -----------------------------------------------------------
    ids_sub = df_test['PassengerId']

    # ... 下面的代码不变 ...
    # === 关键步骤：划分 训练集(80%) 和 验证集(20%) ===
    # stratify=y_all 保证训练集和验证集中正负样本比例一致
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )

    # 转 Tensor
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    X_sub = torch.tensor(X_sub, dtype=torch.float32)

    # DataLoader
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    # 2. 初始化模型
    input_dim = X_train.shape[1]
    model = SimpleMLP(input_dim=input_dim, hidden_dim=HIDDEN_DIM)
    model.to(DEVICE)

    # 优化器 & 损失函数
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()
    
    # 学习率调度器：如果验证集 Loss 10个 epoch 不下降，学习率变成原来的 0.1
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)

    # === 3. 训练循环 ===
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    print(f'\n开始训练 (Device: {DEVICE})...')
    
    for epoch in range(NUM_EPOCHS):
        # --- 训练阶段 ---
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # --- 验证阶段 (评估当前模型能力) ---
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X_v, y_v in val_loader:
                X_v, y_v = X_v.to(DEVICE), y_v.to(DEVICE)
                outputs = model(X_v)
                # 计算验证集准确率
                predicted = (outputs > 0).float()
                val_total += y_v.size(0)
                val_correct += (predicted == y_v).sum().item()
        
        val_acc = val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)

        # 更新学习率 (根据验证集准确率)
        scheduler.step(val_acc)

        # 打印日志
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.4f}')

        # --- 保存最佳模型 ---
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            # print(f'  --> 新的最佳模型! Acc: {best_acc:.4f}')

    print(f'\n训练结束。最佳验证集准确率: {best_acc:.4f}')

    # === 4. 预测 ===
    # 加载表现最好的那个模型权重，而不是最后一个 epoch 的权重
    model.load_state_dict(best_model_wts)
    model.eval()
    
    # 修改 train.py 的最后几行
    with torch.no_grad():
        X_sub = X_sub.to(DEVICE)
        y_pred_sub = model(X_sub)
        # 变成概率 (0~1)，不要变成 bool
        y_prob_sub = torch.sigmoid(y_pred_sub).cpu().numpy().flatten()

    # 保存为 mlp_pred.csv
    sub = pd.DataFrame({'PassengerId': ids_sub, 'Transported_Prob': y_prob_sub})
    sub.to_csv('mlp_pred.csv', index=False)

if __name__ == '__main__':
    main()