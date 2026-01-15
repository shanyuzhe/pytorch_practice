import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout_rate=0.3):
        super(SimpleMLP, self).__init__()
        
        # 第一层：输入层 -> 隐藏层
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), # 批归一化：稳定分布
            nn.LeakyReLU(0.1),          # LeakyReLU 防止神经元死亡
            nn.Dropout(dropout_rate)    # Dropout 防止过拟合
        )
        
        # 第二层：隐藏层 -> 隐藏层
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate)
        )
        
        # 第三层：隐藏层 -> 隐藏层 (再缩小一点)
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate)
        )
        
        # 输出层：输出 Logits (不加 Sigmoid)
        self.output_layer = nn.Linear(hidden_dim // 4, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.output_layer(x)
        return x