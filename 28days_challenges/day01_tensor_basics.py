from utils.torch_playground import *

print('----1.张量创建与Dtype----')

# 创建不同dtype的张量
# 创建默认张量
x_float = torch.randn(3,3)
inspect(x_float, name = "x_float")

