import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        self.register_buffer('fixed_weight', torch.ones(16, 3, 3, 3))
        self.bn = nn.BatchNorm2d(16)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        # 使用缓冲区中的固定权重
        x = torch.nn.functional.conv2d(x, self.fixed_weight)
        return x

# 创建模型
model = MyModel()

# 打印缓冲区
for name, buffer in model.named_buffers():
    print(f"Buffer {name}: {buffer.size()}")