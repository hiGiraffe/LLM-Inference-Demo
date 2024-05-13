from __future__ import print_function
import torch
import time

cuda1 = torch.device('cuda', 0)
cpu = torch.device('cpu', 0)
# 测试从CPU到GPU数据加载时间
s = time.time()
for _ in range(100000):
    a = torch.randn((2, 3), device=cpu)
    a.to(cuda1)
print('cpu --> gpu cost time is', time.time() - s)

# 测试从GPU到GPU数据加载时间
cuda1 = torch.device('cuda', 0)
cuda2 = torch.device('cuda', 1)
s = time.time()
for _ in range(100000):
    a = torch.randn((2, 3), device=cuda2)
    a.to(cuda1)
print('gpu --> gpu cost time is', time.time() - s)

# 测试从GPU到CPU数据加载时间
cuda1 = torch.device('cuda', 0)
cpu = torch.device('cpu', 0)
s = time.time()
for _ in range(100000):
    a = torch.randn((2, 3), device=cuda1)
    a.to(cpu)
print('gpu --> cpu cost time is', time.time() - s)