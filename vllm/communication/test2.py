from line_profiler import LineProfiler
import torch
import time

@profile
def communicate():
    cuda1 = torch.device('cuda', 0)
    cpu = torch.device('cpu', 0)
    # 测试从CPU到GPU数据加载时间
    for _ in range(100):
        a = torch.randn((2, 3), device=cpu)

    # 测试从GPU到GPU数据加载时间
    cuda1 = torch.device('cuda', 0)
    cuda2 = torch.device('cuda', 1)
    for _ in range(100000):
        a = torch.randn((2, 3), device=cuda2)
        a.to(cuda1)

    # 测试从GPU到CPU数据加载时间
    cuda1 = torch.device('cuda', 0)
    cpu = torch.device('cpu', 0)
    for _ in range(100000):
        a = torch.randn((2, 3), device=cuda1)
        a.to(cpu)

if __name__ == '__main__':
    communicate()