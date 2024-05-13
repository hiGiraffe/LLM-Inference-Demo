from line_profiler import LineProfiler
import torch
import time


@profile
def communicate():
    cpu = torch.device('cpu', 0)
    cuda1 = torch.device('cuda', 0)
    cuda2 = torch.device('cuda', 1)

    torch.cuda.synchronize(cuda1)
    torch.cuda.synchronize(cuda2)

    # warm up
    for _ in range(10000):
        a = torch.randn((2, 3), device=cuda2)
        a.to(cuda1)

    for _ in range(10000):
        a = torch.randn((2, 3), device=cpu)
        a.to(cuda1)

    torch.cuda.synchronize(cuda1)
    torch.cuda.synchronize(cuda2)

    # 1

    # 测试从CPU到GPU数据加载时间
    a = torch.randn((1, 32, 4096), device=cpu)
    a.to(cuda1)
    torch.cuda.synchronize(cuda1)

    # 测试从GPU到GPU数据加载时间
    a = torch.randn((1, 32, 4096), device=cuda2)
    a.to(cuda1)
    torch.cuda.synchronize(cuda1)
    torch.cuda.synchronize(cuda2)

    # 测试从GPU到CPU数据加载时间
    a = torch.randn((1, 32, 4096), device=cuda1)
    a.to(cpu)
    torch.cuda.synchronize(cuda1)

    # 10
    # 测试从CPU到GPU数据加载时间
    a = torch.randn((10, 32, 4096), device=cpu)
    a.to(cuda1)
    torch.cuda.synchronize(cuda1)

    # 测试从GPU到GPU数据加载时间
    a = torch.randn((10, 32, 4096), device=cuda2)
    a.to(cuda1)
    torch.cuda.synchronize(cuda1)
    torch.cuda.synchronize(cuda2)

    # 测试从GPU到CPU数据加载时间
    a = torch.randn((10, 32, 4096), device=cuda1)
    a.to(cpu)
    torch.cuda.synchronize(cuda1)

    # 20
    # 测试从CPU到GPU数据加载时间
    a = torch.randn((20, 32, 4096), device=cpu)
    a.to(cuda1)
    torch.cuda.synchronize(cuda1)

    # 测试从GPU到GPU数据加载时间
    a = torch.randn((20, 32, 4096), device=cuda2)
    a.to(cuda1)
    torch.cuda.synchronize(cuda1)
    torch.cuda.synchronize(cuda2)

    # 测试从GPU到CPU数据加载时间
    a = torch.randn((20, 32, 4096), device=cuda1)
    a.to(cpu)
    torch.cuda.synchronize(cuda1)

    # 30
    # 测试从CPU到GPU数据加载时间
    a = torch.randn((30, 32, 4096), device=cpu)
    a.to(cuda1)
    torch.cuda.synchronize(cuda1)

    # 测试从GPU到GPU数据加载时间
    a = torch.randn((30, 32, 4096), device=cuda2)
    a.to(cuda1)
    torch.cuda.synchronize(cuda1)
    torch.cuda.synchronize(cuda2)

    # 测试从GPU到CPU数据加载时间
    a = torch.randn((30, 32, 4096), device=cuda1)
    a.to(cpu)
    torch.cuda.synchronize(cuda1)

    # 40
    # 测试从CPU到GPU数据加载时间
    a = torch.randn((40, 32, 4096), device=cpu)
    a.to(cuda1)
    torch.cuda.synchronize(cuda1)

    # 测试从GPU到GPU数据加载时间
    a = torch.randn((40, 32, 4096), device=cuda2)
    a.to(cuda1)
    torch.cuda.synchronize(cuda1)
    torch.cuda.synchronize(cuda2)

    # 测试从GPU到CPU数据加载时间
    a = torch.randn((40, 32, 4096), device=cuda1)
    a.to(cpu)
    torch.cuda.synchronize(cuda1)

    # 50
    # 测试从CPU到GPU数据加载时间
    a = torch.randn((50, 32, 4096), device=cpu)
    a.to(cuda1)
    torch.cuda.synchronize(cuda1)

    # 测试从GPU到GPU数据加载时间
    a = torch.randn((50, 32, 4096), device=cuda2)
    a.to(cuda1)
    torch.cuda.synchronize(cuda1)
    torch.cuda.synchronize(cuda2)

    # 测试从GPU到CPU数据加载时间
    a = torch.randn((50, 32, 4096), device=cuda1)
    a.to(cpu)
    torch.cuda.synchronize(cuda1)

    # 60
    # 测试从CPU到GPU数据加载时间
    a = torch.randn((60, 32, 4096), device=cpu)
    a.to(cuda1)
    torch.cuda.synchronize(cuda1)

    # 测试从GPU到GPU数据加载时间
    a = torch.randn((60, 32, 4096), device=cuda2)
    a.to(cuda1)
    torch.cuda.synchronize(cuda1)
    torch.cuda.synchronize(cuda2)

    # 测试从GPU到CPU数据加载时间
    a = torch.randn((60, 32, 4096), device=cuda1)
    a.to(cpu)
    torch.cuda.synchronize(cuda1)

    # 70
    # 测试从CPU到GPU数据加载时间
    a = torch.randn((70, 32, 4096), device=cpu)
    a.to(cuda1)
    torch.cuda.synchronize(cuda1)

    # 测试从GPU到GPU数据加载时间
    a = torch.randn((70, 32, 4096), device=cuda2)
    a.to(cuda1)
    torch.cuda.synchronize(cuda1)
    torch.cuda.synchronize(cuda2)

    # 测试从GPU到CPU数据加载时间
    a = torch.randn((70, 32, 4096), device=cuda1)
    a.to(cpu)
    torch.cuda.synchronize(cuda1)

    # 80
    # 测试从CPU到GPU数据加载时间
    a = torch.randn((80, 32, 4096), device=cpu)
    a.to(cuda1)
    torch.cuda.synchronize(cuda1)

    # 测试从GPU到GPU数据加载时间
    a = torch.randn((80, 32, 4096), device=cuda2)
    a.to(cuda1)
    torch.cuda.synchronize(cuda1)
    torch.cuda.synchronize(cuda2)

    # 测试从GPU到CPU数据加载时间
    a = torch.randn((80, 32, 4096), device=cuda1)
    a.to(cpu)
    torch.cuda.synchronize(cuda1)

    # 90
    # 测试从CPU到GPU数据加载时间
    a = torch.randn((90, 32, 4096), device=cpu)
    a.to(cuda1)
    torch.cuda.synchronize(cuda1)

    # 测试从GPU到GPU数据加载时间
    a = torch.randn((90, 32, 4096), device=cuda2)
    a.to(cuda1)
    torch.cuda.synchronize(cuda1)
    torch.cuda.synchronize(cuda2)

    # 测试从GPU到CPU数据加载时间
    a = torch.randn((90, 32, 4096), device=cuda1)
    a.to(cpu)
    torch.cuda.synchronize(cuda1)

    # 100
    # 测试从CPU到GPU数据加载时间
    a = torch.randn((100, 32, 4096), device=cpu)
    a.to(cuda1)
    torch.cuda.synchronize(cuda1)

    # 测试从GPU到GPU数据加载时间
    a = torch.randn((100, 32, 4096), device=cuda2)
    a.to(cuda1)
    torch.cuda.synchronize(cuda1)
    torch.cuda.synchronize(cuda2)

    # 测试从GPU到CPU数据加载时间
    a = torch.randn((100, 32, 4096), device=cuda1)
    a.to(cpu)
    torch.cuda.synchronize(cuda1)

if __name__ == '__main__':
    communicate()
