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
    for _ in range(100000):
        a = torch.randn((2, 3), device=cuda2)
        a.to(cuda1)

    for _ in range(100000):
        a = torch.randn((2, 3), device=cpu)
        a.to(cuda1)

    torch.cuda.synchronize(cuda1)
    torch.cuda.synchronize(cuda2)

    # 第一次测试从CPU到GPU数据加载时间
    a = torch.randn((200, 300), device=cpu)
    a.to(cuda1)
    torch.cuda.synchronize(cuda1)


    # 第一次测试从GPU到GPU数据加载时间
    a = torch.randn((200, 300), device=cuda2)
    a.to(cuda1)
    torch.cuda.synchronize(cuda1)
    torch.cuda.synchronize(cuda2)

    # 第一次测试从GPU到CPU数据加载时间
    a = torch.randn((200, 300), device=cuda1)
    a.to(cpu)
    torch.cuda.synchronize(cuda1)

    # 第二次测试从CPU到GPU数据加载时间
    a = torch.randn((200, 300), device=cpu)
    a.to(cuda1)
    torch.cuda.synchronize(cuda1)

    # 第二次测试从GPU到GPU数据加载时间
    a = torch.randn((200, 300), device=cuda2)
    a.to(cuda1)
    torch.cuda.synchronize(cuda1)
    torch.cuda.synchronize(cuda2)

    # 第二次测试从GPU到CPU数据加载时间
    a = torch.randn((200, 300), device=cuda1)
    a.to(cpu)
    torch.cuda.synchronize(cuda1)

    # 第三次测试从CPU到GPU数据加载时间
    a = torch.randn((200, 300), device=cpu)
    a.to(cuda1)
    torch.cuda.synchronize(cuda1)

@profile
def test():
    communicate()

if __name__ == '__main__':
    test()