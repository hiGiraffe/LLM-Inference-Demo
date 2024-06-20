import torch
import torch.cuda
import threading
from line_profiler import LineProfiler

#@profile
def count():
    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()

    A = torch.rand(1000,1000,device="cuda")
    B = torch.rand(1000,1000,device="cuda")

    torch.cuda.synchronize()
    with torch.cuda.stream(s1):
        C = torch.mm(A,A)
    with torch.cuda.stream(s2):
        D = torch.mm(B,B)

    torch.cuda.synchronize()
    with torch.cuda.stream(s1):
        print(C)
    with torch.cuda.stream(s2):
        print(D)

if __name__ == '__main__':
    count()