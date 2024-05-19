import torch
import torch.cuda
import concurrent.futures

from line_profiler import LineProfiler


@profile
def count():
    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()

    A = torch.rand(10000, 10000, device="cuda")
    B = torch.rand(10000, 10000, device="cuda")

    C = torch.mm(A, A)
    torch.cuda.synchronize()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        with torch.cuda.stream(s1):
            future1 = executor.submit(torch.mm, A, A)
        with torch.cuda.stream(s2):
            future2 = executor.submit(torch.mm, B, B)

        C = future1.result()
        D = future2.result()

        print(C)
        print(D)
        # with torch.cuda.stream(s1):
        #     C = torch.mm(A,A)
        #     print(C)
        # with torch.cuda.stream(s2):
        #     D = torch.mm(A,A)
        #     print(D)

    torch.cuda.synchronize()


if __name__ == '__main__':
    count()
