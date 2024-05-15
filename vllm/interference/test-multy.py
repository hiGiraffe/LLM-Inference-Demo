import concurrent.futures
from line_profiler import LineProfiler

def add(a, b):
    for i in range(100000000):
        a=a+b
    return a

# 使用 ThreadPoolExecutor 进行并发执行
@profile
def test():
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future1 = executor.submit(add, 2,1)
        future2 = executor.submit(add, 3,1)

        # 获取结果
        ans_1 = future1.result()
        ans_2 = future2.result()

        print(ans_1,ans_2)

if __name__ == '__main__':
    test()