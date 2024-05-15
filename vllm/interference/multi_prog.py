import asyncio

async def func1():
    print("Function 1 is running")
    for i in range(5):
        print(f"Function 1: {i}")
        await asyncio.sleep(1)

async def func2():
    print("Function 2 is running")
    for i in range(5):
        print(f"Function 2: {i}")
        await asyncio.sleep(1)

async def main():
    # 并发运行两个函数
    await asyncio.gather(func1(), func1())

# 运行事件循环
asyncio.run(main())

print("Both functions are done")