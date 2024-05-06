import torch

# 创建两个随机的 BF16 张量
tensor1 = torch.randn(3, 3, device='cpu',dtype=torch.bfloat16)
tensor2 = torch.randn(3, 3,device='cpu', dtype=torch.bfloat16)

# 输出两个张量
print("Tensor 1:")
print(tensor1)
print("\nTensor 2:")
print(tensor2)

# 计算两个张量的乘积
result = torch.mul(tensor1, tensor2)

# 输出结果
print("\nResult of multiplication:")
print(result)