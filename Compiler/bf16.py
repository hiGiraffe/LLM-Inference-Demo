import torch
import timeit

tensor1 = torch.full((1000, 1000), 0.00001, device='cpu',dtype=torch.bfloat16)
tensor2 = torch.full((1000, 1000), 0.00001,device='cpu', dtype=torch.bfloat16)

start_time = timeit.default_timer()

result = torch.mul(tensor1, tensor2)
elapsed_time = timeit.default_timer() - start_time

print("time ",elapsed_time)
