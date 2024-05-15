import torch
s1 = torch.cuda.stream()
s2 = torch.cuda.stream()

A = torch.rand(1000,1000,device="cuda")
B = torch.rand(1000,1000,device="cuda")

torch.cuda.synchronize()
with torch.cuda.stream(s1):
    C = torch.mm(A,A)
with torch.cuda.stream(s2):
    D = torch.mm(B,B)

torch.cuda.synchronize()

print(C)
print(D)
