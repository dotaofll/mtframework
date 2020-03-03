import torch
import torch.nn as nn

input_size =4
hidden_size =4
output_size=10
embedd = nn.Embedding(12,4)
soft = nn.LogSoftmax(dim=1)

a=[1,2,3]
b=[4,5,6]

s_a="hello world"
s_b="jdiosaudg asjdoi"
t = (s_a,s_b)
s_a_t= torch.Tensor(s_a)
a_t = torch.tensor(a,dtype=torch.long, device='cpu').view(-1, 1)
b_t = torch.tensor(b,dtype=torch.long, device='cpu').view(-1, 1)

l = [a_t,b_t]

longt= torch.stack(l)
print(longt.size())
print(1)