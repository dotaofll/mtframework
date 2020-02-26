import torch
import torch.nn as nn

input_size =4
hidden_size =4
output_size=10
embedd = nn.Embedding(12,4)


a= torch.zeros(1,1,5)
a= torch.randn(4,2,4)
s=torch.LongTensor([[1,2,3,4]])
input = torch.LongTensor([[1,2,4,5,6],[4,3,2,9,8]])
batch_size = input.size(0) #2
output_size = input.size(1) #4
word_to_ix = {'hello': 0, 'world': 1}
hello_idx = torch.LongTensor([word_to_ix['hello']])
hello_idx
world_idx = torch.LongTensor([word_to_ix['world']])
out = embedd(input)
input
world_idx
output=out.view(-1,4).view(batch_size,output_size,-1)
output[1]
a[:,:,-1]