import torch.nn as nn
import torch
import torch.functional as F


class Attention(nn.Module):
    def __init__(self, dim=1):
        super(Attention, self).__init__()
        self.linear = nn.Linear(dim*2, dim)

    def forward(self, output: torch.Tensor, context: torch.Tensor):
        batch_size = output.size(0)
        hidden_size = output.size(2)
        output_size = context.size(1)

        attn = torch.bmm(output, context.transpose(2, 1))
        attn = F.softmax(attn.view(-1, output_size),
                         dim=1).view(batch_size, -1, output_size)

        concat = torch.bmm(attn, context)
        combined = torch.cat((concat, output), dim=2)

        output = F.tanh(self.linear(combined.view(-1, 2*hidden_size))
                        ).view(batch_size, -1, hidden_size)

        return output, attn
