import torch
import torch.nn as nn


class scaled_dot_product_attention(nn.Module):
    def __init__(self, key_dim):
        super(scaled_dot_product_attention, self).__init__()
        self.key_dim = key_dim
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs, query, key, value):

        dot_product = torch.bmm(query, key)
        attention_score = self.softmax(dot_product)
        output = torch.bmm(value, attention_score.permute(0, 2, 1))

        return output, attention_score