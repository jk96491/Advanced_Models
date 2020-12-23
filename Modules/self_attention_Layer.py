import torch
import torch.nn as nn


class self_attention(nn.Module):
    def __init__(self, input_dim):
        super(self_attention, self).__init__()

        self.query = nn.Conv2d(input_dim, input_dim // 2, kernel_size=1)
        self.key = nn.Conv2d(input_dim, input_dim // 2, kernel_size=1)
        self.value = nn.Conv2d(input_dim, input_dim, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        batch_size, channels, width, height = inputs.size()

        query = self.query(inputs).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key(inputs).view(batch_size, -1, width * height)
        value = self.value(inputs).view(batch_size, -1, width * height)

        dot_product = torch.bmm(query, key)
        attention_score = self.softmax(dot_product)

        output = torch.bmm(value, attention_score.permute(0, 2, 1))
        output = output.view(batch_size, channels, width, height)

        output = self.gamma * output + inputs

        return output, attention_score
