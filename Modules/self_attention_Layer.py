import torch
import torch.nn as nn
from Modules.scaled_dot_product_attention_layer import scaled_dot_product_attention


class self_attention(nn.Module):
    def __init__(self, input_dim):
        super(self_attention, self).__init__()

        self.query = nn.Conv2d(input_dim, input_dim // 2, kernel_size=1)
        self.key = nn.Conv2d(input_dim, input_dim // 2, kernel_size=1)
        self.value = nn.Conv2d(input_dim, input_dim, kernel_size=1)

        self.scaled_dot_product_attention_layer = scaled_dot_product_attention(input_dim // 2)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        batch_size, channels, width, height = inputs.size()

        query = self.query(inputs).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key(inputs).view(batch_size, -1, width * height)
        value = self.value(inputs).view(batch_size, -1, width * height)

        output, attention_score = self.scaled_dot_product_attention_layer(query, key, value)
        output = output.view(batch_size, channels, width, height)
        output = self.gamma * output + inputs

        return output, attention_score
