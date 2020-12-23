import torch
import torch.nn as nn
from Modules.scaled_dot_product_attention_layer import scaled_dot_product_attention


class multi_head_attention(nn.Module):
    def __init__(self, input_dim, n_head):
        super(multi_head_attention, self).__init__()
        self.n_head = n_head
        self.out_dim = input_dim // 2

        self.query = nn.Conv2d(input_dim, input_dim // 2 * n_head, kernel_size=1)
        self.key = nn.Conv2d(input_dim, input_dim // 2 * n_head, kernel_size=1)
        self.value = nn.Conv2d(input_dim, input_dim * n_head, kernel_size=1)

        self.fc = nn.Linear(input_dim * input_dim, self.out_dim * self.out_dim)

        self.scaled_dot_product_attention_layer = scaled_dot_product_attention(input_dim // 2)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        batch_size, channels, width, height = inputs.size()

        query = self.query(inputs).view(batch_size, -1, width * height * self.n_head).permute(0, 2, 1)
        key = self.key(inputs).view(batch_size, -1, width * height * self.n_head)
        value = self.value(inputs).view(batch_size, -1, width * height * self.n_head)

        output, attention_score = self.scaled_dot_product_attention_layer(inputs, query, key, value)
        output = self.fc(output)
        output = output.view(batch_size, channels, self.out_dim, self.out_dim)
        output = self.gamma * output + inputs

        return output, attention_score
