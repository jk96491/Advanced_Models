import torch
import torch.nn as nn
from Modules.scaled_dot_product_attention_layer import scaled_dot_product_attention


class multi_head_attention(nn.Module):
    def __init__(self, input_dim, n_head):
        super(multi_head_attention, self).__init__()
        self.n_head = n_head
        self.out_dim = input_dim // 2

        self.query_layers = nn.ModuleList()
        self.key_layers = nn.ModuleList()
        self.value_layers = nn.ModuleList()

        for i in range(self.n_head):
            self.query_layers.append(nn.Conv2d(input_dim, input_dim // 2 , kernel_size=1))
            self.key_layers.append(nn.Conv2d(input_dim, input_dim // 2, kernel_size=1))
            self.value_layers.append(nn.Conv2d(input_dim, input_dim, kernel_size=1))

        self.fc = nn.Linear(self.out_dim * self.out_dim * self.n_head, self.out_dim * self.out_dim)

        self.scaled_dot_product_attention_layer = scaled_dot_product_attention(input_dim // 2)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        batch_size, channels, width, height = inputs.size()

        attention_output = []
        attention_score_output = []

        for i in range(self.n_head):
            query = self.query_layers[i](inputs).view(batch_size, -1, width * height).permute(0, 2, 1)
            key = self.key_layers[i](inputs).view(batch_size, -1, width * height)
            value = self.value_layers[i](inputs).view(batch_size, -1, width * height)

            output, attention_score = self.scaled_dot_product_attention_layer(inputs, query, key, value)

            attention_output.append(output)
            attention_score_output.append(attention_score)

        attention_output = torch.cat(attention_output, dim=-1)
        attention_score_output = torch.cat(attention_score_output, dim=-1)

        output = self.fc(attention_output)
        output = output.view(batch_size, channels, self.out_dim, self.out_dim)
        output = self.gamma * output + inputs

        return output, attention_score_output
