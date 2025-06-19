import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.conv_i = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=self.padding, bias=bias)
        self.conv_o = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=self.padding, bias=bias)

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        combined_conv = self.conv_i(combined)
        i, f, g, o = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, bias=True):
        super(ConvLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.bias = bias

        cell_list = []
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim[i - 1]
            cell_list.append(ConvLSTMCell(cur_input_dim, hidden_dim[i], kernel_size, bias))

        self.cell_list = nn.ModuleList(cell_list)
        self.transform_pred = nn.Conv2d(hidden_dim[-1], 6, kernel_size=1)

    def forward(self, input_seq):
        batch_size, seq_len, _, height, width = input_seq.size()
        hidden_states = []
        cell_states = []

        for i in range(self.num_layers):
            hidden_states.append(torch.zeros(batch_size, self.hidden_dim[i], height, width).to(input_seq.device))
            cell_states.append(torch.zeros(batch_size, self.hidden_dim[i], height, width).to(input_seq.device))

        transform_preds = []
        h = input_seq[:, 0, :, :, :]

        for t in range(seq_len):
            x = input_seq[:, t, :, :, :] if t > 0 else h
            for i in range(self.num_layers):
                h, c = self.cell_list[i](x, hidden_states[i], cell_states[i])
                hidden_states[i] = h
                cell_states[i] = c
                x = h

            # transform_pred = self.transform_pred(h)
            # transform_pred = torch.nn.functional.avg_pool2d(transform_pred,
            #                                                 kernel_size=transform_pred.size()[2:]).squeeze()
            # transform_preds.append(transform_pred)
            transform_pred = self.transform_pred(h)
            transform_pred = torch.nn.functional.avg_pool2d(transform_pred,
                                                            kernel_size=transform_pred.size()[2:])
            transform_pred = transform_pred.view(batch_size, -1)  # [B, 6]
            transform_preds.append(transform_pred)

        return torch.stack(transform_preds, dim=1)