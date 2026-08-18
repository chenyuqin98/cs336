import torch


class softmax(torch.nn.Module):
    def forward(self, x, dim=-1):
        x_max = torch.max(x, dim=dim, keepdim=True).values
        x_exp = torch.exp(x - x_max)
        x_sum = torch.sum(x_exp, dim=dim, keepdim=True)
        return x_exp / x_sum
