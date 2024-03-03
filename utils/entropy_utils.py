import torch


def shannon_entropy(A):
    A = A.squeeze()
    size = torch.tensor(float(A.size(0)))
    return - torch.sum(A * torch.log(A)) / torch.log(size)
