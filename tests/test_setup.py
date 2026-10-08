import torch


def test_autograd_matches_calculus():
    x = torch.tensor(3.0, requires_grad=True)
    (x**2).backward()
    assert x.grad.item() == 6.0