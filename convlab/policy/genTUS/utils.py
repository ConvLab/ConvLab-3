import torch


def append_tokens(tokens, new_token, device):
    return torch.cat((tokens, torch.tensor([new_token]).to(device)), dim=1)
