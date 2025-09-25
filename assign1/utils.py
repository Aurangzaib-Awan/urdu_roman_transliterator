# utils.py

import torch

# count trainable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# save model
def save_checkpoint(model, path="seq2seq_model.pt"):
    torch.save(model.state_dict(), path)

# load model
def load_checkpoint(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    return model

# convert tensor ids back to tokens
def tensor_to_text(tensor, vocab):
    tokens = [vocab.lookup_token(idx) for idx in tensor.tolist()]
    tokens = [tok for tok in tokens if tok not in ["<sos>", "<eos>", "<pad>"]]
    return " ".join(tokens)

# compute perplexity from loss
def calculate_perplexity(loss):
    return torch.exp(torch.tensor(loss)).item()
