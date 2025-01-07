import torch


def train_test_val_split(
    data: str, train_size: float = 0.8, val_size: float = 0.1, test_size: float = 0.1
):
    assert train_size + val_size + test_size == 1
    data_len = len(data)
    train_len = int(data_len * train_size)
    val_len = int(data_len * val_size)
    test_len = data_len - train_len - val_len
    train_data = data[:train_len]
    val_data = data[train_len : train_len + val_len]
    test_data = data[train_len + val_len :]
    return train_data, val_data, test_data


# Data loading
def get_batch(split, train_data, val_data, block_size, batch_size, device):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# Get loss func
@torch.no_grad()
def estimate_loss(model, eval_iters, train_data, val_data,
                  block_size, batch_size, device):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)  # eval_iters: number of iterations
        for i in range(eval_iters):
            x, y = get_batch(split, train_data, val_data,
                             block_size, batch_size, device)
            logits, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
