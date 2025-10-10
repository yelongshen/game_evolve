import torch
import torch.nn.functional as F
import torch.optim as optim
from models import PolicyTransformer
import random

# Synthetic task: predict whether the agent_id at timestep t has been seen earlier in the same sequence.
# label: 1 if seen before, 0 if first occurrence.

def make_id_pool(id_dim, pool_size):
    # create a small pool of distinct id vectors (binary 0/1 floats)
    pool = []
    for _ in range(pool_size):
        v = (torch.randint(0, 2, (id_dim,), dtype=torch.float32) * 1.0)
        pool.append(v)
    return pool


def sample_batch(batch_size, seq_len, id_dim, pool, reuse_prob=0.5):
    # pool: list of id vectors
    token_dim = id_dim + 10
    batch_tokens = torch.zeros((batch_size, seq_len, token_dim), dtype=torch.float32)
    labels = torch.zeros((batch_size, seq_len), dtype=torch.long)
    for b in range(batch_size):
        seen = []
        for t in range(seq_len):
            if t == 0:
                # first always new
                idx = random.randrange(len(pool))
                seen.append(idx)
            else:
                if random.random() < reuse_prob and len(seen) > 0:
                    # reuse a previous id
                    idx = random.choice(seen)
                else:
                    idx = random.randrange(len(pool))
                    seen.append(idx)
            id_vec = pool[idx]
            # token: [id_vec (id_dim), padding for action/reward bits]
            token = torch.zeros((token_dim,), dtype=torch.float32)
            token[:id_dim] = id_vec
            batch_tokens[b, t] = token
            # label 1 if this id index has appeared before in the same sequence (earlier index)
            labels[b, t] = 1 if idx in seen[:-1] else 0
    return batch_tokens, labels


def train_overfit_small():
    device = 'cuda:2'
    id_dim = 64
    token_dim = id_dim + 10
    pool = make_id_pool(id_dim, pool_size=40000)
    seq_len = 128

    # small transformer
    model = PolicyTransformer(token_dim=token_dim, d_model=512, nhead=8, num_layers=8, max_len=seq_len, mode='qnet').to(device)
    opt = optim.Adam(model.parameters(), lr=1e-4)

    batch_size = 16
    epochs = 100000

    for epoch in range(1, epochs + 1):
        model.train()
        tokens, labels = sample_batch(batch_size, seq_len, id_dim, pool, reuse_prob=0.4)
        tokens = tokens.to(device)
        labels = labels.to(device)
        
        if epoch % 20 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                logits = model.forward(tokens)
                preds = logits.argmax(dim=-1)
                acc = (preds == labels).float().mean().item()
                # count label distribution and prediction distribution
                labels_flat = labels.view(-1)
                preds_flat = preds.view(-1)
                n_pos_labels = int((labels_flat == 1).sum().item())
                n_neg_labels = int((labels_flat == 0).sum().item())
                n_pos_preds = int((preds_flat == 1).sum().item())
                n_neg_preds = int((preds_flat == 0).sum().item())
            model.train()

        logits = model.forward(tokens)  # (batch, seq_len, n_actions)
        # we want to predict action 1 if seen before else 0
        # cross_entropy expects (N, C) and targets (N,)
        b, s, c = logits.shape
        loss = F.cross_entropy(logits.view(b * s, c), labels.view(b * s))

        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % 20 == 0 or epoch == 1:
            print(f"epoch {epoch:03d} loss={loss.item():.4f} acc={acc:.4f} labels(+/-)={n_pos_labels}/{n_neg_labels} preds(+/-)={n_pos_preds}/{n_neg_preds}")

    # final evaluation on held-out sample
    model.eval()
    tokens, labels = sample_batch(8, seq_len, id_dim, pool, reuse_prob=0.6)
    tokens = tokens.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        logits = model.forward(tokens)
        preds = logits.argmax(dim=-1)
    # final counts for held-out sample
    final_labels = labels.view(-1)
    final_preds = preds.view(-1)
    final_n_pos_labels = int((final_labels == 1).sum().item())
    final_n_neg_labels = int((final_labels == 0).sum().item())
    final_n_pos_preds = int((final_preds == 1).sum().item())
    final_n_neg_preds = int((final_preds == 0).sum().item())
    print(f"Held-out counts labels(+/-)={final_n_pos_labels}/{final_n_neg_labels} preds(+/-)={final_n_pos_preds}/{final_n_neg_preds}")

if __name__ == '__main__':
    train_overfit_small()
