import torch
import torch.nn.functional as F
import torch.optim as optim
from models import PolicyTransformer
import random

# Synthetic task: "eye for eye" policy — the model should output the previous
# action taken by the agent at the immediately preceding timestep. Actions are
# discrete {0,1}. We provide the previous action as part of the token features
# so the model must learn to copy that feature to the current prediction.


def sample_batch_eye_for_eye(batch_size, seq_len, id_dim, reuse_prob=0.5):
    token_dim = id_dim + 10
    batch_tokens = torch.zeros((batch_size, seq_len, token_dim), dtype=torch.float32)
    labels = torch.zeros((batch_size, seq_len), dtype=torch.long)
    for b in range(batch_size):
        # randomly choose initial previous action for t=0 (we define label for t=0 as 0)
        prev_action = random.randrange(2)
        for t in range(seq_len):
            # put a random id (not needed for eye_for_eye but kept for consistency)
            id_vec = (torch.randint(0, 2, (id_dim,), dtype=torch.float32) * 1.0)
            token = torch.zeros((token_dim,), dtype=torch.float32)
            token[:id_dim] = id_vec
            # encode prev_action in a small slot of the token (use one-hot in last dims)
            # use token[-2:] as one-hot for previous action
            if prev_action == 0:
                token[-2] = 1.0
            else:
                token[-1] = 1.0
            batch_tokens[b, t] = token
            # label is the previous action (the policy copies it)
            labels[b, t] = prev_action
            # sample current action for next step (this simulates environment history)
            # for training we don't expose current action; we simulate that later steps will
            # see this as previous action.
            prev_action = random.randrange(2)
    return batch_tokens, labels


def train_eye_for_eye():
    device = 'cuda:6'
    id_dim = 64
    token_dim = id_dim + 10
    seq_len = 128
    model = PolicyTransformer(token_dim=token_dim, d_model=128, nhead=8, num_layers=4, max_len=seq_len + 1, mode='qnet').to(device)
    opt = optim.Adam(model.parameters(), lr=1e-4)

    batch_size = 16
    epochs = 100000

    for epoch in range(1, epochs + 1):
        model.train()
        tokens, labels = sample_batch_eye_for_eye(batch_size, seq_len, id_dim)
        tokens = tokens.to(device)
        labels = labels.to(device)

        logits = model.forward(tokens)  # (batch, seq_len, n_actions)
        b, s, c = logits.shape
        loss = F.cross_entropy(logits.view(b * s, c), labels.view(b * s))

        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % 50 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                logits = model.forward(tokens)
                preds = logits.argmax(dim=-1)
                acc = (preds == labels).float().mean().item()
                labels_flat = labels.view(-1)
                preds_flat = preds.view(-1)
                n_pos_labels = int((labels_flat == 1).sum().item())
                n_neg_labels = int((labels_flat == 0).sum().item())
                n_pos_preds = int((preds_flat == 1).sum().item())
                n_neg_preds = int((preds_flat == 0).sum().item())
            print(f"epoch {epoch:04d} loss={loss.item():.4f} acc={acc:.4f} labels(+/-)={n_pos_labels}/{n_neg_labels} preds(+/-)={n_pos_preds}/{n_neg_preds}")

    # final evaluation on held-out sample
    model.eval()
    tokens, labels = sample_batch_eye_for_eye(8, seq_len, id_dim)
    with torch.no_grad():
        logits = model.forward(tokens)
        preds = logits.argmax(dim=-1)
        heldout_acc = (preds == labels).float().mean().item()
        print(f"Held-out accuracy: {heldout_acc:.4f}")
    # summary counts
    final_labels = labels.view(-1)
    final_preds = preds.view(-1)
    final_n_pos_labels = int((final_labels == 1).sum().item())
    final_n_neg_labels = int((final_labels == 0).sum().item())
    final_n_pos_preds = int((final_preds == 1).sum().item())
    final_n_neg_preds = int((final_preds == 0).sum().item())
    print(f"Held-out counts labels(+/-)={final_n_pos_labels}/{final_n_neg_labels} preds(+/-)={final_n_pos_preds}/{final_n_neg_preds}")


if __name__ == '__main__':
    train_eye_for_eye()
