import torch
import torch.nn.functional as F
import torch.optim as optim
from models import PolicyTransformer
import random

# Simulate multi-agent communication episodes for eye-for-eye policy cloning
# Each agent copies its previous action; we train the model to clone this policy using q-bc

def sample_eye_for_eye_episodes(batch_size, seq_len, id_dim, n_agents=4):
    token_dim = id_dim + 10
    batch_tokens = torch.zeros((batch_size, seq_len, token_dim), dtype=torch.float32)
    labels = torch.zeros((batch_size, seq_len), dtype=torch.long)
    agent_ids = [torch.randint(0, 2, (id_dim,), dtype=torch.float32) for _ in range(n_agents)]
    for b in range(batch_size):
        # For each agent, track its previous action
        prev_actions = [random.randrange(2) for _ in range(n_agents)]
        for t in range(seq_len):
            # Pick an agent for this timestep (simulate communication)
            agent_idx = random.randrange(n_agents)
            id_vec = agent_ids[agent_idx]
            token = torch.zeros((token_dim,), dtype=torch.float32)
            token[:id_dim] = id_vec
            # encode previous action for this agent
            if prev_actions[agent_idx] == 0:
                token[-2] = 1.0
            else:
                token[-1] = 1.0
            batch_tokens[b, t] = token
            labels[b, t] = prev_actions[agent_idx]
            # sample current action for next step
            prev_actions[agent_idx] = random.randrange(2)
    return batch_tokens, labels


def train_eye_for_eye_qbc():
    device = 'cuda:6'
    id_dim = 64
    token_dim = id_dim + 10
    seq_len = 128
    model = PolicyTransformer(token_dim=token_dim, d_model=128, nhead=8, num_layers=4, max_len=seq_len + 1, mode='qnet').to(device)
    opt = optim.Adam(model.parameters(), lr=1e-4)
    
    n_agents = 4
    
    batch_size = 32
    epochs = 5000

    for epoch in range(1, epochs + 1):
        model.train()
        tokens, labels = sample_eye_for_eye_episodes(batch_size, seq_len, id_dim, n_agents)
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
    tokens, labels = sample_eye_for_eye_episodes(8, seq_len, id_dim, n_agents)
    with torch.no_grad():
        logits = model.forward(tokens)
        preds = logits.argmax(dim=-1)
    heldout_acc = (preds == labels).float().mean().item()
    print(f"\nHeld-out accuracy: {heldout_acc:.4f}")
    for i in range(preds.size(0)):
        print('seq', i, 'preds:', preds[i].tolist()[:20], 'labels:', labels[i].tolist()[:20])
    # summary counts
    final_labels = labels.view(-1)
    final_preds = preds.view(-1)
    final_n_pos_labels = int((final_labels == 1).sum().item())
    final_n_neg_labels = int((final_labels == 0).sum().item())
    final_n_pos_preds = int((final_preds == 1).sum().item())
    final_n_neg_preds = int((final_preds == 0).sum().item())
    print(f"Held-out counts labels(+/-)={final_n_pos_labels}/{final_n_neg_labels} preds(+/-)={final_n_pos_preds}/{final_n_neg_preds}")


if __name__ == '__main__':
    train_eye_for_eye_qbc()
