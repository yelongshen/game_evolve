import torch
import numpy as np
from models import PolicyTransformer


def run_test(model, seq_len=256):
    """Run forward vs incremental forward_with_cache test on `model`.

    Args:
      model: a PolicyTransformer instance in 'qnet' mode
      seq_len: sequence length to test
    """
    torch.manual_seed(0)
    model.eval()

    token_dim = model.input_proj.in_features

    # create a batch of 1 sequence: shape (1, seq_len, token_dim)
    seq = torch.randn((1, seq_len, token_dim), dtype=torch.float32)

    # full forward
    with torch.no_grad():
        full_q = model.forward(seq)  # (batch, seq_len, n_actions)

    # incremental forward: feed tokens one-by-one using build_cache and forward_with_cache
    memories = model.build_cache(prealloc_len=seq_len)

    incr_outs = []
    for t in range(seq_len):
        # forward_with_cache expects a 2D tensor (new_seq_len, token_dim)
        new_tokens_2d = seq[0, t:t+1, :]
        with torch.no_grad():
            q_step = model.forward_with_cache(memories, new_tokens_2d, position_idx=t)
        # q_step is tensor (n_actions,)
        incr_outs.append(q_step.unsqueeze(0))

    incr_q = torch.cat(incr_outs, dim=0).unsqueeze(0)  # (1, seq_len, n_actions)

    # compare
    diff = (full_q - incr_q).abs()
    maxdiff = diff.max().item()
    print(f"max abs diff between full forward and incremental forward_with_cache: {maxdiff:.6e}")
    if maxdiff > 1e-5:
        # print per-step diffs
        print("Per-step differences:")
        for t in range(seq_len):
            f = full_q[0, t].cpu().numpy()
            i = incr_q[0, t].cpu().numpy()
            print(f"t={t} full={f} incr={i} diff={(np.abs(f-i)).tolist()}")
        raise SystemExit(2)
    else:
        print("PASS: forward and forward_with_cache outputs match within tolerance")


if __name__ == '__main__':
    # build a default small model for quick local runs
    default_seq_len = 64
    token_dim = 64 + 10
    model = PolicyTransformer(token_dim=token_dim, d_model=128, nhead=8, num_layers=4, max_len=default_seq_len + 1, mode='qnet')
    run_test(model, seq_len=default_seq_len)
