"""Quick smoke tests for utils.lambda_return

Run this from the repository root (works on Windows powershell):
    python run_lambda_test.py
"""
import torch
from utils import lambda_return


def test_lambda_zero_equals_td0():
    T, B, A = 5, 3, 2
    rewards = torch.arange(T*B, dtype=torch.float32).view(T, B) * 0.1
    dones = torch.zeros(T, B)
    q_next = torch.ones(T, B) * 2.0
    G = lambda_return(rewards, dones, q_next, gamma=0.9, lam=0.0)
    # TD(0): r_t + gamma * q_next[t]
    td0 = rewards + 0.9 * q_next
    assert torch.allclose(G, td0), f"lam=0 mismatch\n{G}\n{td0}"
    print("test_lambda_zero_equals_td0 passed")


def test_lambda_sequence_shape_and_device():
    T, B = 7, 4
    rewards = torch.randn(T, B, device='cpu')
    dones = (torch.rand(T, B) < 0.2).float()
    q_next = torch.randn(T+1, B)
    G = lambda_return(rewards, dones, q_next, gamma=0.99, lam=0.95)
    assert G.shape == (T, B), f"unexpected shape: {G.shape}"
    assert G.device == rewards.device
    print("test_lambda_sequence_shape_and_device passed")


if __name__ == '__main__':
    test_lambda_zero_equals_td0()
    test_lambda_sequence_shape_and_device()
    print("All lambda_return tests passed")
