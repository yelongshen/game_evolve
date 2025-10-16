
import random
import copy
import torch


# --- PD payoff ---
# actions: 1 == cooperate (C), 0 == defect (D)
CO_PAYOFF = {
    (1, 1): (3.0, 3.0),
    (1, 0): (-2.0, 2.0),
    (0, 1): (2.0, -2.0),
    (0, 0): (-1.0, -1.0),
}

JAIL_PAYOFF = {
    (1, 1): (1.0, 1.0),
    (1, 0): (-1.0, 2.0),
    (0, 1): (2.0, -1.0),
    (0, 0): (0.0, 0.0),
}

JAIL2_PAYOFF = {
    (1, 1): (1.0, 1.0),
    (1, 0): (-0.2, 1.2),
    (0, 1): (1.2, -0.2),
    (0, 0): (0.0, 0.0),
}

DEFECTION_PAYOFF = {
    (1, 1): (1.0, 1.0),
    (1, 0): (-3.0, 2.0),
    (0, 1): (2.0, -3.0),
    (0, 0): (0.0, 0.0),
}

ALT_PAYOFF = {
    (1, 1): (4.0, 4.0),
    (1, 0): (-1.0, 5.0),
    (0, 1): (5.0, -1.0),
    (0, 0): (0.0, 0.0),
}


PAYOFF_TYPES = {
    'co': CO_PAYOFF,
    'jail': JAIL_PAYOFF,
    'defect': DEFECTION_PAYOFF,
    'alt': ALT_PAYOFF,
    'jail2': JAIL2_PAYOFF,
}

class RandomPayoff:
    def __init__(self, payoff_tables):
        self.payoff_tables = list(payoff_tables.values())
    def __getitem__(self, key):
        table = random.choice(self.payoff_tables)
        return table[key]


def get_payoff(payoff_type='co'):
    if payoff_type == 'random':
        return RandomPayoff(PAYOFF_TYPES)
    return PAYOFF_TYPES.get(payoff_type, CO_PAYOFF)

def action_to_bits(a, device):
    # None -> [0,0] (no action yet), 1 -> [0,1] cooperate, 0 -> [1,0] defect
    if a is None:
        return torch.tensor([0.0, 0.0], device=device)
    a_int = int(a)
    if a_int == 1:
        return torch.tensor([0.0, 1.0], device=device)
    else:
        return torch.tensor([1.0, 0.0], device=device)


def reward_to_6bits(r, device):
    """Encode numeric reward into 6-bit two's-complement (MSB-first) on `device`.

    We quantize by rounding to nearest integer and clip to [-32, 31] which fits
    in 6-bit two's complement: range -32..31.
    """
    r_int = int(round(float(r)))
    r_clipped = max(-32, min(31, r_int))
    unsigned = r_clipped & 0x3F
    bits = [((unsigned >> (5 - i)) & 1) for i in range(6)]
    return torch.tensor(bits, dtype=torch.float32, device=device)


def lambda_return(rewards, dones, q_bootstrap, gamma=0.99, lam=0.95):
    """
    Compute G^lambda targets for a sequence (vectorized over batch).

    Args:
        rewards:    Tensor [T, B]
        dones:      Tensor [T, B] (1.0 if terminal at t)
        q_bootstrap:Tensor [T, B] or [T+1, B]. If shape is [T, B], each
                     element is the 1-step bootstrap value for the next
                     state aligned per-step (i.e. q_bootstrap[t] == Q(s_{t+1})).
                     If shape is [T+1, B], it's interpreted such that
                     q_bootstrap[t+1] == Q(s_{t+1}) and q_bootstrap[0]
                     is ignored.
        gamma:      discount factor in (0,1)
        lam:        lambda in [0,1)

    Returns:
        Tensor G of shape [T, B] with the lambda-returns.

    Notes:
        - q_bootstrap is detached inside the function (treated as target).
        - Operates in the same dtype/device as `rewards`.
    """
    # Basic checks
    if rewards.dim() != 2 or dones.dim() != 2:
        raise ValueError("rewards and dones must be 2-D tensors [T,B]")
    T, B = rewards.shape

    # Normalize types/devices
    device = rewards.device
    dtype = rewards.dtype
    dones = dones.to(device=device, dtype=dtype)

    qb = q_bootstrap.detach()  # treat bootstrap as target (no grad)
    if qb.device != device:
        qb = qb.to(device=device)
    if qb.dtype != dtype:
        qb = qb.to(dtype=dtype)

    # Accept either [T,B] or [T+1,B]
    if qb.shape[0] == T:
        # qb[t] corresponds to Q(s_{t+1}) aligned with rewards[t]
        q_next = qb
        q_last = qb[-1]
    elif qb.shape[0] == T + 1:
        # qb[t+1] corresponds to Q(s_{t+1}); shift for convenience
        q_next = qb[1:]
        q_last = qb[-1]
    else:
        raise ValueError("q_bootstrap must have shape [T,B] or [T+1,B]")

    G = torch.zeros_like(rewards, device=device, dtype=dtype)
    # start from last step's 1-step piece
    next_G = q_last
    gamma = torch.tensor(float(gamma), device=device, dtype=dtype)
    lam = torch.tensor(float(lam), device=device, dtype=dtype)

    for t in reversed(range(T)):
        nonterm = 1.0 - dones[t]
        # TD(0) target at step t using q_next[t] == Q(s_{t+1})
        td0 = rewards[t] + gamma * nonterm * q_next[t]
        # backward lambda recursion
        next_G = td0 + gamma * lam * nonterm * (next_G - q_next[t])
        G[t] = next_G

    return G
