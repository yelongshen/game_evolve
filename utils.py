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
}

def get_payoff(payoff_type='co'):
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
