import torch
import random
from collections import deque

class BasicAgent:
    def __init__(self, id_, history_len=4, id_dim=16, agent_id=None):
        self.id = id_
        self.history_len = history_len
        self.id_dim = id_dim
        self.token_dim = id_dim + 10
        self.death_age = int(history_len * (random.uniform(0.8, 0.95)))
        self.agent_id = agent_id
        self.history = deque(maxlen=history_len)
        self.local_logps = []
        self.local_values = []
        
    def observe_and_store(self, other_agent_id, a_self, a_other, r, logp, value):
        if isinstance(other_agent_id, torch.Tensor):
            other_id_copy = other_agent_id.detach()
        else:
            other_id_copy = torch.tensor(other_agent_id, dtype=torch.float32)
        self.history.append((other_id_copy, a_self, a_other, float(r)))
        # logp expected to be a tensor
        try:
            self.local_logps.append(float(logp.detach().cpu().item()))
        except Exception:
            try:
                self.local_logps.append(float(logp))
            except Exception:
                self.local_logps.append(0.0)

        # value may be a tensor scalar, or a tuple (mu, logvar) for qnet
        try:
            if isinstance(value, tuple) and len(value) == 2:
                mu, logvar = value
                # extract predicted mean for the taken action
                try:
                    val_scalar = float(mu[a_self].detach().cpu().item())
                except Exception:
                    # mu may already be a Python sequence
                    val_scalar = float(mu[a_self])
            else:
                val_scalar = float(value.detach().cpu().item())
        except Exception:
            try:
                val_scalar = float(value)
            except Exception:
                val_scalar = 0.0
        self.local_values.append(val_scalar)
        return None

class AlwaysDefectAgent(BasicAgent):
    """A minimal agent that always plays action 0 (defect).

    Implements the small interface expected by PopulationEnv:
      - attributes: agent_id, history (deque), local_actions, local_logps, local_rewards, local_values
      - methods: act(partner_id) -> (action:int, logp:Tensor, value:Tensor)
                 observe_and_store(other_agent_id, a_self, a_other, r, logp, value)
    """
    def __init__(self, id_, history_len=4, id_dim=16, agent_id=None):
        super().__init__(id_, history_len, id_dim, agent_id)

    def act(self, partner_id):
        # always defect (0). Return a logp and value scalar tensors for compatibility.
        # logp: return 0.0 as a float-tensor (log prob placeholder)
        # value: zero tensor
        logp = torch.tensor(0.0, dtype=torch.float32)
        value = torch.tensor(0.0, dtype=torch.float32)
        a = 0
        return a, logp, value

    # observe_and_store inherited from BasicAgent


class AlwaysCooperateAgent(BasicAgent):
    """A minimal agent that always plays action 1 (cooperate).

    Implements the small interface expected by PopulationEnv:
      - attributes: agent_id, history (deque)
      - methods: act(partner_id) -> (action:int, logp:Tensor, value:Tensor)
                 observe_and_store(other_agent_id, a_self, a_other, r, logp, value)
    """
    def __init__(self, id_, history_len=4, id_dim=16, agent_id=None):
        super().__init__(id_, history_len, id_dim, agent_id)

    def act(self, partner_id):
        # always cooperate (1). Return a logp and value scalar tensors for compatibility.
        # logp: return 0.0 as a float-tensor (log prob placeholder)
        # value: zero tensor
        logp = torch.tensor(0.0, dtype=torch.float32)
        value = torch.tensor(0.0, dtype=torch.float32)
        a = 1
        return a, logp, value

    # observe_and_store inherited from BasicAgent
    
class RandomAgent(BasicAgent):
    """A minimal agent that plays random actions.

    Implements the small interface expected by PopulationEnv:
      - attributes: agent_id, history (deque)
      - methods: act(partner_id) -> (action:int, logp:Tensor, value:Tensor)
                 observe_and_store(other_agent_id, a_self, a_other, r, logp, value)
    """
    def __init__(self, id_, history_len=4, id_dim=16, agent_id=None, agent_id_generator=None):
        super().__init__(id_, history_len, id_dim, agent_id)

    def act(self, partner_id):
        # random action (0 or 1). Return a logp and value scalar tensors for compatibility.
        # logp: return 0.0 as a float-tensor (log prob placeholder)
        # value: zero tensor
        logp = torch.tensor(0.0, dtype=torch.float32)
        value = torch.tensor(0.0, dtype=torch.float32)
        a = random.choice([0, 1])
        return a, logp, value

    # observe_and_store inherited from BasicAgent

class EyeForEyeAgent(BasicAgent):
    """A minimal agent that plays "eye for eye" strategy: start cooperating, then mimic opponent's last action.
    Implements the small interface expected by PopulationEnv:
      - attributes: agent_id, history (deque)
      - methods: act(partner_id) -> (action:int, logp:Tensor, value:Tensor)
                 observe_and_store(other_agent_id, a_self, a_other, r, logp, value)
    """
    def __init__(self, id_, history_len=4, id_dim=16, agent_id=None, agent_id_generator=None):
        super().__init__(id_, history_len, id_dim, agent_id)

    def act(self, partner_id):
        # eye-for-eye action: cooperate first (1), then mimic opponent's last action.
        # Return a logp and value scalar tensors for compatibility.
        # logp: return 0.0 as a float-tensor (log prob placeholder)
        # value: zero tensor
        logp = torch.tensor(0.0, dtype=torch.float32)
        value = torch.tensor(0.0, dtype=torch.float32)
        # normalize partner_id to a tensor when possible for comparison
        partner = partner_id
        if not isinstance(partner_id, torch.Tensor):
            try:
                partner = torch.tensor(partner_id, dtype=torch.float32)
            except Exception:
                partner = partner_id

        # search history (most recent first) for an entry with matching other_id
        matched_action = 1  # default to cooperate if no match found
        for other_id_copy, _, last_a_other, _ in reversed(self.history):
            try:
                if isinstance(other_id_copy, torch.Tensor) and isinstance(partner, torch.Tensor):
                    if other_id_copy.shape == partner.shape and torch.equal(other_id_copy, partner):
                        matched_action = int(last_a_other)
                        break
                else:
                    if other_id_copy == partner:
                        matched_action = int(last_a_other)
                        break
            except Exception:
                # on any comparison error, skip this history entry
                continue

        if matched_action is not None:
            a = matched_action
        else:
            # fallback behavior: cooperate on first move, otherwise mimic most recent partner
            if len(self.history) == 0:
                a = 1
            else:
                a = int(self.history[-1][2])
        return a, logp, value

    # observe_and_store inherited from BasicAgent


class AggressiveEyeForEyeAgent(BasicAgent):
    """An agent that always defects (plays 0) against any partner who has ever played 0 with it in history. Otherwise, mimics eye-for-eye."""
    def __init__(self, id_, history_len=4, id_dim=16, agent_id=None, agent_id_generator=None):
        super().__init__(id_, history_len, id_dim, agent_id)

    def act(self, partner_id):
        logp = torch.tensor(0.0, dtype=torch.float32)
        value = torch.tensor(0.0, dtype=torch.float32)
        partner = partner_id
        if not isinstance(partner_id, torch.Tensor):
            try:
                partner = torch.tensor(partner_id, dtype=torch.float32)
            except Exception:
                partner = partner_id

        # Check if partner has ever played 0 with us
        has_defected = False
        for other_id_copy, _, last_a_other, _ in self.history:
            try:
                if isinstance(other_id_copy, torch.Tensor) and isinstance(partner, torch.Tensor):
                    if other_id_copy.shape == partner.shape and torch.equal(other_id_copy, partner):
                        if int(last_a_other) == 0:
                            has_defected = True
                            break
                else:
                    if other_id_copy == partner:
                        if int(last_a_other) == 0:
                            has_defected = True
                            break
            except Exception:
                continue

        if has_defected:
            a = 0
        else:
            a = 1
        return a, logp, value

    # observe_and_store inherited from BasicAgent