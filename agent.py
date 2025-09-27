
from collections import deque, defaultdict
import torch
import random
from utils import action_to_bits, reward_to_6bits
from agents_simple import BasicAgent

class Agent(BasicAgent):
    def __init__(self, id_, shared_model, history_len=4, device="cpu", id_dim=32, agent_id=None, exploration_eps=0.01):
        super().__init__(id_, history_len, id_dim, agent_id)
        self.device = device
        self.shared_model = shared_model
        self.kv_cache = self.shared_model.build_cache(self.death_age)
        self.epsilon = float(exploration_eps)

    def act(self, partner_id):
        # Action selection (previously the acting path of step())
        if not isinstance(partner_id, torch.Tensor):
            partner = torch.tensor(partner_id, dtype=torch.float32)
        else:
            partner = partner_id
        partner = partner.to(self.device)

        if len(self.history) == 0:
            partner_token = torch.cat([partner, torch.zeros(10, device=self.device)])
        else:
            last_other_id, last_a_self, last_a_other, last_r = self.history[-1]
            partner_token = torch.cat([partner,
                                       action_to_bits(last_a_self, self.device),
                                       action_to_bits(last_a_other, self.device),
                                       reward_to_6bits(last_r, self.device)])

        pos_idx = len(self.history) 
        if self.shared_model.mode == 'vnet':
            prob, value = self.shared_model.forward_with_cache(self.kv_cache, partner_token.to(self.shared_model.pos_emb.device), position_idx=pos_idx)
            m = torch.distributions.Bernoulli(probs=prob)
            a = int(m.sample().item())
            logp = m.log_prob(torch.tensor(float(a), device=self.device))
            return a, logp, value
        elif self.shared_model.mode =='qnet':
            q_values = self.shared_model.forward_with_cache(self.kv_cache, partner_token.to(self.shared_model.pos_emb.device), position_idx=pos_idx)
            #print(q_values)
            #exit(0)
            q_values = q_values + torch.randn_like(q_values) * 0.01

            # q_values: tensor([... n_actions])
            # simple epsilon-greedy exploration to avoid getting stuck deterministic
            if random.random() < self.epsilon:
                a = int(random.randrange(0, q_values.size(-1)))
            else:
                a = int(torch.argmax(q_values).item())
            # compute logp as if from a softmax policy with temperature (for bookkeeping)
            temp = 5.0
            logits = q_values / temp
            logp_all = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
            # logp_all may be a 1D tensor; pick chosen action
            logp = logp_all[a]
            return a, logp, q_values[a]

