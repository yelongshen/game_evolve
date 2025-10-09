
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
        self.epsilon = 0.0 # float(exploration_eps)

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
        elif self.shared_model.mode == 'qnet':
            q_values = self.shared_model.forward_with_cache(self.kv_cache, partner_token.to(self.shared_model.pos_emb.device), position_idx=pos_idx)
            # Option: small noise for initial exploration (kept small); sampling from softmax
            #q_values = q_values + torch.randn_like(q_values) * 0.01
            temp = 0.3
            logits = q_values / temp
            probs = torch.softmax(logits, dim=-1)

            # epsilon still allows uniform random exploration
            if random.random() < self.epsilon:
                a = int(random.randrange(0, probs.size(-1)))
                # compute log-prob of uniform choice for bookkeeping
                logp = torch.log(probs.new_tensor(1.0 / float(probs.size(-1))))
            else:
                m = torch.distributions.Categorical(probs=probs)
                a_tensor = m.sample()
                a = int(a_tensor.item())
                logp = m.log_prob(a_tensor)

            return a, logp, q_values[a]
        
        elif self.shared_model.mode =='qnet-bay':
            mu, logvar = self.shared_model.forward_with_cache(self.kv_cache, partner_token.to(self.shared_model.pos_emb.device), position_idx=pos_idx)
            # add small noise for exploration in mu space
            mu = mu + torch.randn_like(mu) * 0.01

            # mu: tensor of shape (n_actions,)
            if random.random() < self.epsilon:
                a = int(random.randrange(0, mu.size(-1)))
            else:
                a = int(torch.argmax(mu).item())

            # Compute a surrogate log-prob for bookkeeping: softmax over mu with temperature
            temp = 5.0
            logits = mu / temp
            logp_all = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
            logp = logp_all[a]
            # return chosen action, logp, and the predicted distribution params for the chosen action
            # For qnet we return the predicted mean for the chosen action as the Q-value placeholder
            return a, logp, (mu, logvar)

