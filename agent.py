from collections import deque, defaultdict
 
import torch
import random
from utils import action_to_bits, reward_to_6bits

 
class Agent:
    def __init__(self, id_, shared_model, history_len=4, device="cpu", id_dim=32, agent_id=None,
                 pair_history_len=None, exploration_eps=0.01):
        self.id = id_
        self.history_len = history_len
        self.device = device
        self.id_dim = id_dim
        self.token_dim = id_dim + 10  # 2 bits for a_self, 2 bits for a_other, 6 bits for reward

        self.death_age = int(history_len * (random.uniform(0.8, 0.95)))
        # shared model reference (no per-agent optimizer)
        self.shared_model = shared_model

        self.agent_id = agent_id
        self.agent_id_str = str(agent_id.int().tolist())
        
        self.history = deque(maxlen=history_len)

        self.local_actions = []
        self.local_rewards = []

        self.local_logps = []
        self.local_values = [] # values from the shared model
        
        self.kv_cache = self.shared_model.build_cache(self.death_age)
        # epsilon for simple exploration in qnet mode (epsilon-greedy)
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

    def observe_and_store(self, other_agent_id, a_self, a_other, r, logp, value):
        # store a CPU-copy of other_agent_id to avoid cross-device tensor issues
        if isinstance(other_agent_id, torch.Tensor):
            other_id_copy = other_agent_id.detach()
        else:
            other_id_copy = torch.tensor(other_agent_id, dtype=torch.float32)
        # append to overall recent history
        self.history.append((other_id_copy, a_self, a_other, float(r)))

        # store local transition (keep CPU copy of state)
        self.local_actions.append(int(a_self))
        self.local_rewards.append(float(r))

        self.local_logps.append(float(logp.detach().cpu().item()))
        self.local_values.append(float(value.detach().cpu().item()))
        return None