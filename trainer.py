import time
import os
import copy

import torch
import random
import torch.optim as optim
import numpy as np
from utils import action_to_bits, reward_to_6bits
import torch.nn.functional as F
import logging
import collections
import torch.nn as nn

logger = logging.getLogger(__name__)
class Trainer:
    def __init__(self, model, buffer, device='cpu', ppo_epochs=1, clip_eps=0.8,
                 value_coef=0.01, value_clip=0.2, lam=0.95, entropy_coef=0.01, gamma=0.95, lr=3e-4, min_buffer_size=32, normalize_returns=True, ckpt_dir=None, algorithm='ppo'):
        # synchronous trainer that trains on a deep copy of the shared model
        self.shared_model = model
        self.buffer = buffer
        self.device = device
        # deep copy the shared model to perform training privately
        self.train_model = copy.deepcopy(model).to(self.device)
        self.ppo_epochs = ppo_epochs
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        # PPO value clipping parameter (prevents large value updates)
        self.value_clip = value_clip
        # GAE lambda parameter
        self.lam = lam
        # whether to normalize returns (and normalize model value preds accordingly)
        self.normalize_returns = normalize_returns
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        # temperature for soft-aggregation over horizons (q-agg)
        self.tau_agg = 0.5
        self.min_buffer_size = min_buffer_size
        # algorithm: 'ppo' (default) or 'gpo' (vanilla policy-gradient / actor-critic)
        # GPO here is implemented as a simple on-policy actor-critic update that
        # uses precomputed advantages (GAE) and a value-function baseline.
        # It does not use clipping or the PPO surrogate.
        self.algorithm = algorithm

        if self.shared_model.mode == 'qnet':
            # allow either plain q update or the aggregation variant
            assert self.algorithm in ('q', 'q-agg')
        if self.shared_model.mode == 'qnet-bay':
            assert self.algorithm == 'q-bay'

        self.opt = optim.Adam(self.train_model.parameters(), lr=lr)
        # track how many samples we've processed from the buffer (approx)
        # self._last_update_count = 0
        # checkpoint storage (list of dicts) and optional on-disk directory
        self.checkpoints = []
        self.ckpt_dir = ckpt_dir
        self.update_per_sync = 10  # how many updates per sync from shared model
        self.steps_since_sync = 0
        if ckpt_dir is not None:
            os.makedirs(ckpt_dir, exist_ok=True)

    def _save_checkpoint(self):
        # Save a deep copy of state_dict (moved to CPU) so it won't be modified in-place
        state = {k: v.detach().cpu().clone() for k, v in self.train_model.state_dict().items()}
        self.checkpoints.append(state)
        # optional: write to disk with a timestamped filename
        if self.ckpt_dir is not None:
            fname = os.path.join(self.ckpt_dir, f"ckpt_{len(self.checkpoints):04d}.pt")
            torch.save(state, fname)
        return len(self.checkpoints) - 1

    def _sync_to_shared_model(self):
        # copy train_model weights into the shared model (in-place)
        self.shared_model.load_state_dict(self.train_model.state_dict())

    def _prepare_batch(self, batch):
        """Convert sampled batch (list of seq_items lists) into model inputs
        and precompute GAE advantages + returns. Returns a dict with prepared
        tensors and lists used by the specific update algorithms.
        """
        token_list = []
        actions_seqs = []
        old_logps_seqs = []
        values_seqs = []
        returns_seqs = []

        max_len = getattr(self.train_model, 'pos_emb').size(0)

        for seq_items in batch:
            tokens = []
            last_a_self = last_a_other = None
            last_r = 0.0
            for i, item in enumerate(seq_items):
                agent_id = item['agent'].to(self.device)
                if i == 0:
                    token = torch.cat([agent_id, torch.zeros(10, device=self.device)])
                else:
                    token = torch.cat([
                        agent_id,
                        action_to_bits(last_a_self, self.device),
                        action_to_bits(last_a_other, self.device),
                        reward_to_6bits(last_r, self.device)
                    ])
                tokens.append(token)
                last_a_self, last_a_other, last_r = item['a_self'], item['a_other'], item['r']

            seq = torch.stack(tokens)  # (seq_len, token_dim)
            seq_len = seq.size(0)
            if seq_len > max_len:
                seq = seq[-max_len:]
                seq_len = seq.size(0)

            pad_len = max_len - seq_len
            if pad_len > 0:
                seq = F.pad(seq, (0, 0, 0, pad_len))
            
            if len(seq_items) > max_len:
                seq_items_trunc = seq_items[-max_len:]
            else:
                seq_items_trunc = seq_items

            actions_seq = [int(it.get('a_self', 0)) for it in seq_items_trunc]
            old_logps_seq = [float(it.get('logp', 0.0)) for it in seq_items_trunc]
            values_seq = [float(it.get('value', 0.0)) for it in seq_items_trunc]
            returns_seq = [float(it.get('r', 0.0)) for it in seq_items_trunc]

            token_list.append(seq.unsqueeze(0))
            actions_seqs.append(actions_seq)
            old_logps_seqs.append(old_logps_seq)
            values_seqs.append(values_seq)
            returns_seqs.append(returns_seq)

        seq_list = [s.squeeze(0).to(self.device) for s in token_list]
        if len(seq_list) == 0:
            return None
        batch_tokens = torch.cat([s.unsqueeze(0) for s in seq_list], dim=0)

        prepared = {
            'batch_tokens': batch_tokens,
            'actions_seqs': actions_seqs,
            'old_logps_seqs': old_logps_seqs,
            'values_seqs': values_seqs,
            'returns_seqs': returns_seqs,
        }

        return prepared
        
    def _prepare_batch_with_gae(self, prepared):
        
        batch_tokens = prepared['batch_tokens']
        actions_seqs = prepared['actions_seqs']
        old_logps_seqs = prepared['old_logps_seqs']
        values_seqs = prepared['values_seqs']
        returns_seqs = prepared['returns_seqs']

        with torch.no_grad():
            _, values_pre = self.train_model.forward(batch_tokens)

        # compute GAE and returns per sequence
        advantages_flat_pre = []
        returns_flat_pre = []
        old_values_flat_pre = []
        for i, actions in enumerate(actions_seqs):
            seq_len = len(actions)
            if seq_len == 0:
                continue
            rewards = torch.tensor(returns_seqs[i], dtype=torch.float32, device=self.device)
            vals = values_pre[i, :seq_len].to(self.device)

            if seq_len < values_pre.size(1):
                next_value = values_pre[i, seq_len].to(self.device)
            else:
                next_value = torch.tensor(0.0, device=self.device)

            gae = 0.0
            advs = torch.zeros(seq_len, device=self.device)
            rets = torch.zeros(seq_len, device=self.device)
            next_v = next_value
            for t in reversed(range(seq_len)):
                delta = rewards[t] + self.gamma * next_v - vals[t]
                gae = delta + self.gamma * self.lam * gae
                advs[t] = gae
                rets[t] = gae + vals[t]
                next_v = vals[t]

            advantages_flat_pre.append(advs)
            returns_flat_pre.append(rets)
            old_values_flat_pre.append(vals)

        if len(advantages_flat_pre) == 0:
            return None

        advantages_pre = torch.cat(advantages_flat_pre, dim=0).view(-1)
        returns_pre = torch.cat(returns_flat_pre, dim=0).view(-1)
        old_values_pre = torch.cat(old_values_flat_pre, dim=0).view(-1)

        if advantages_pre.std() > 1e-8:
            advantages_pre = (advantages_pre - advantages_pre.mean()) / (advantages_pre.std() + 1e-8)

        if self.normalize_returns:
            self.returns_mean = returns_pre.mean()
            self.returns_std = returns_pre.std()
            if self.returns_std < 1e-8:
                self.returns_std = 1.0
            returns_pre = (returns_pre - self.returns_mean) / (self.returns_std + 1e-8)

        prepared = {
            'batch_tokens': batch_tokens,
            'actions_seqs': actions_seqs,
            'old_logps_seqs': old_logps_seqs,
            'values_seqs': values_seqs,
            'returns_seqs': returns_seqs,
            'advantages_pre': advantages_pre,
            'returns_pre': returns_pre,
            'old_values_pre': old_values_pre,
        }
        return prepared

    def _run_gpo(self, prepared):
        """Run a single GPO (vanilla actor-critic) update using prepared batch."""
        batch_tokens = prepared['batch_tokens']
        actions_seqs = prepared['actions_seqs']
        advantages_pre = prepared['advantages_pre']
        returns_pre = prepared['returns_pre']

        probs, values = self.train_model.forward(batch_tokens)

        new_logps_flat = []
        new_values_flat = []
        entropy_flat = []
        for i, actions in enumerate(actions_seqs):
            seq_len = len(actions)
            if seq_len == 0:
                continue
            seq_probs = probs[i, :seq_len]
            seq_values = values[i, :seq_len]

            acts = torch.tensor(actions, dtype=torch.float32, device=self.device)
            m = torch.distributions.Bernoulli(probs=seq_probs)
            per_step_logps = m.log_prob(acts)
            per_step_entropy = m.entropy()

            new_logps_flat.append(per_step_logps)
            new_values_flat.append(seq_values)
            entropy_flat.append(per_step_entropy)

        if len(new_logps_flat) == 0:
            return False

        new_logps = torch.cat(new_logps_flat, dim=0).view(-1)
        new_values = torch.cat(new_values_flat, dim=0).view(-1)
        entropies = torch.cat(entropy_flat, dim=0).view(-1)

        advantages = advantages_pre.to(new_logps.device)
        b_returns = returns_pre.to(new_logps.device)

        min_len = min(new_logps.numel(), advantages.numel())
        if new_logps.numel() != min_len:
            new_logps = new_logps[:min_len]
            new_values = new_values[:min_len]
            entropies = entropies[:min_len]
            advantages = advantages[:min_len]
            b_returns = b_returns[:min_len]

        policy_loss = -(advantages * new_logps).mean()

        if self.normalize_returns:
            mu = getattr(self, 'returns_mean')
            sigma = getattr(self, 'returns_std')
            new_values_norm = (new_values - mu.to(new_values.device)) / (sigma + 1e-8)
            b_returns_norm = b_returns
        else:
            new_values_norm = new_values
            b_returns_norm = b_returns

        value_loss = ((new_values_norm - b_returns_norm) ** 2).mean()
        entropy = entropies.mean()

        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        self.last_update_stats = {
            'policy_loss': float(policy_loss.item()),
            'value_loss': float(value_loss.item()),
            'entropy': float(entropy.item()),
            'loss': float(loss.item()),
            'clipped_frac': 0.0,
            'approx_kl': 0.0,
        }

        logger.info(
            "GPO update: policy_loss=%.6f value_loss=%.6f entropy=%.6f loss=%.6f",
            self.last_update_stats['policy_loss'],
            self.last_update_stats['value_loss'],
            self.last_update_stats['entropy'],
            self.last_update_stats['loss'],
        )

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return True

    def _run_q_bay(self, prepared):
        batch_tokens = prepared['batch_tokens']
        actions_seqs = prepared['actions_seqs']
        old_logps_seqs = prepared['old_logps_seqs']
        values_seqs = prepared['values_seqs']
        returns_seqs = prepared['returns_seqs']
        # train_model.forward now returns (mu, logvar) per action
        mu_pred, logvar_pred = self.train_model.forward(batch_tokens)

        
        # compute next-state Q estimates using the shared_model (as a target source)
        with torch.no_grad():
            mu_target, logvar_target = self.shared_model.forward(batch_tokens)
            # mu_target: (batch, seq_len, n_actions)
            # For each time step, next-state value is max_a mu(next_state, a)
            q_next_values = mu_target.max(dim=2).values

        batch_size = batch_tokens.size(0)

        total_loss = 0.0
        count = 0
        for i, actions in enumerate(actions_seqs):
            returns = returns_seqs[i]
            seq_len = len(actions)
            if seq_len == 0:
                continue
            acts = torch.tensor(actions, dtype=torch.int64, device=self.device)
            r = torch.tensor(returns, dtype=torch.float32, device=self.device)

            # Build target y: y_t = r_t + gamma * max_a Q_{target}(s_{t+1}, a)
            # q_next_values[i] has shape (seq_len,)
            shifted_q = torch.zeros(seq_len, device=self.device)
            if seq_len > 1:
                shifted_q[:-1] = q_next_values[i, 1:seq_len]
            # last next value stays zero
            y = r + self.gamma * shifted_q

            # gather predicted mu/logvar for actions taken
            # mu_pred: (batch, seq_len, n_actions)
            mu_sa = torch.gather(mu_pred[i, :seq_len], 1, acts.unsqueeze(-1)).squeeze(-1)
            logvar_sa = torch.gather(logvar_pred[i, :seq_len], 1, acts.unsqueeze(-1)).squeeze(-1)

            # Gaussian negative log-likelihood loss: 0.5 * (logvar + (y - mu)^2 / exp(logvar)) + const
            # We'll use mean over seq steps
            var = torch.exp(logvar_sa)
            nll = 0.5 * (logvar_sa + ((y - mu_sa) ** 2) / (var + 1e-8))
            loss_i = nll.mean()
            total_loss = total_loss + loss_i
            count += 1

        if count == 0:
            return False

        loss = total_loss / float(count)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        # clip grads on train_model parameters
        nn.utils.clip_grad_norm_(self.train_model.parameters(), 1.0)
        self.opt.step()

        self.last_update_stats = {'q_loss': float(loss.item())}

        logger.info("Q update: q_loss=%.6f ", self.last_update_stats['q_loss'])
        return True
    
    def _run_q(self, prepared):
        batch_tokens = prepared['batch_tokens']
        actions_seqs = prepared['actions_seqs']
        old_logps_seqs = prepared['old_logps_seqs']
        values_seqs = prepared['values_seqs']
        returns_seqs = prepared['returns_seqs']
        # train_model.forward returns q-values of shape (batch, seq_len, n_actions)
        qvalues = self.train_model.forward(batch_tokens)

        # compute next-state Q estimates using the shared_model (as a target source)
        with torch.no_grad():
            q_target_buff = self.shared_model.forward(batch_tokens)
            # q_target_buff: (batch, seq_len, n_actions)
            # For each time step, next-state value is max_a Q(next_state, a)
            q_next_values = q_target_buff.max(dim=2).values

        batch_size = batch_tokens.size(0)

        total_loss = 0.0
        count = 0
        for i, actions in enumerate(actions_seqs):
            returns = returns_seqs[i]
            seq_len = len(actions)
            if seq_len == 0:
                continue
            acts = torch.tensor(actions, dtype=torch.int64, device=self.device)
            r = torch.tensor(returns, dtype=torch.float32, device=self.device)

            # Build target y: y_t = r_t + gamma * max_a Q_{target}(s_{t+1}, a)
            # q_next_values[i] has shape (seq_len,)
            shifted_q = torch.zeros(seq_len, device=self.device)
            if seq_len > 1:
                shifted_q[:-1] = q_next_values[i, 1:seq_len]
            # last next value stays zero
            y = r + self.gamma * shifted_q

            # gather q(s,a) for actions taken
            q_sa = torch.gather(qvalues[i, :seq_len], 1, acts.unsqueeze(-1)).squeeze(-1)

            loss_i = F.smooth_l1_loss(q_sa, y)
            total_loss = total_loss + loss_i
            count += 1

        if count == 0:
            return False

        loss = total_loss / float(count)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        # clip grads on train_model parameters
        nn.utils.clip_grad_norm_(self.train_model.parameters(), 1.0)
        self.opt.step()

        self.last_update_stats = {'q_loss': float(loss.item())}

        logger.info("Q update: q_loss=%.6f ", self.last_update_stats['q_loss'])
        return True
    
    def _run_q_agg(self, prepared):
        """
        Q-learning with soft aggregation over multi-step horizons per time-step.
        For each time t in a sequence we build a set of n-step returns G^{(n)}_t
        for n=1..N (until episode end or seq end) and aggregate with:
            G_t(\tau) = \tau * log( (1/N) * sum_n exp(G^{(n)}_t / \tau) )
        where tau = self.tau_agg. This reduces bias from picking a single horizon.
        """
        batch_tokens = prepared['batch_tokens']
        actions_seqs = prepared['actions_seqs']
        returns_seqs = prepared['returns_seqs']

        # train_model.forward returns q-values of shape (batch, seq_len, n_actions)
        qvalues = self.train_model.forward(batch_tokens)

        # compute next-state Q estimates using the shared_model (as a target source)
        with torch.no_grad():
            q_target_buff = self.shared_model.forward(batch_tokens)
            q_next_values = q_target_buff.max(dim=2).values

        total_loss = 0.0
        count = 0
        tau = float(self.tau_agg)
        for i, actions in enumerate(actions_seqs):
            seq_len = len(actions)
            if seq_len == 0:
                continue
            acts = torch.tensor(actions, dtype=torch.int64, device=self.device)
            r = torch.tensor(returns_seqs[i], dtype=torch.float32, device=self.device)

            # Precompute prefix sums of discounts for efficient n-step returns
            # We'll compute all n-step returns G^{(n)}_t = sum_{k=0}^{n-1} gamma^k r_{t+k} + gamma^n q_{t+n}
            # For simplicity (small seq_len) we'll compute straightforwardly O(T^2)
            G_targets = torch.zeros((seq_len, seq_len), device=self.device)  # G_targets[t, n-1]
            for t in range(seq_len):
                # accumulate rewards
                for n in range(1, seq_len - t + 1):
                    # sum rewards r_t ... r_{t+n-1} discounted
                    discounts = torch.tensor([self.gamma ** k for k in range(n)], device=self.device)
                    rews = r[t:t+n]
                    discounted_sum = (discounts * rews).sum()
                    # bootstrap value at t+n (if within seq) else 0
                    if t + n < seq_len:
                        q_boot = q_next_values[i, t + n]
                    else:
                        q_boot = torch.tensor(0.0, device=self.device)
                    Gn = discounted_sum + (self.gamma ** n) * q_boot
                    G_targets[t, n-1] = Gn

            # For each t, aggregate across horizons with softmax (log-sum-exp)
            # shape: (seq_len,)
            G_agg = torch.zeros(seq_len, device=self.device)
            for t in range(seq_len):
                Gn = G_targets[t, :seq_len - t]  # available horizons
                if Gn.numel() == 1 or tau <= 1e-8:
                    G_agg[t] = Gn[0]
                else:
                    # compute tau * (logsumexp(Gn / tau) - log(N))
                    # where N = Gn.numel()
                    N = float(Gn.numel())
                    lse = torch.logsumexp(Gn / tau, dim=0)
                    G_agg[t] = tau * (lse - torch.log(torch.tensor(N, device=self.device)))

            # gather q(s,a) for actions taken
            q_sa = torch.gather(qvalues[i, :seq_len], 1, acts.unsqueeze(-1)).squeeze(-1)

            loss_i = F.smooth_l1_loss(q_sa, G_agg)
            total_loss = total_loss + loss_i
            count += 1

        if count == 0:
            return False

        loss = total_loss / float(count)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.train_model.parameters(), 1.0)
        self.opt.step()

        self.last_update_stats = {'q_agg_loss': float(loss.item())}
        logger.info("Q-agg update: q_agg_loss=%.6f ", self.last_update_stats['q_agg_loss'])
        return True
      
      
    def _run_ppo(self, prepared):
        """Run PPO updates using prepared batch dict."""
        batch_tokens = prepared['batch_tokens']
        actions_seqs = prepared['actions_seqs']
        old_logps_seqs = prepared['old_logps_seqs']
        values_seqs = prepared['values_seqs']
        returns_seqs = prepared['returns_seqs']
        advantages_pre = prepared['advantages_pre']
        returns_pre = prepared['returns_pre']
        old_values_pre = prepared['old_values_pre']

        for _ in range(self.ppo_epochs):
            probs, values = self.train_model.forward(batch_tokens)

            new_logps_flat = []
            new_values_flat = []
            old_logps_flat = []
            old_values_flat = []
            returns_flat = []
            entropy_flat = []

            for i, actions in enumerate(actions_seqs):
                seq_len = len(actions)
                if seq_len == 0:
                    continue
                seq_probs = probs[i, :seq_len]
                seq_values = values[i, :seq_len]

                acts = torch.tensor(actions, dtype=torch.float32, device=self.device)
                old_logps = torch.tensor(old_logps_seqs[i], dtype=torch.float32, device=self.device)
                old_vals = torch.tensor(values_seqs[i], dtype=torch.float32, device=self.device)
                rets = torch.tensor(returns_seqs[i], dtype=torch.float32, device=self.device)

                m = torch.distributions.Bernoulli(probs=seq_probs)
                per_step_logps = m.log_prob(acts)
                per_step_entropy = m.entropy()

                new_logps_flat.append(per_step_logps)
                new_values_flat.append(seq_values)
                old_logps_flat.append(old_logps)
                old_values_flat.append(old_vals)
                returns_flat.append(rets)
                entropy_flat.append(per_step_entropy)

            if len(new_logps_flat) == 0:
                continue

            new_logps = torch.cat(new_logps_flat, dim=0).view(-1)
            new_values = torch.cat(new_values_flat, dim=0).view(-1)
            b_old_logps = torch.cat(old_logps_flat, dim=0).view(-1)
            b_old_values = torch.cat(old_values_flat, dim=0).view(-1)
            b_returns = torch.cat(returns_flat, dim=0).view(-1)
            entropies = torch.cat(entropy_flat, dim=0).view(-1)

            advantages = advantages_pre.to(new_logps.device)
            b_returns = returns_pre.to(new_logps.device)
            b_old_values = old_values_pre.to(new_logps.device)

            min_len = min(new_logps.numel(), advantages.numel(), b_old_logps.numel(), b_returns.numel())
            if new_logps.numel() != min_len:
                new_logps = new_logps[:min_len]
                new_values = new_values[:min_len]
                b_old_logps = b_old_logps[:min_len]
                b_old_values = b_old_values[:min_len]
                b_returns = b_returns[:min_len]
                advantages = advantages[:min_len]
                entropies = entropies[:min_len]

            ratio = torch.exp(new_logps - b_old_logps)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            if self.normalize_returns:
                mu = getattr(self, 'returns_mean')
                sigma = getattr(self, 'returns_std')
                new_values_norm = (new_values - mu.to(new_values.device)) / (sigma + 1e-8)
                b_old_values_norm = (b_old_values - mu.to(b_old_values.device)) / (sigma + 1e-8)
                b_returns_norm = b_returns
            else:
                new_values_norm = new_values
                b_old_values_norm = b_old_values
                b_returns_norm = b_returns

            if hasattr(self, 'value_clip') and self.value_clip is not None and self.value_clip > 0.0:
                value_pred_clipped = b_old_values_norm + torch.clamp(new_values_norm - b_old_values_norm, -self.value_clip, self.value_clip)
                value_loss_unclipped = (new_values_norm - b_returns_norm) ** 2
                value_loss_clipped = (value_pred_clipped - b_returns_norm) ** 2
                value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()
            else:
                value_loss = ((new_values_norm - b_returns_norm) ** 2).mean()

            entropy = entropies.mean()

            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            with torch.no_grad():
                # Mask of samples where ratio moved outside the clipping range
                clamped_mask_total = (ratio > 1.0 + self.clip_eps) | (ratio < 1.0 - self.clip_eps)
                clipped_mask = (clamped_mask_total & ((advantages > 0) | (advantages < 0))) & (((ratio > 1.0 + self.clip_eps) & (advantages > 0)) | ((ratio < 1.0 - self.clip_eps) & (advantages < 0)))
                # fraction that were clamped at all
                clipped_frac_total = float(clamped_mask_total.float().mean().item())
                # fraction that were clamped in the direction that would have improved surrogate (traditional diagnostic)
                clipped_frac = float(clipped_mask.float().mean().item())
                clipped_count = int(clamped_mask_total.long().sum().item())
                approx_kl = float((b_old_logps - new_logps).mean().item())

                # extra diagnostics to explain tiny policy loss
                adv_mean = float(advantages.mean().item())
                adv_std = float(advantages.std().item()) if advantages.numel() > 1 else 0.0
                try:
                    adv_p10 = float(torch.quantile(advantages, 0.1).item())
                    adv_p50 = float(torch.quantile(advantages, 0.5).item())
                    adv_p90 = float(torch.quantile(advantages, 0.9).item())
                except Exception:
                    # torch.quantile may not be available in very old torch versions
                    adv_p10 = adv_p50 = adv_p90 = 0.0

                ratio_mean = float(ratio.mean().item())
                ratio_std = float(ratio.std().item()) if ratio.numel() > 1 else 0.0
                try:
                    ratio_p10 = float(torch.quantile(ratio, 0.1).item())
                    ratio_p50 = float(torch.quantile(ratio, 0.5).item())
                    ratio_p90 = float(torch.quantile(ratio, 0.9).item())
                except Exception:
                    ratio_p10 = ratio_p50 = ratio_p90 = 1.0

                probs_mean = float(torch.sigmoid(new_logps).mean().item()) if new_logps.numel() > 0 else 0.0
                probs_std = float(torch.sigmoid(new_logps).std().item()) if new_logps.numel() > 1 else 0.0
                old_logp_mean = float(b_old_logps.mean().item()) if b_old_logps.numel() > 0 else 0.0

            self.last_update_stats = {
                'policy_loss': float(policy_loss.item()),
                'value_loss': float(value_loss.item()),
                'entropy': float(entropy.item()),
                'loss': float(loss.item()),
                'clipped_frac': clipped_frac,
                'clipped_frac_total': clipped_frac_total,
                'clipped_count': clipped_count,
                'approx_kl': approx_kl,
                'adv_mean': adv_mean,
                'adv_std': adv_std,
                'adv_p10': adv_p10,
                'adv_p50': adv_p50,
                'adv_p90': adv_p90,
                'ratio_mean': ratio_mean,
                'ratio_std': ratio_std,
                'ratio_p10': ratio_p10,
                'ratio_p50': ratio_p50,
                'ratio_p90': ratio_p90,
                'probs_mean': probs_mean,
                'probs_std': probs_std,
                'old_logp_mean': old_logp_mean,
            }

            logger.info(
                "PPO update: policy_loss=%.6f value_loss=%.6f entropy=%.6f loss=%.6f clipped_frac=%.3f clipped_frac_total=%.3f clipped_count=%d approx_kl=%.6f adv_mean=%.6f adv_std=%.6f ratio_mean=%.6f ratio_std=%.6f probs_mean=%.6f",
                self.last_update_stats['policy_loss'],
                self.last_update_stats['value_loss'],
                self.last_update_stats['entropy'],
                self.last_update_stats['loss'],
                self.last_update_stats['clipped_frac'],
                self.last_update_stats['clipped_frac_total'],
                self.last_update_stats['clipped_count'],
                self.last_update_stats['approx_kl'],
                self.last_update_stats['adv_mean'],
                self.last_update_stats['adv_std'],
                self.last_update_stats['ratio_mean'],
                self.last_update_stats['ratio_std'],
                self.last_update_stats['probs_mean'],
            )

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

        return True
    
    def maybe_update(self, train_sample_size=32, global_step=0):
        """Top-level update entry: prepare a batch and dispatch to the configured algorithm."""
        n = len(self.buffer.storage)
        if n < self.min_buffer_size:
            return False
        if n < train_sample_size:
            return False

        batch = self.buffer.pop_sample(train_sample_size)
        if not batch:
            return False

        prepared = self._prepare_batch(batch)
        if prepared is None:
            return False

        if self.algorithm == 'q':
            updated = self._run_q(prepared)
        elif self.algorithm == 'q-agg':
            updated = self._run_q_agg(prepared)
        if self.algorithm == 'q-bay':
            updated = self._run_q_bay(prepared)
        elif self.algorithm == 'gpo':
            prepared = self._prepare_batch_with_gae(prepared)
            updated = self._run_gpo(prepared)
        elif self.algorithm == 'ppo':
            prepared = self._prepare_batch_with_gae(prepared)
            updated = self._run_ppo(prepared)

        if not updated:
            return False

        # save checkpoint and sync weights back into the shared model
        ckpt_idx = self._save_checkpoint()

        self.steps_since_sync += 1
        if self.steps_since_sync >= self.update_per_sync:
            self._sync_to_shared_model()
            self.steps_since_sync = 0

        return True
    
    def stop(self):
        # no-op kept for compatibility with earlier API
        return

    def latest_checkpoint(self):
        if not self.checkpoints:
            return None
        return self.checkpoints[-1]

    def checkpoint_count(self):
        return len(self.checkpoints)
