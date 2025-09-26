



import random
import csv
import time
from collections import defaultdict
import copy
import torch

from buffer import GlobalReplayBuffer
from trainer import Trainer
from agent import Agent
from models import PolicyTransformer
from utils import PAYOFF
import logging
from agents_simple import AlwaysDefectAgent, AlwaysCooperateAgent, RandomAgent, EyeForEyeAgent, AggressiveEyeForEyeAgent

logger = logging.getLogger(__name__)


class PopulationEnv:
    def __init__(self, N=50, history_len=4, device="cpu", lr=1e-2, p_death=0.001, id_dim=16, verbose=False, trainer_update_every=256, trainer_ckpt_dir=None):
        self.N = N
        self.p_death = p_death
        self.device = device
        self.id_dim = id_dim
        self.verbose = verbose
        self.history_len = history_len

        if verbose:
            logger.setLevel(logging.DEBUG)
        logger.info(f"PopulationEnv init: N={N}, history_len={history_len}, p_death={p_death}, device={device}, id_dim={id_dim}")
        # agent id generator: callable(idx) -> 1D array-like or torch tensor of shape (id_dim,)
        
        self.agent_id_generator = lambda idx: __import__('torch').randint(0, 2, (id_dim,), dtype=__import__('torch').float32)
        
        # create shared model
        shared_model = PolicyTransformer(token_dim=id_dim + 10, d_model=64, num_layers=8, mode='qnet', max_len=history_len + 1).to(self.device)
        self.shared_model = shared_model
        # create agents and give each a private deep-copied model so they keep their own weights
        # global replay buffer
        self.global_buffer = GlobalReplayBuffer()
        # trainer (async PPO) working on shared_model
        self.trainer = Trainer(shared_model, self.global_buffer, device=self.device, algorithm='q', ckpt_dir=trainer_ckpt_dir)

        # track per-agent-type reward statistics
        self.type_rewards = defaultdict(float)
        self.type_counts = defaultdict(int)

        self.agents = [self.create_agents(i) for i in range(N)]
        self.time = 0

    def create_agents(self, idx):
        if idx in [0]:
            # create always defect player.
            agent = AlwaysDefectAgent(id_=idx, history_len=self.history_len, id_dim=self.id_dim, agent_id=self.agent_id_generator(idx))
            agent.agent_type = 'always_defect'
        elif idx in [1]:
            # create always cooperate player.
            agent = AlwaysCooperateAgent(id_=idx, history_len=self.history_len, id_dim=self.id_dim, agent_id=self.agent_id_generator(idx))
            agent.agent_type = 'always_cooperate'
        elif idx in [2]:
            # create random player.
            agent = RandomAgent(id_=idx, history_len=self.history_len, id_dim=self.id_dim, agent_id=self.agent_id_generator(idx))
            agent.agent_type = 'random'
        elif idx in [3, 4]:
            # create eye-for-eye player.
            agent = EyeForEyeAgent(id_=idx, history_len=self.history_len, id_dim=self.id_dim, agent_id=self.agent_id_generator(idx))
            agent.agent_type = 'eye_for_eye'
        elif idx in [5, 6]:
            agent = AggressiveEyeForEyeAgent(id_=idx, history_len=self.history_len, id_dim=self.id_dim, agent_id=self.agent_id_generator(idx))
            agent.agent_type = 'aggressive_eye_for_eye'
        else:
            # create model-based player.
            agent = Agent(id_=idx, shared_model=copy.deepcopy(self.trainer.shared_model).to(self.device), history_len=self.history_len, device=self.device, id_dim=self.id_dim, agent_id=self.agent_id_generator(idx))
            agent.agent_type = 'model'
        return agent
    
    def dump_agent_memory(self, idx):
        # dump local buffers of agent idx into global buffer
        agent = self.agents[idx]
        items = []
        for his, a, logp, r, v in zip(agent.history, agent.local_actions, agent.local_logps, agent.local_rewards, agent.local_values):
            (other_id_copy, a_self, a_other, _r) = his
            item = {
                'agent': other_id_copy,
                'a_self': a_self,
                'a_other': a_other,
                'r': _r,
                'action': a,
                'logp': logp,
                'reward': r,
                'value': v
            }
            items.append(item)
        self.global_buffer.add(items)

        logger.debug(f"Dumped {len(items)} transitions from agent {idx} into global buffer (buffer_size={len(self.global_buffer.storage)})")
        # clear agent local buffers after dumping
        agent.local_actions = []
        agent.local_logps = []
        agent.local_rewards = []
        agent.local_values = []
        agent.history.clear()

    def step(self, k=1):
        # sample k pairs and process them (pairs may repeat)
        results = []
        for _ in range(k):
            i, j = random.sample(range(self.N), 2)

            #participants.update([i, j])
            # partner id vectors
            id_i_vec = self.agents[i].agent_id
            id_j_vec = self.agents[j].agent_id
            
            # actions conditioned on partner id (also obtain value estimates)
            ai, logpi, vi = self.agents[i].act(id_j_vec)
            aj, logpj, vj = self.agents[j].act(id_i_vec)
            ri, rj = PAYOFF[(ai, aj)]

            # let agents observe and update (store partner id and partner index too)
            self.agents[i].observe_and_store(other_agent_id=id_j_vec, a_self=ai, a_other=aj, r=ri, logp=logpi, value=vi)
            self.agents[j].observe_and_store(other_agent_id=id_i_vec, a_self=aj, a_other=ai, r=rj, logp=logpj, value=vj)

            results.append((i, j, ai, aj, ri, rj))

            # death check for participants here.
            for idx in (i, j):
                agent = self.agents[idx]
                if len(agent.history) >= agent.death_age:
                    # dump local memory to global buffer
                    if agent.agent_type == 'model':
                        self.dump_agent_memory(idx)
                    old_id = agent.agent_id
                    new_agent = self.create_agents(idx)
                    self.agents[idx] = new_agent
                    if self.verbose:
                        logger.info(f"t={self.time} Agent {idx} died (age={len(agent.history)}) and was replaced (old_id={old_id}) using latest shared_model")

            # track per-agent-type reward statistics
            try:
                self.type_rewards[self.agents[i].agent_type] += ri
                self.type_counts[self.agents[i].agent_type] += 1
            except Exception:
                pass
            try:
                self.type_rewards[self.agents[j].agent_type] += rj
                self.type_counts[self.agents[j].agent_type] += 1
            except Exception:
                pass
        return results


def run_sim(steps=10000, N=50, history_len=4, p_death=1e-3, log_every=500, out_csv=None, pairs_per_step=20, train_every=50, verbose=False, device="cpu"):
    # configure logging when verbose to ensure Trainer.info messages are visible
    if verbose:
        logging.basicConfig(level=logging.INFO)

    env = PopulationEnv(N=N, history_len=history_len, p_death=p_death, verbose=verbose, device=device)
    coop_history = []
    avg_reward_history = []
    global_avg_history = []

    window_coop = 0
    window_rewards = 0.0
    # cumulative / global statistics
    cumulative_rewards = 0.0
    cumulative_actions = 0

    #train_steps = train_every
    for t in range(1, steps + 1):
        # sample pairs_per_step pairs per time-step
        interactions = env.step(k=pairs_per_step)

        # flatten interactions to accumulate stats
        for (ai_idx, aj_idx, ai, aj, ri, rj) in interactions:
            window_coop += (ai == 1) + (aj == 1)
            window_rewards += (ri + rj)
            # update global cumulative stats
            cumulative_rewards += (ri + rj)
            cumulative_actions += 2

        if t % log_every == 0:
            total_actions = log_every * pairs_per_step * 2
            coop_rate = window_coop / total_actions
            avg_reward = window_rewards / total_actions
            coop_history.append((t, coop_rate))
            avg_reward_history.append((t, avg_reward))
            global_avg = cumulative_rewards / cumulative_actions if cumulative_actions > 0 else 0.0

            # summary printout plus buffer/trainer status
            buf_size = len(env.global_buffer.storage)
            trainer_active = getattr(env.trainer, '_last_update_count', 0) > 0
            logger.info(f"t={t:6d} coop_rate={coop_rate:.4f} avg_reward={avg_reward:.4f} global_avg={global_avg:.4f} buffer_size={buf_size} trainer_active={trainer_active}")
            # compute per-agent-type average rewards (cumulative)
            try:
                type_stats = []
                for atype, total in env.type_rewards.items():
                    cnt = env.type_counts.get(atype, 0)
                    avg_t = total / cnt if cnt > 0 else 0.0
                    type_stats.append(f"{atype}={avg_t:.4f}({cnt})")
                type_stats_str = ", ".join(type_stats)
                logger.info("Per-type averages: %s", type_stats_str)
            except Exception:
                logger.exception("Failed to compute per-type averages")
            global_avg_history.append((t, global_avg))
            window_coop = 0
            window_rewards = 0.0

        if len(env.global_buffer.storage) >= train_every:
            #logger.info(f"Trainer before update buffer_size={len(env.global_buffer.storage)}")
            try:
                trained = env.trainer.maybe_update(train_every)
                if trained:
                    logger.info(f"Trainer performed a synchronous update at t={t}; buffer_size={len(env.global_buffer.storage)}")
                    stats = getattr(env.trainer, 'last_update_stats', None)
                    if stats is not None:
                        logger.info("Trainer stats post-update: %s", stats)
            except Exception:
                logger.exception("Trainer update failed")
        # do model trainer update here. 
        
    if out_csv:
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "coop_rate", "avg_reward_window", "global_avg_reward"])
            for (t_c, c), (_, a), (_, g) in zip(coop_history, avg_reward_history, global_avg_history):
                writer.writerow([t_c, c, a, g])

    # stop trainer gracefully
    env.trainer.stop()



    return coop_history, avg_reward_history
