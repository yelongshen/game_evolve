class Community:
    def __init__(self, N, history_len, id_dim, agent_id_generator):
        self.N = N
        self.history_len = history_len
        self.id_dim = id_dim
        self.agent_id_generator = agent_id_generator
        self.agent_types = []
    def get_agent_types(self):
        return self.agent_types

class FairCommunity(Community):
    def __init__(self, N, history_len, id_dim, agent_id_generator):
        super().__init__(N, history_len, id_dim, agent_id_generator)
        n_eye = int(0.10 * N)
        n_agg = int(0.10 * N)
        n_model = N - n_eye - n_agg
        self.agent_types = (
            ['eye_for_eye'] * n_eye +
            ['aggressive_eye_for_eye'] * n_agg +
            ['model'] * n_model
        )

class GreedyCommunity(Community):
    def __init__(self, N, history_len, id_dim, agent_id_generator):
        super().__init__(N, history_len, id_dim, agent_id_generator)
        n_eye = int(0.10 * N)
        n_agg = int(0.10 * N)
        n_def = int(0.10 * N)
        n_model = N - n_eye - n_agg - n_def
        self.agent_types = (
            ['eye_for_eye'] * n_eye +
            ['aggressive_eye_for_eye'] * n_agg +
            ['always_defect'] * n_def +
            ['model'] * n_model
        )

class PureCommunity(Community):
    def __init__(self, N, history_len, id_dim, agent_id_generator):
        super().__init__(N, history_len, id_dim, agent_id_generator)
        n_model = N 
        self.agent_types = (
            ['model'] * n_model
        )

class KindCommunity(Community):
    def __init__(self, N, history_len, id_dim, agent_id_generator):
        super().__init__(N, history_len, id_dim, agent_id_generator)
        n_coop = int(0.10 * N)
        n_eve = int(0.05 * N)
        n_agg = int(0.05 * N)
        n_model = N - n_coop - n_eve - n_agg
        self.agent_types = (
            ['always_cooperate'] * n_coop +
            ['eye_for_eye'] * n_eve +
            ['aggressive_eye_for_eye'] * n_agg +
            ['model'] * n_model
        )

class MixedCommunity(Community):
    def __init__(self, N, history_len, id_dim, agent_id_generator):
        super().__init__(N, history_len, id_dim, agent_id_generator)
        n_coop = int(0.05 * N)
        n_eye = int(0.10 * N)
        n_agg = int(0.10 * N)
        n_def = int(0.05 * N)
        n_rand = int(0.05 * N)
        n_model = N - n_coop - n_eye - n_agg - n_def - n_rand
        self.agent_types = (
            ['always_cooperate'] * n_coop +
            ['eye_for_eye'] * n_eye +
            ['aggressive_eye_for_eye'] * n_agg +
            ['always_defect'] * n_def +
            ['random'] * n_rand +
            ['model'] * n_model
        )


class RuleCommunity(Community):
    def __init__(self, N, history_len, id_dim, agent_id_generator):
        super().__init__(N, history_len, id_dim, agent_id_generator)
        n_coop = int(0.10 * N)
        n_eye = int(0.30 * N)
        n_agg = int(0.25 * N)
        n_def = int(0.10 * N)
        n_rand = int(0.05 * N)
        n_model = N - n_coop - n_eye - n_agg - n_def - n_rand
        self.agent_types = (
            ['always_cooperate'] * n_coop +
            ['eye_for_eye'] * n_eye +
            ['aggressive_eye_for_eye'] * n_agg +
            ['always_defect'] * n_def +
            ['random'] * n_rand +
            ['model'] * n_model
        )

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
from utils import get_payoff
import logging
from agents_simple import AlwaysDefectAgent, AlwaysCooperateAgent, RandomAgent, EyeForEyeAgent, AggressiveEyeForEyeAgent

logger = logging.getLogger(__name__)


class PopulationEnv:
    def __init__(self, N=50, history_len=4, device="cpu", lr=1e-2, p_death=0.001, id_dim=64, verbose=False, trainer_update_every=256, trainer_ckpt_dir=None, payoff_type='co', community_type='fair', algorithm='q', lr_schedule='cosine', lr_schedule_kwargs=None):
        self.N = N
        self.p_death = p_death
        self.device = device
        self.id_dim = id_dim
        self.verbose = verbose
        self.history_len = history_len

        if verbose:
            logger.setLevel(logging.DEBUG)
        logger.info(f"PopulationEnv init: N={N}, history_len={history_len}, p_death={p_death}, device={device}, id_dim={id_dim}, payoff_type={payoff_type}, algorithm={algorithm}")
        # agent id generator: callable(idx) -> 1D array-like or torch tensor of shape (id_dim,)

        self.agent_id_generator = lambda idx: __import__('torch').randint(0, 2, (id_dim,), dtype=__import__('torch').float32)

        # determine model and trainer algorithm modes based on requested algorithm
        self.algorithm = algorithm
        if algorithm == 'qnet-bayesain' or algorithm == 'q-bayes' or algorithm == 'q-bay':
            model_mode = 'qnet-bay'
            trainer_algo = 'q-bay'
        elif algorithm == 'q':
            model_mode = 'qnet'
            trainer_algo = 'q'
        elif algorithm == 'q-agg':
            model_mode = 'qnet'
            trainer_algo = 'q-agg'
        elif algorithm == 'q-bc':
            model_mode = 'qnet'
            trainer_algo = 'q-bc'
        else:
            # default to qnet mode for unknown strings that imply q-learning; otherwise fall back to q
            model_mode = 'vnet'
            trainer_algo = 'gpo'

        # create shared model
        shared_model = PolicyTransformer(token_dim=id_dim + 10, d_model=256, num_layers=6, mode=model_mode, max_len=history_len + 1).to(self.device)
        self.shared_model = shared_model
        # global replay buffer
        self.global_buffer = GlobalReplayBuffer()
        # trainer (synchronous trainer) working on shared_model
        # forward any lr_schedule settings to Trainer if supported
        try:
            self.trainer = Trainer(shared_model, self.global_buffer, device=self.device, algorithm=trainer_algo, ckpt_dir=trainer_ckpt_dir, lr_schedule=lr_schedule, lr_schedule_kwargs=lr_schedule_kwargs)
        except TypeError:
            # fallback for older Trainer signatures
            self.trainer = Trainer(shared_model, self.global_buffer, device=self.device, algorithm=trainer_algo, ckpt_dir=trainer_ckpt_dir)

        # track per-agent-type reward statistics
        self.type_rewards = defaultdict(float)
        self.type_counts = defaultdict(int)

        # Setup community if specified
        self.community_type = community_type
        if community_type == 'fair':
            community = FairCommunity(N, history_len, id_dim, self.agent_id_generator)
        elif community_type == 'greedy':
            community = GreedyCommunity(N, history_len, id_dim, self.agent_id_generator)
        elif community_type == 'pure':
            community = PureCommunity(N, history_len, id_dim, self.agent_id_generator)
        elif community_type == 'kind':
            community = KindCommunity(N, history_len, id_dim, self.agent_id_generator)
        elif community_type == 'mixed':
            community = MixedCommunity(N, history_len, id_dim, self.agent_id_generator)
        elif community_type == 'rule':
            community = RuleCommunity(N, history_len, id_dim, self.agent_id_generator)
        else:
            raise ValueError(f"Unknown community_type: {community_type}")
        agent_types = community.get_agent_types()

        self.agents = [self.create_agents(i, agent_types[i] if agent_types else 'model') for i in range(N)]
        self.time = 0
        self.payoff = get_payoff(payoff_type)
        self.sample_weight = 8 # how many times more likely to sample from buffer than not
        # If running behavioral cloning on eye-for-eye rollouts, avoid heavy duplication
        try:
            if trainer_algo == 'q-bc':
                self.sample_weight = 1
        except Exception:
            pass

    def create_agents(self, idx, agent_type=None):
        atype = agent_type
        if atype == 'always_defect':
            agent = AlwaysDefectAgent(id_=idx, history_len=self.history_len, id_dim=self.id_dim, agent_id=self.agent_id_generator(idx))
        elif atype == 'always_cooperate':
            agent = AlwaysCooperateAgent(id_=idx, history_len=self.history_len, id_dim=self.id_dim, agent_id=self.agent_id_generator(idx))
        elif atype == 'random':
            agent = RandomAgent(id_=idx, history_len=self.history_len, id_dim=self.id_dim, agent_id=self.agent_id_generator(idx))
        elif atype == 'eye_for_eye':
            agent = EyeForEyeAgent(id_=idx, history_len=self.history_len, id_dim=self.id_dim, agent_id=self.agent_id_generator(idx))
        elif atype == 'aggressive_eye_for_eye':
            agent = AggressiveEyeForEyeAgent(id_=idx, history_len=self.history_len, id_dim=self.id_dim, agent_id=self.agent_id_generator(idx))
        else:
            agent = Agent(id_=idx, shared_model=copy.deepcopy(self.trainer.shared_model).to(self.device), history_len=self.history_len, device=self.device, id_dim=self.id_dim, agent_id=self.agent_id_generator(idx))
        agent.agent_type = atype
        return agent
    
    def dump_agent_memory(self, idx):
        # dump local buffers of agent idx into global buffer
        agent = self.agents[idx]
        # If trainer is running in q-bc mode, only store rollouts from eye_for_eye agents
        try:
            trainer_algo = getattr(self.trainer, 'algorithm', None)
        except Exception:
            trainer_algo = None

        if trainer_algo == 'q-bc' and (agent.agent_type != 'eye_for_eye' and agent.agent_type != 'aggressive_eye_for_eye'):
            # clear agent local buffers and skip dumping
            agent.history.clear()
            return
        
        items = []
        for his, logp, v in zip(agent.history, agent.local_logps, agent.local_values):
            (other_id_copy, a_self, a_other, _r) = his
            item = {
                'agent': other_id_copy,
                'a_self': a_self,
                'a_other': a_other,
                'r': _r,
                'logp': logp,
                'value': v
            }
            items.append(item)
        
        for i in range(self.sample_weight):
            self.global_buffer.add(items)

        logger.debug(f"Dumped {len(items)} transitions from agent {idx} into global buffer (buffer_size={len(self.global_buffer.storage)})")
        # clear agent local buffers after dumping
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
            ri, rj = self.payoff[(ai, aj)]

            # let agents observe and update (store partner id and partner index too)
            self.agents[i].observe_and_store(other_agent_id=id_j_vec, a_self=ai, a_other=aj, r=ri, logp=logpi, value=vi)
            self.agents[j].observe_and_store(other_agent_id=id_i_vec, a_self=aj, a_other=ai, r=rj, logp=logpj, value=vj)

            results.append((i, j, ai, aj, ri, rj))

            # death check for participants here.
            for idx in (i, j):
                agent = self.agents[idx]
                if len(agent.history) >= agent.death_age:
                    # dump local memory to global buffer
                    #if agent.agent_type == 'model':
                    self.dump_agent_memory(idx)
                    old_id = agent.agent_id
                    new_agent = self.create_agents(idx, agent_type=agent.agent_type)
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


def run_sim(steps=10000, N=50, history_len=4, p_death=1e-3, log_every=500, out_csv=None, pairs_per_step=20, train_every=50, verbose=False, device="cpu", payoff_type='co', community_type='fair', algorithm='q', lr_schedule='cosine', lr_schedule_kwargs=None):
    # For model agent cooperate ratio by age region
    model_coop_count = [0 for _ in range(4)]
    model_total_count = [0 for _ in range(4)]
    # For eye_for_eye agent cooperate ratio by age region
    eye_coop_count = [0 for _ in range(4)]
    eye_total_count = [0 for _ in range(4)]
    # Divide ages into four regions (quartiles)
    age_bins = [0, history_len // 4, history_len // 2, 3 * history_len // 4, history_len + 1]
    def get_age_region(age):
        for i in range(4):
            if age < age_bins[i+1]:
                return i
        return 3
    # Structure: region_idx -> agent_type -> sum/count
    region_type_reward_sum = defaultdict(lambda: defaultdict(float))
    region_type_count = defaultdict(lambda: defaultdict(int))
    # configure logging when verbose to ensure Trainer.info messages are visible
    if verbose:
        logging.basicConfig(level=logging.INFO)

    # prepare lr_schedule_kwargs defaulting total_steps to 'steps' and warmup fraction to 0.1
    lr_kwargs = dict(lr_schedule_kwargs or {})
    if 'total_steps' not in lr_kwargs:
        lr_kwargs['total_steps'] = 100000
    if 'warmup_frac' not in lr_kwargs:
        lr_kwargs['warmup_frac'] = 0.01

    env = PopulationEnv(N=N, history_len=history_len, p_death=p_death, verbose=verbose, device=device, payoff_type=payoff_type, community_type=community_type, algorithm=algorithm, lr_schedule=lr_schedule, lr_schedule_kwargs=lr_kwargs)
    coop_history = []
    avg_reward_history = []
    global_avg_history = []

    window_coop = 0
    window_rewards = 0.0
    window_type_rewards = defaultdict(float)
    window_type_counts = defaultdict(int)
    # cumulative / global statistics
    cumulative_rewards = 0.0
    cumulative_actions = 0

    #train_steps = train_every
    for t in range(1, steps + 1):
        # sample pairs_per_step pairs per time-step
        interactions = env.step(k=pairs_per_step)

        # flatten interactions to accumulate stats
        for (ai_idx, aj_idx, ai, aj, ri, rj) in interactions:
            # Record model agent cooperate ratio by age region
            for idx, action, agent_type in [(ai_idx, ai, env.agents[ai_idx].agent_type), (aj_idx, aj, env.agents[aj_idx].agent_type)]:
                if agent_type == 'model':
                    agent = env.agents[idx]
                    age = len(agent.history)
                    region = get_age_region(age)
                    if action == 1:
                        model_coop_count[region] += 1
                    model_total_count[region] += 1
                # track eye_for_eye agents cooperation by age region
                if agent_type == 'eye_for_eye':
                    agent = env.agents[idx]
                    age = len(agent.history)
                    region = get_age_region(age)
                    if action == 1:
                        eye_coop_count[region] += 1
                    eye_total_count[region] += 1
            # Record agent type's reward by age region
            for idx, reward, agent_type in [(ai_idx, ri, env.agents[ai_idx].agent_type), (aj_idx, rj, env.agents[aj_idx].agent_type)]:
                agent = env.agents[idx]
                age = len(agent.history)
                region = get_age_region(age)
                region_type_reward_sum[region][agent_type] += reward
                region_type_count[region][agent_type] += 1
            window_coop += (ai == 1) + (aj == 1)
            window_rewards += (ri + rj)
            # update per-type window stats
            ai_type = env.agents[ai_idx].agent_type
            aj_type = env.agents[aj_idx].agent_type
            window_type_rewards[ai_type] += ri
            window_type_counts[ai_type] += 1
            window_type_rewards[aj_type] += rj
            window_type_counts[aj_type] += 1
            # update global cumulative stats
            cumulative_rewards += (ri + rj)
            cumulative_actions += 2

        # Every 1000 steps, log average reward of each agent type for each age region
        if t % 1000 == 0:
            # Log model agent cooperate ratio by age region
            coop_lines = []
            for region in range(4):
                coop = model_coop_count[region]
                total = model_total_count[region]
                ratio = coop * 1.0 / total if total > 0 else None
                coop_lines.append(f"region {region+1}: coop_ratio={ratio:.4f} (n={total})" if ratio is not None else f"region {region+1}: -")
            logger.info(f"Model agent cooperate ratio by age region at step {t}:\n" + "\n".join(coop_lines))
            # reset model agent coop counters
            model_coop_count = [0 for _ in range(4)]
            model_total_count = [0 for _ in range(4)]
            # Log eye_for_eye agent cooperate ratio by age region
            eye_lines = []
            for region in range(4):
                coop = eye_coop_count[region]
                total = eye_total_count[region]
                ratio = coop * 1.0 / total if total > 0 else None
                eye_lines.append(f"region {region+1}: coop_ratio={ratio:.4f} (n={total})" if ratio is not None else f"region {region+1}: -")
            logger.info(f"Eye-for-eye agent cooperate ratio by age region at step {t}:\n" + "\n".join(eye_lines))
            # reset eye_for_eye coop counters
            eye_coop_count = [0 for _ in range(4)]
            eye_total_count = [0 for _ in range(4)]
            # Collect all agent types present in any region
            all_types = set()
            for region in range(4):
                all_types.update(region_type_reward_sum[region].keys())
            log_lines = []
            
            for agent_type in sorted(all_types):
                avg_rewards = []
                for region in range(4):
                    count = region_type_count[region][agent_type]
                    if count > 0:
                        avg = region_type_reward_sum[region][agent_type] / count
                        avg_rewards.append(f"{avg:.4f}")
                    else:
                        avg_rewards.append("-")
                log_lines.append(f"{agent_type}:\n  " + ", ".join(avg_rewards) + ".")
            
            if log_lines:
                logger.info(f"Agent average reward by age region at step {t}:\n" + "\n".join(log_lines))
            # reset region stats after logging
            region_type_reward_sum.clear()
            region_type_count.clear()

        if t % log_every == 0:
        # Every 1000 steps, log model agent's average reward at each age
        
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

            # compute per-agent-type average rewards (window)
            try:
                window_type_stats = []
                for atype, total in window_type_rewards.items():
                    cnt = window_type_counts.get(atype, 0)
                    avg_t = total / cnt if cnt > 0 else 0.0
                    window_type_stats.append(f"{atype}={avg_t:.4f}({cnt})")
                window_type_stats_str = ", ".join(window_type_stats)
                logger.info("Per-type averages (window): %s", window_type_stats_str)
            except Exception:
                logger.exception("Failed to compute per-type averages (window)")

            # compute per-agent-type average rewards (cumulative)
            try:
                type_stats = []
                for atype, total in env.type_rewards.items():
                    cnt = env.type_counts.get(atype, 0)
                    avg_t = total / cnt if cnt > 0 else 0.0
                    type_stats.append(f"{atype}={avg_t:.4f}({cnt})")
                type_stats_str = ", ".join(type_stats)
                logger.info("Per-type averages (cumulative): %s", type_stats_str)
            except Exception:
                logger.exception("Failed to compute per-type averages (cumulative)")

            global_avg_history.append((t, global_avg))
            window_coop = 0
            window_rewards = 0.0
            window_type_rewards.clear()
            window_type_counts.clear()

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
