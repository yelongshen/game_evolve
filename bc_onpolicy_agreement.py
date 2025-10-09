import random
import copy
import logging
from collections import defaultdict

import torch
from env import PopulationEnv
from agents_simple import EyeForEyeAgent

logger = logging.getLogger(__name__)

def run_agreement_test(steps=2000, N=50, history_len=4, pairs_per_step=10, device='cpu', algorithm='q-bc'):
    env = PopulationEnv(N=N, history_len=history_len, device=device, verbose=False, algorithm=algorithm)

    total = 0
    agree = 0
    agree_by_age = defaultdict(lambda: [0,0])  # age_region -> [agree, total]

    age_bins = [0, history_len // 4, history_len // 2, 3 * history_len // 4, history_len + 1]
    def get_age_region(age):
        for i in range(4):
            if age < age_bins[i+1]:
                return i
        return 3

    for t in range(steps):
        for _ in range(pairs_per_step):
            i, j = random.sample(range(env.N), 2)
            id_i_vec = env.agents[i].agent_id
            id_j_vec = env.agents[j].agent_id

            # model actions
            ai, logpi, vi = env.agents[i].act(id_j_vec)
            aj, logpj, vj = env.agents[j].act(id_i_vec)

            # what would eye_for_eye do for these agents, given the same history and partner id?
            # create temp EyeForEyeAgent clones for local evaluation
            # copy histories to temp agents
            tmp_i = EyeForEyeAgent(id_=0, history_len=env.history_len, id_dim=env.id_dim, agent_id=env.agents[i].agent_id)
            tmp_i.history = copy.deepcopy(env.agents[i].history)
            ti, _, _ = tmp_i.act(id_j_vec)

            tmp_j = EyeForEyeAgent(id_=0, history_len=env.history_len, id_dim=env.id_dim, agent_id=env.agents[j].agent_id)
            tmp_j.history = copy.deepcopy(env.agents[j].history)
            tj, _, _ = tmp_j.act(id_i_vec)

            # record agreement
            total += 2
            if ai == ti:
                agree += 1
                region = get_age_region(len(env.agents[i].history))
                agree_by_age[region][0] += 1
            agree_by_age[get_age_region(len(env.agents[i].history))][1] += 1

            if aj == tj:
                agree += 1
                region = get_age_region(len(env.agents[j].history))
                agree_by_age[region][0] += 1
            agree_by_age[get_age_region(len(env.agents[j].history))][1] += 1

            # compute payoffs and observe
            ri, rj = env.payoff[(ai, aj)]
            env.agents[i].observe_and_store(other_agent_id=id_j_vec, a_self=ai, a_other=aj, r=ri, logp=logpi, value=vi)
            env.agents[j].observe_and_store(other_agent_id=id_i_vec, a_self=aj, a_other=ai, r=rj, logp=logpj, value=vj)

            # death check and buffer dump (duplicate logic from env.step)
            for idx in (i, j):
                agent = env.agents[idx]
                if len(agent.history) >= agent.death_age:
                    env.dump_agent_memory(idx)
                    new_agent = env.create_agents(idx, agent_type=agent.agent_type)
                    env.agents[idx] = new_agent

    overall = agree / total if total > 0 else 0.0
    print(f"Overall agreement (model vs eye_for_eye) on-policy: {overall:.4f} (agree={agree} total={total})")
    for region in range(4):
        a, b = agree_by_age[region]
        ratio = a / b if b > 0 else None
        print(f" region {region}: agree={a} total={b} ratio={ratio}")

    return overall, agree_by_age

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    run_agreement_test(steps=2000, N=50, history_len=64, pairs_per_step=10)
