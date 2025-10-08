import torch
from env import PopulationEnv


def inspect(batch_size=4, steps=2000, device='cpu'):
    env = PopulationEnv(N=20, history_len=512, device=device, verbose=False, algorithm='q-bc')
    # run interactions until buffer has something
    pairs_per_step = 10
    for t in range(steps):
        env.step(k=pairs_per_step)
        if len(env.global_buffer.storage) >= batch_size:
            break
    print(f'buffer_size={len(env.global_buffer.storage)} after {t+1} iterations')

    batch = env.global_buffer.pop_sample(batch_size)
    if not batch:
        print('no batch available')
        return

    # print one example sequence from raw batch and prepared tokens
    from trainer import Trainer
    trainer = env.trainer
    prepared = trainer._prepare_batch(batch)

    id_dim = env.id_dim
    print('id_dim', id_dim)

    for i, seq_items in enumerate(batch):
        print('='*40)
        print('Sequence', i)
        print('raw seq len', len(seq_items))
        # collect agent ids as CPU tensors
        agent_ids = [it['agent'] for it in seq_items]
        # compare each timestep agent id to all previous agent ids
        for t_idx, aid in enumerate(agent_ids):
            aid_cpu = aid.detach().cpu()
            matched = False
            for prev in agent_ids[:t_idx]:
                try:
                    if isinstance(prev, torch.Tensor):
                        if prev.detach().cpu().shape == aid_cpu.shape and torch.equal(prev.detach().cpu(), aid_cpu):
                            matched = True
                            break
                except Exception:
                    continue
            print(f' t={t_idx} matched_prev={matched} id_sample={aid_cpu[:8].tolist()}...')

        # Compare prepared tokens' first id_dim elements to stored agent ids (per timestep)
        tokens = prepared['batch_tokens'][i]  # shape (max_len, token_dim)
        print('Prepared tokens shape', tokens.shape)
        for t_idx in range(tokens.size(0)):
            token_id_slice = tokens[t_idx, :id_dim].cpu()
            # if this position corresponds to a stored seq item
            if t_idx < len(agent_ids):
                aid_cpu = agent_ids[t_idx].detach().cpu()
                equal = False
                try:
                    if token_id_slice.shape == aid_cpu.shape and torch.allclose(token_id_slice, aid_cpu, atol=1e-6):
                        equal = True
                except Exception:
                    equal = False
                print(f' token t={t_idx} matches_agent={equal} token_id_head={token_id_slice[:8].tolist()} agent_id_head={aid_cpu[:8].tolist()}')
            else:
                print(f' token t={t_idx} (padded) token_id_head={token_id_slice[:8].tolist()}')


if __name__ == '__main__':
    inspect()
