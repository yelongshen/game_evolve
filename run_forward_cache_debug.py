import torch
import numpy as np
from models import PolicyTransformer


def full_forward_per_layer(model, seq):
	# seq: (1, seq_len, token_dim)
	batch, seq_len, token_dim = seq.shape
	# replicate _encode but capture per-layer outputs
	x = model.input_proj(seq)  # (batch, seq_len, d_model)
	try:
		id_emb = model.id_mlp(seq[:, :seq_len, :model.id_dim])
		x = x + id_emb
	except Exception:
		pass
	x = x + model.pos_emb[:seq_len].unsqueeze(0)
	out = x.permute(1, 0, 2)  # (seq_len, batch, d_model)

	per_layer = []
	per_layer_debug = []
	per_layer.append(out.clone())
	per_layer_debug.append(None)

	src_mask = torch.tril(torch.ones((seq_len, seq_len), device=out.device, dtype=out.dtype))
	for layer in model.layers:
		res = layer.forward(out, src_mask=src_mask, debug=True)
		if isinstance(res, tuple):
			out, ldbg = res
			per_layer_debug.append(ldbg)
		else:
			out = res
			per_layer_debug.append(None)
		# out is (seq_len, batch, d_model)
		per_layer.append(out.clone())
	return per_layer, per_layer_debug


def incr_forward_per_layer(model, seq):
	# seq: (1, seq_len, token_dim)
	batch, seq_len, token_dim = seq.shape
	memories = model.build_cache(prealloc_len=seq_len)

	# We'll store per-layer outputs for each time step as list of lists: per_timestep[layer_idx] = out (1,batch,d_model) last token
	per_timestep = [ [] for _ in range(seq_len) ]
	per_timestep_debug = [ [] for _ in range(seq_len) ]

	for t in range(seq_len):
		new_tokens_2d = seq[0, t:t+1, :]
		start_pos = t
		# build new_x same as forward_with_cache
		try:
			id_slice = new_tokens_2d[:, :model.id_dim]
			id_emb = model.id_mlp(id_slice)
			new_x = model.input_proj(new_tokens_2d) + id_emb + model.pos_emb[start_pos:start_pos + 1]
		except Exception:
			new_x = model.input_proj(new_tokens_2d) + model.pos_emb[start_pos:start_pos + 1]
		new_x = new_x.unsqueeze(1)  # (new_len=1, batch=1, d_model)

		out = new_x
		per_timestep[t].append(out.clone())
		per_timestep_debug[t].append(None)

		# run through each layer's incremental forward and capture out
		for l_idx, layer in enumerate(model.layers):
			res = layer.forward_incremental(memories[l_idx], out, position_idx=start_pos, debug=True)
			if isinstance(res, tuple):
				out, ldbg = res
				per_timestep_debug[t].append(ldbg)
			else:
				out = res
				per_timestep_debug[t].append(None)
			# record the current output (seq_len=1, batch=1, d_model) for this timestep/layer
			per_timestep[t].append(out.clone())

	# per_timestep is list length seq_len, each element is list length num_layers+1 of tensors (1,1,d_model)
	return per_timestep, per_timestep_debug


def find_first_mismatch(model, seq, atol=1e-6):
	full_layers, full_layer_debug = full_forward_per_layer(model, seq)
	incr, incr_debug = incr_forward_per_layer(model, seq)

	seq_len = seq.shape[1]
	num_layers = len(full_layers)

	# full_layers[l] is (seq_len, batch, d_model)
	for t in range(seq_len):
		for l in range(num_layers):
			full_out = full_layers[l][t:t+1, 0:1, :].contiguous()  # shape (1,1,d_model)
			incr_out = incr[t][l].contiguous()
			diff = (full_out - incr_out).abs()
			maxdiff = diff.max().item()
			if maxdiff > atol:
				print(f"Mismatch at time t={t} layer={l} maxdiff={maxdiff:.6e}")
				print('full_out sample:', full_out.view(-1)[:16].detach().cpu().numpy())
				print('incr_out sample:', incr_out.view(-1)[:16].detach().cpu().numpy())
				# print detailed debug info when available
				fdbg = full_layer_debug[l]
				idbg = incr_debug[t][l]
				print('\n--- Full-path layer debug ---')
				if fdbg is None:
					print('no debug info for full path at this layer')
				else:
					att = fdbg.get('attn')
					if att is not None:
						print('full qh shape:', att['qh'].shape)
						print('full kh shape:', att['kh'].shape)
						print('full vh shape:', att['vh'].shape)
						print('full attn_scores sample:', att['attn_scores'].view(-1)[:16].numpy())
						print('full attn_weights sample:', att['attn_weights'].view(-1)[:16].numpy())
				print('\n--- Incr-path layer debug ---')
				if idbg is None:
					print('no debug info for incremental path at this layer')
				else:
					att = idbg.get('attn')
					if att is not None:
						print('incr qh shape:', att['qh'].shape)
						print('incr kh shape:', att['kh'].shape)
						print('incr vh shape:', att['vh'].shape)
						print('incr attn_scores sample:', att['attn_scores'].view(-1)[:16].numpy())
						print('incr attn_weights sample:', att['attn_weights'].view(-1)[:16].numpy())
				return False
	print('No mismatches detected within tolerance')
	return True


if __name__ == '__main__':
	torch.manual_seed(0)
	seq_len = 8
	id_dim = 16
	token_dim = id_dim + 10
	model = PolicyTransformer(token_dim=token_dim, d_model=64, nhead=8, num_layers=4, max_len=seq_len+1, mode='qnet')
	model.eval()
	seq = torch.randn((1, seq_len, token_dim))
	ok = find_first_mismatch(model, seq, atol=1e-6)
	if not ok:
		raise SystemExit(2)
	else:
		print('debug run complete - no mismatches')

