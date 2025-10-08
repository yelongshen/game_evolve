import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoderLayer(nn.Module):
    """A Transformer encoder layer that supports incremental encoding for new tokens
    given cached per-layer memories (past outputs). The layer exposes the same
    submodules as nn.TransformerEncoderLayer but provides a lightweight
    incremental path that only computes outputs for `new_x` while attending to
    `past` (concatenated as keys/values).
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        # Use explicit q/k/v projections so the layer can project and
        # cache keys/values independently of the attention combine step.
        # This avoids relying on the internal combined in-projection of
        # nn.MultiheadAttention which makes external KV caching awkward.
        self.nhead = nhead
        self.d_model = d_model
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.head_dim = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        # output projection after concatenating heads
        self.out_proj = nn.Linear(d_model, d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = F.relu

    def forward(self, src, src_mask=None):
        # src: (seq_len, batch, d_model)
        # project q/k/v
        q = self.q_proj(src)
        k = self.k_proj(src)
        v = self.v_proj(src)
        
        attn_out = self._multihead_attn(q, k, v, attn_mask=src_mask)
        src2 = src + self.dropout(attn_out)
        src2 = self.norm1(src2)
        ff = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        out = src2 + self.dropout(ff)
        out = self.norm2(out)
        return out

    def forward_incremental(self, past, new_x, position_idx=0):
        # past: (past_len, batch, d_model) or None
        # new_x: (new_len, batch, d_model)
        # we compute attention where queries=new_x and keys/values=concat(past, new_x)
        # Project queries for the new tokens
        #print('new_x', new_x)
        q = self.q_proj(new_x)

        k_new = self.k_proj(new_x)
        v_new = self.v_proj(new_x)

        #print('position_idx', position_idx)
        #print('q', q)
        #print('k', k_new)
        #print('v', v_new)
        
        past[0][position_idx:position_idx + k_new.size(0)] = k_new
        past[1][position_idx:position_idx + v_new.size(0)] = v_new

        k = past[0][0:position_idx + k_new.size(0)]
        v = past[1][0:position_idx + v_new.size(0)]

        attn_out = self._multihead_attn(q, k, v)
        
        #print('attn_out', attn_out)

        src2 = new_x + self.dropout(attn_out)
        #print('src1', src2)

        src2 = self.norm1(src2)
        #print('norm1', self.norm1.weight)
        #print('src2', src2)

        ff = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        out = src2 + self.dropout(ff)
        #print('ff', out)

        out = self.norm2(out)
        #print('out', out)

        return out

    def _multihead_attn(self, q, k, v, attn_mask=None):
        """Lightweight multi-head attention using explicit q/k/v projections.

        q, k, v: (seq_len, batch, d_model)
        attn_mask: optional (tgt_len, src_len) mask (same semantics as
                   nn.MultiheadAttention.attn_mask)
        Returns: attn_output: (seq_len, batch, d_model)
        """
        # shapes
        q_len, batch, _ = q.shape
        k_len = k.size(0)

        # reshape to (batch, nhead, seq_len, head_dim)
        def reshape_for_heads(x):
            # x: (seq_len, batch, d_model) -> (batch, nhead, seq_len, head_dim)
            seq_len = x.size(0)
            x = x.view(seq_len, batch, self.nhead, self.head_dim)
            x = x.permute(1, 2, 0, 3)
            return x

        qh = reshape_for_heads(q)  # (batch, nhead, q_len, head_dim)
        kh = reshape_for_heads(k)  # (batch, nhead, k_len, head_dim)
        vh = reshape_for_heads(v)  # (batch, nhead, k_len, head_dim)

        # merge batch and head dims for dot-product: (batch*nhead, q_len, head_dim)
        qh = qh.reshape(batch * self.nhead, q_len, self.head_dim)
        kh = kh.reshape(batch * self.nhead, k_len, self.head_dim)
        vh = vh.reshape(batch * self.nhead, k_len, self.head_dim)

        # scaled dot-product
        attn_scores = torch.bmm(qh, kh.transpose(1, 2))
        attn_scores = attn_scores / math.sqrt(self.head_dim)

        if attn_mask is not None:
            # attn_mask expected shape (q_len, k_len). Broadcast to (batch*nhead, q_len, k_len)
            if attn_mask.dim() == 2:
                mask = attn_mask.unsqueeze(0).expand(batch * self.nhead, -1, -1)
            else:
                mask = attn_mask
            attn_scores.masked_fill(attn_mask < 0.1, float('-inf'))
            #attn_scores = attn_scores + mask

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_out = torch.bmm(attn_weights, vh)  # (batch*nhead, q_len, head_dim)

        # reshape back to (seq_len, batch, d_model)
        attn_out = attn_out.view(batch, self.nhead, q_len, self.head_dim)
        attn_out = attn_out.permute(2, 0, 1, 3).contiguous()
        attn_out = attn_out.view(q_len, batch, self.d_model)

        # final linear projection
        attn_out = self.out_proj(attn_out)
        return attn_out


class PolicyTransformer(nn.Module):
    def __init__(self, token_dim, d_model=64, nhead=8, num_layers=8, max_len=16, mode='vnet'):
        super().__init__()
        
        # input projection layer.
        self.input_proj = nn.Linear(token_dim, d_model)
        # Replace built-in TransformerEncoder with a small stack of cached layers
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model=d_model, nhead=nhead)
                                     for _ in range(num_layers)])
        # learned positional embeddings (max_len should be >= history_len + 1)
        self.pos_emb = nn.Parameter(torch.zeros(max_len, d_model))
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

        
        # model weight initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)

        self.mode = mode
        if mode =='vnet':
            self.head = nn.Linear(d_model, 1)      # policy head (logit)
            self.value_head = nn.Linear(d_model, 1)  # value head
        elif mode == 'qnet' or mode == 'qnet-bay':
            self.n_actions = 2
            if mode == 'qnet':
                self.head = nn.Linear(d_model, self.n_actions)
            elif mode == 'qnet-bay':
                # For qnet we predict a Gaussian per action: (mu, logvar) for each action.
                # Current environment uses 2 discrete actions, so output dim = 2 actions * 2 (mu,logvar)
                self.head = nn.Linear(d_model, self.n_actions * 2)
            # initialize to small values
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
            # Per-position learnable rescaling (FiLM-style) to adapt q-value ranges
            # pos_scale and pos_bias have shape (max_len, n_actions) and are applied
            # to the raw head outputs so the model can adjust output ranges by position.
            max_len = self.pos_emb.size(0)
            self.pos_scale = nn.Parameter(torch.ones(max_len, self.n_actions))
            self.pos_bias = nn.Parameter(torch.zeros(max_len, self.n_actions))
            

    def build_cache(self, prealloc_len):
        """Allocate and return an empty (zero) KV cache preallocated to `prealloc_len`.

        Returns a list of `num_layers + 1` tensors with shape
        `(prealloc_len, batch=1, d_model)` right-aligned (all zeros initially).
        This function does not perform any forward computation — it only
        preallocates memory for per-agent KV caches.
        """
        max_len = self.pos_emb.size(0)
        if prealloc_len is None:
            prealloc_len = max_len

        d_model = self.pos_emb.size(1)
        device = self.pos_emb.device
        dtype = self.pos_emb.dtype

        # number of memory layers = input projection + one per transformer layer
        num_mem_layers = len(self.layers)
        # return a list of (k_buf, v_buf) tuples so callers can store projected
        # keys/values per memory layer: each buffer shape (prealloc_len, batch=1, d_model)
        preallocated = []
        for _ in range(num_mem_layers):
            k_buf = torch.zeros((prealloc_len, 1, d_model), device=device, dtype=dtype)
            v_buf = torch.zeros((prealloc_len, 1, d_model), device=device, dtype=dtype)
            preallocated.append((k_buf, v_buf))
        return preallocated

    def _encode(self, seq_tokens):
        """Internal helper: project inputs, add positional embeddings, apply
        the transformer layers with a causal mask and return the final layer
        outputs `out` with shape (seq_len, batch, d_model).

        Accepts the same input shapes as `encode_full`/`forward_all`.
        """
        # Build input projection and positional embeddings
        if seq_tokens.dim() == 2:
            seq_len = seq_tokens.size(0)
            x = self.input_proj(seq_tokens)  # (seq_len, d_model)
            x = x + self.pos_emb[:seq_len]
            x = x.unsqueeze(1)  # (seq_len, batch=1, d_model)
        elif seq_tokens.dim() == 3:
            batch, seq_len, _ = seq_tokens.shape
            x = self.input_proj(seq_tokens)  # (batch, seq_len, d_model)
            x = x + self.pos_emb[:seq_len].unsqueeze(0)
            x = x.permute(1, 0, 2)  # (seq_len, batch, d_model)
        else:
            raise ValueError("seq_tokens must be 2D or 3D tensor")

        # causal mask to prevent attention to future tokens
        seq_len = x.size(0)
        src_mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device, dtype=x.dtype))

        out = x
        for layer in self.layers:
            out = layer.forward(out, src_mask=src_mask)
        # normalize return shape: always return (batch, seq_len, d_model)
        # internal representation `out` is (seq_len, batch, d_model)
        return out.permute(1, 0, 2)


    def forward(self, seq_tokens):
        """Full forward that returns per-timestep probabilities and values.

        Input: `seq_tokens` can be either (seq_len, token_dim) or
        (batch, seq_len, token_dim). Returns `(probs, values)` with shapes
        `(batch, seq_len)` each. Padding should be handled by the caller via
        masks.
        """
        out = self._encode(seq_tokens)
        # out: (batch, seq_len, d_model) -> project to logits/values per timestep

        if self.mode == 'vnet':
            logits = self.head(out)  # (batch, seq_len, 1)
            values = self.value_head(out)  # (batch, seq_len, 1)

            # reshape to (batch, seq_len)
            probs = torch.sigmoid(logits).squeeze(-1)
            values = values.squeeze(-1)
            return probs, values
        elif self.mode =='qnet':
            qvalues = self.head(out).contiguous()  # (batch, seq_len, n_actions)
            # Apply per-position scale and bias to adapt dynamic ranges across time
            seq_len = qvalues.size(1)
            # pos_scale/bias: (max_len, n_actions) -> take first seq_len positions
            scale = self.pos_scale[:seq_len].unsqueeze(0).to(qvalues.device)
            bias = self.pos_bias[:seq_len].unsqueeze(0).to(qvalues.device)
            qvalues = qvalues * scale + bias
            return qvalues
        elif self.mode == 'qnet-bay':
            # head output shape: (batch, seq_len, n_actions*2)
            q_out = self.head(out).contiguous()
            batch, seq_len, _ = q_out.shape
            q_out = q_out.view(batch, seq_len, self.n_actions, 2)
            # split into mean and logvar
            mu = q_out[..., 0]
            logvar = q_out[..., 1]
            # apply per-position scaling to the predicted means
            scale = self.pos_scale[:seq_len].unsqueeze(0).to(mu.device)
            bias = self.pos_bias[:seq_len].unsqueeze(0).to(mu.device)
            mu = mu * scale + bias
            return mu, logvar
        
    def forward_with_cache(self, memories, new_tokens, position_idx=0):
        """Compute outputs for `new_tokens` (seq_len_new, token_dim) using cached
        per-layer `memories` (as returned by build_cache). This does not mutate
        the provided memories unless `update_memories=True` in which case the
        function also returns updated detached per-layer memories suitable for
        storing as an agent's KV cache.

        Parameters:
          - memories: list of per-layer tensors (or None)
          - new_tokens: new token(s) to process
          - update_memories: if True, also return updated memories
          - position_idx: optional integer start index to use for positional embeddings
            (overrides automatic detection). Use this when caller already tracks
            how many tokens are present in the cached memories.

        Backwards-compatible return values:
          - (prob, value, new_layer_outputs)
        New optional return values when `update_memories=True`:
          - (prob, value, new_layer_outputs, updated_memories)
        """
        # Normalize shapes to (seq_len_new, batch, token_dim)
        if new_tokens is None:
            raise ValueError("new_tokens must be provided (got None)")

        # Determine start position (how many tokens are already in memories)
        start_pos = int(position_idx)
        
        if new_tokens.dim() == 1:
            new_tokens = new_tokens.unsqueeze(0)
        if new_tokens.dim() == 2:
            # new_tokens is (new_seq_len, token_dim) -> make batch dim
            new_seq_len, token_dim = new_tokens.shape
            new_x = self.input_proj(new_tokens) + self.pos_emb[start_pos:start_pos + new_seq_len]
            new_x = new_x.unsqueeze(1)  # (new_len, batch=1, d_model)
        else:
            raise ValueError("new_tokens must be 2D or 3D tensor")

        #print('embedding', new_x)

        # new_x is now (new_len, batch, d_model)
        new_layer_outputs = [new_x]
        out = new_x
        # For each layer, attend over past (memories[layer_idx+0]) and new outputs
        for l_idx, layer in enumerate(self.layers):
            out = layer.forward_incremental(memories[l_idx], out, position_idx=start_pos)
            new_layer_outputs.append(out)

            #print('layer', l_idx, out)

        # final representation is last token of last layer (batch, d_model)
        rep = out[-1, 0, :]

        if self.mode == 'vnet':
            logit = self.head(rep)
            value = self.value_head(rep).squeeze(-1)
            prob = torch.sigmoid(logit).squeeze(-1)

            return prob, value
        elif self.mode =='qnet':
            q_values = self.head(rep)
            # apply per-position scaling for this token's absolute position
            pos = start_pos + new_x.size(0) - 1
            max_pos = self.pos_scale.size(0)
            pos = min(pos, max_pos - 1)
            scale = self.pos_scale[pos].to(q_values.device)
            bias = self.pos_bias[pos].to(q_values.device)
            q_values = q_values * scale + bias
            return q_values
        elif self.mode == 'qnet-bay':
            # rep: (d_model,) -> head -> (n_actions*2,)
            q_out = self.head(rep)
            q_out = q_out.view(self.n_actions, 2)
            mu = q_out[:, 0]
            logvar = q_out[:, 1]
            pos = start_pos + new_x.size(0) - 1
            max_pos = self.pos_scale.size(0)
            pos = min(pos, max_pos - 1)
            scale = self.pos_scale[pos].to(mu.device)
            bias = self.pos_bias[pos].to(mu.device)
            mu = mu * scale + bias
            return mu, logvar