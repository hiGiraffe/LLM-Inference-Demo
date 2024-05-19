
    import torch

    修改llamaAttention的forward函数


    @profile
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        torch.cuda.synchronize(device)
        s1 = torch.cuda.Stream()
        s2 = torch.cuda.Stream()
        with torch.cuda.stream(s1):
            attn_output = self.attn(q, k, v, kv_cache, attn_metadata,
                                self.kv_scale)
        with torch.cuda.stream(s2):
            qkv, tmp = self.qkv_proj(hidden_states)
        output, _ = self.o_proj(attn_output)
        return output
