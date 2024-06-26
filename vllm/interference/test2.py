

    import torch

    修改llamaDecoderLayer的forward函数


    @profile
    def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            kv_cache: torch.Tensor,
            attn_metadata: AttentionMetadata,
            residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention

        # 开始
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        hidden_states_ = hidden_states.clone()

        s1 = torch.cuda.Stream()
        s2 = torch.cuda.Stream()
        with torch.cuda.stream(s1):
            hidden_states1 = self.self_attn(
                positions=positions,
                hidden_states=hidden_states_,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
            )
        with torch.cuda.stream(s2):
            hidden_states2 = self.self_attn(
                positions=positions,
                hidden_states=hidden_states_,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
            )

        hidden_states=hidden_states1

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual