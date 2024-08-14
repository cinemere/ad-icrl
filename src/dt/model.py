from typing import Optional, Literal
import math
import torch
import torch.nn as nn
from torch import Tensor

class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float = 0,
                 maxlen: int = 5000
        ) -> None:
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding) # #[1, maxlen, emb_size]
        
    def forward(self, token_embedding: Tensor) -> Tensor:
        # return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])
        return self.dropout(token_embedding + self.pos_embedding[:, :token_embedding.size(1), :])

# Decision Transformer implementation
class TransformerBlock(nn.Module):
    def __init__(
        self,
        seq_len: int,
        hidden_dim: int,
        num_heads: int,
        attention_dropout: float,
        residual_dropout: float,
        ln_placem: Literal["postnorm", "prenorm"] = "postnorm",
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ln_placem = ln_placem
        self.drop = nn.Dropout(residual_dropout)

        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, attention_dropout, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(residual_dropout),
        )
        # True value indicates that the corresponding position is not allowed to attend
        self.register_buffer(
            "causal_mask", ~torch.tril(torch.ones(seq_len, seq_len)).to(bool)
        )
        self.seq_len = seq_len

    # [batch_size, seq_len, emb_dim] -> [batch_size, seq_len, emb_dim]
    def forward(
        self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        causal_mask = self.causal_mask[:x.shape[1], :x.shape[1]]

        if self.ln_placem == "postnorm":  # post-norm
            attention_out = self.attention(
                query=x,
                key=x,
                value=x,
                attn_mask=causal_mask,
                key_padding_mask=padding_mask,
                need_weights=False,
            )[0]
            # # by default pytorch attention does not use dropout
            # # after final attention weights projection, while minGPT does:
            # # https://github.com/karpathy/minGPT/blob/7218bcfa527c65f164de791099de715b81a95106/mingpt/model.py#L70 # noqa
            x = self.norm1(x + self.drop(attention_out))
            x = self.norm2(x + self.mlp(x))

        else:  # pre-norm
            norm_x = self.norm1(x)
            attention_out = self.attention(
                query=norm_x,
                key=norm_x,
                value=norm_x,
                attn_mask=causal_mask,
                key_padding_mask=padding_mask,
                need_weights=False,
            )[0]
            # by default pytorch attention does not use dropout
            # after final attention weights projection, while minGPT does:
            # https://github.com/karpathy/minGPT/blob/7218bcfa527c65f164de791099de715b81a95106/mingpt/model.py#L70 # noqa
            x = x + self.drop(attention_out)
            x = x + self.mlp(self.norm2(x))

        return x


class DecisionTransformer(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        seq_len: int = 10,
        embedding_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        embedding_dropout: float = 0.0,
        ln_placem: Literal["postnorm", "prenorm"] = "postnorm",
        add_reward_head: bool = False,
    ):
        super().__init__()        
        self.state_emb = nn.Embedding(state_dim, embedding_dim)
        self.action_emb = nn.Embedding(action_dim, embedding_dim)
        self.reward_emb = nn.Embedding(2, embedding_dim)
        # self.reward_emb = nn.Linear(1, embedding_dim)

        self.pos_enc = PositionalEncoding(embedding_dim, embedding_dropout, 3*seq_len)

        if embedding_dim != hidden_dim:
            self.emb2hid = nn.Linear(embedding_dim, hidden_dim)
    
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    seq_len=3 * seq_len,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    attention_dropout=attention_dropout,
                    residual_dropout=residual_dropout,
                    ln_placem=ln_placem,
                )
                for _ in range(num_layers)
            ]
        )
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.add_reward_head = add_reward_head
        if add_reward_head:
            self.reward_head = nn.Linear(hidden_dim * 2, 1)
        
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        # self.episode_len = episode_len
        # self.max_action = max_action

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        states: torch.Tensor,  # [batch_size, seq_len]
        actions: torch.Tensor,  # [batch_size, seq_len]
        rewards: torch.Tensor,  # [batch_size, seq_len]
        padding_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
    ) -> torch.FloatTensor:
        batch_size, seq_len = states.shape[0], states.shape[1]
        
        state_emb = self.pos_enc(self.state_emb(states))
        act_emb = self.pos_enc(self.action_emb(actions))
        rew_emb = self.pos_enc(self.reward_emb(rewards))
        # rew_emb = self.pos_enc(self.reward_emb(rewards.unsqueeze(-1)))
        # [batch_size, seq_len, emb_dim]

        # [batch_size, seq_len * 3, emb_dim], (s_0, a_0, r_0, s_1, a_1, r_1, ...)
        sequence = (
            torch.stack([state_emb, act_emb, rew_emb], dim=1)  # [batch_size, 3, seq_len, emb_dim]
            .permute(0, 2, 1, 3)  # [batch_size, seq_len, 3, emb_dim]
            .reshape(batch_size, 3 * seq_len, self.embedding_dim)
        )
        # sequence = self.pos_enc(sequence)  # positional encoding should be per-timestep
    
        if self.embedding_dim != self.hidden_dim:
            sequence = self.emb2hid(sequence)
        
        if padding_mask is not None:
            # [batch_size, seq_len * 3], stack mask identically to fit the sequence
            padding_mask = (
                torch.stack([padding_mask, padding_mask, padding_mask], dim=1)
                .permute(0, 2, 1)
                .reshape(batch_size, 3 * seq_len)
            )
        # # LayerNorm and Dropout (!!!) as in original implementation,
        # # while minGPT & huggingface uses only embedding dropout
        # out = self.emb_norm(sequence)
        # out = self.emb_drop(out)

        for block in self.blocks:
            sequence = block(sequence, padding_mask=padding_mask)
        # sequence = self.out_norm(sequence)

        # [batch_size, seq_len, emb_dim] -> [batch_size, seq_len, action_dim]
        # predict actions only from state embeddings
        state_embs = sequence[:, 0::3]
        
        action_output = self.action_head(state_embs)
        if self.add_reward_head:
            action_embs = sequence[:, 1::3]
                        
            reward_output = self.reward_head(torch.cat((state_embs, action_embs), dim=-1))
            # reward_output = self.reward_head(torch.cat((state_embs, action_output), dim=-1))
            return action_output, reward_output.squeeze(-1)
        
        return action_output, None