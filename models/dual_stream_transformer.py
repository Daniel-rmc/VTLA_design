"""
Dual-Stream Transformer
双流Transformer：为视觉和触觉维护独立的处理通路
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple
import copy


def _get_clones(module, N):
    """创建N个模块的深拷贝"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class DualStreamTransformer(nn.Module):
    """
    双流Transformer架构

    维护两个独立的encoder-decoder路径：
    - Vision Stream: 处理视觉信息
    - Tactile Stream: 处理触觉信息

    Args:
        d_model: 特征维度
        nhead: 注意力头数
        num_encoder_layers: encoder层数
        num_decoder_layers: decoder层数
        dim_feedforward: FFN维度
        dropout: dropout率
        shared_encoder: 是否共享encoder参数
        shared_decoder: 是否共享decoder参数
        enable_cross_stream: 是否在decoder中启用跨流交互
        cross_stream_layers: 启用跨流交互的层索引列表
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = 'relu',
        normalize_before: bool = False,
        shared_encoder: bool = True,
        shared_decoder: bool = False,
        enable_cross_stream: bool = False,
        cross_stream_layers: Optional[list] = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.shared_encoder = shared_encoder
        self.shared_decoder = shared_decoder
        self.enable_cross_stream = enable_cross_stream
        self.cross_stream_layers = cross_stream_layers or []

        # Encoder层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=normalize_before
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None

        if shared_encoder:
            # 共享encoder参数
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_encoder_layers,
                norm=encoder_norm
            )
            self.vision_encoder = self.encoder
            self.tactile_encoder = self.encoder
        else:
            # 独立encoder
            self.vision_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_encoder_layers,
                norm=encoder_norm
            )
            self.tactile_encoder = nn.TransformerEncoder(
                copy.deepcopy(encoder_layer),
                num_layers=num_encoder_layers,
                norm=copy.deepcopy(encoder_norm) if encoder_norm else None
            )

        # Decoder层
        if shared_decoder:
            # 共享decoder参数（通常不推荐）
            decoder_layer = DualStreamDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                batch_first=True,
                enable_cross_stream=enable_cross_stream
            )
            self.vision_decoder_layers = _get_clones(decoder_layer, num_decoder_layers)
            self.tactile_decoder_layers = self.vision_decoder_layers
        else:
            # 独立decoder
            vision_decoder_layer = DualStreamDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                batch_first=True,
                enable_cross_stream=enable_cross_stream
            )
            tactile_decoder_layer = DualStreamDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                batch_first=True,
                enable_cross_stream=enable_cross_stream
            )

            self.vision_decoder_layers = _get_clones(vision_decoder_layer, num_decoder_layers)
            self.tactile_decoder_layers = _get_clones(tactile_decoder_layer, num_decoder_layers)

        self.num_decoder_layers = num_decoder_layers

        # Decoder norm
        self.vision_decoder_norm = nn.LayerNorm(d_model)
        self.tactile_decoder_norm = nn.LayerNorm(d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        vision_src: torch.Tensor,
        tactile_src: torch.Tensor,
        query_embed: torch.Tensor,
        vision_pos: Optional[torch.Tensor] = None,
        tactile_pos: Optional[torch.Tensor] = None,
        vision_mask: Optional[torch.Tensor] = None,
        tactile_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            vision_src: [B, N_v, D] 视觉tokens
            tactile_src: [B, N_t, D] 触觉tokens
            query_embed: [T, D] query位置编码
            vision_pos: [B, N_v, D] 视觉位置编码
            tactile_pos: [B, N_t, D] 触觉位置编码
            vision_mask: [B, N_v] 视觉padding mask
            tactile_mask: [B, N_t] 触觉padding mask

        Returns:
            vision_output: [B, T, D] 视觉流decoder输出
            tactile_output: [B, T, D] 触觉流decoder输出
        """
        bs = vision_src.shape[0]

        # 添加位置编码
        if vision_pos is not None:
            vision_src = vision_src + vision_pos
        if tactile_pos is not None:
            tactile_src = tactile_src + tactile_pos

        # ===== Encoder: 独立编码 =====
        vision_memory = self.vision_encoder(
            vision_src,
            src_key_padding_mask=vision_mask
        )  # [B, N_v, D]

        tactile_memory = self.tactile_encoder(
            tactile_src,
            src_key_padding_mask=tactile_mask
        )  # [B, N_t, D]

        # ===== Decoder: 独立解码（可选跨流交互） =====
        # 初始化query
        num_queries = query_embed.shape[0]
        tgt = torch.zeros(bs, num_queries, self.d_model, device=vision_src.device)
        query_pos = query_embed.unsqueeze(0).expand(bs, -1, -1)  # [B, T, D]

        vision_output = tgt
        tactile_output = tgt

        # 逐层解码
        for layer_idx in range(self.num_decoder_layers):
            enable_cross_for_this_layer = (
                self.enable_cross_stream and
                layer_idx in self.cross_stream_layers
            )

            # Vision decoder layer
            vision_output = self.vision_decoder_layers[layer_idx](
                tgt=vision_output,
                memory=vision_memory,
                query_pos=query_pos,
                memory_key_padding_mask=vision_mask,
                cross_stream_memory=tactile_memory if enable_cross_for_this_layer else None,
                cross_stream_mask=tactile_mask if enable_cross_for_this_layer else None
            )

            # Tactile decoder layer
            tactile_output = self.tactile_decoder_layers[layer_idx](
                tgt=tactile_output,
                memory=tactile_memory,
                query_pos=query_pos,
                memory_key_padding_mask=tactile_mask,
                cross_stream_memory=vision_memory if enable_cross_for_this_layer else None,
                cross_stream_mask=vision_mask if enable_cross_for_this_layer else None
            )

        # 最终norm
        vision_output = self.vision_decoder_norm(vision_output)
        tactile_output = self.tactile_decoder_norm(tactile_output)

        return vision_output, tactile_output


class DualStreamDecoderLayer(nn.Module):
    """
    双流Decoder层
    支持可选的跨流交互
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = 'relu',
        batch_first: bool = True,
        enable_cross_stream: bool = False,
    ):
        super().__init__()

        self.enable_cross_stream = enable_cross_stream

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Cross-attention (query attends to memory)
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        # Cross-stream attention (可选)
        if enable_cross_stream:
            self.cross_stream_attn = nn.MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=batch_first
            )
            self.cross_stream_gate = nn.Linear(d_model, 1)
            self.norm_cross_stream = nn.LayerNorm(d_model)
            self.dropout_cross_stream = nn.Dropout(dropout)

        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU() if activation == 'relu' else nn.GELU()

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        query_pos: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        cross_stream_memory: Optional[torch.Tensor] = None,
        cross_stream_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            tgt: [B, T, D] target序列
            memory: [B, N, D] 当前流的memory
            query_pos: [B, T, D] query位置编码
            memory_key_padding_mask: [B, N] memory padding mask
            cross_stream_memory: [B, N', D] 另一个流的memory（可选）
            cross_stream_mask: [B, N'] 跨流memory的mask（可选）
        """
        # Self-attention with query positional encoding
        q = k = tgt + query_pos if query_pos is not None else tgt
        tgt2, _ = self.self_attn(q, k, tgt)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross-attention to same-stream memory
        q = tgt + query_pos if query_pos is not None else tgt
        tgt2, _ = self.cross_attn(
            q, memory, memory,
            key_padding_mask=memory_key_padding_mask
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Cross-stream attention (可选)
        if self.enable_cross_stream and cross_stream_memory is not None:
            q = tgt + query_pos if query_pos is not None else tgt
            cross_stream_info, _ = self.cross_stream_attn(
                q, cross_stream_memory, cross_stream_memory,
                key_padding_mask=cross_stream_mask
            )

            # Gated fusion: 让模型学习何时使用跨流信息
            gate = torch.sigmoid(self.cross_stream_gate(tgt))
            tgt2 = gate * cross_stream_info

            tgt = tgt + self.dropout_cross_stream(tgt2)
            tgt = self.norm_cross_stream(tgt)

        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


if __name__ == '__main__':
    # 测试双流Transformer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    B, N_v, N_t, T, D = 2, 100, 49, 10, 512

    # 创建模型
    model = DualStreamTransformer(
        d_model=D,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=6,
        shared_encoder=True,
        shared_decoder=False,
        enable_cross_stream=True,
        cross_stream_layers=[3, 5]  # 在第3和第5层启用跨流交互
    ).to(device)

    # 准备输入
    vision_src = torch.randn(B, N_v, D).to(device)
    tactile_src = torch.randn(B, N_t, D).to(device)
    query_embed = torch.randn(T, D).to(device)

    # 前向传播
    vision_out, tactile_out = model(
        vision_src=vision_src,
        tactile_src=tactile_src,
        query_embed=query_embed
    )

    print("Dual-Stream Transformer Test:")
    print(f"  Vision output: {vision_out.shape}")
    print(f"  Tactile output: {tactile_out.shape}")

    # 统计参数量
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {n_params / 1e6:.2f}M")

    # 测试不同配置
    print("\n--- Testing shared_encoder=False ---")
    model_independent = DualStreamTransformer(
        d_model=D,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=6,
        shared_encoder=False,
        shared_decoder=False,
    ).to(device)

    vision_out, tactile_out = model_independent(
        vision_src, tactile_src, query_embed
    )
    n_params_indep = sum(p.numel() for p in model_independent.parameters() if p.requires_grad)
    print(f"  Independent encoder parameters: {n_params_indep / 1e6:.2f}M")
    print(f"  Parameter increase: {(n_params_indep - n_params) / n_params * 100:.1f}%")
