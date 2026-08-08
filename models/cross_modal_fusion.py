"""
Cross-Modal Fusion Module
视觉和触觉特征的交叉注意力融合
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CrossModalFusion(nn.Module):
    """
    视触觉交叉注意力融合模块

    实现双向交叉注意力：
    1. Vision -> Tactile: 视觉token查询触觉信息
    2. Tactile -> Vision: 触觉token查询视觉信息
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        dropout: float = 0.1,
        fusion_type: str = 'bidirectional'
    ):
        """
        Args:
            d_model: 特征维度
            nhead: 注意力头数
            dropout: dropout率
            fusion_type: 融合类型 ('bidirectional', 'v2t', 't2v', 'concat')
        """
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.fusion_type = fusion_type

        if fusion_type in ['bidirectional', 'v2t']:
            # Vision queries Tactile (视觉查询触觉)
            self.v2t_attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=True
            )
            self.v2t_norm = nn.LayerNorm(d_model)
            self.v2t_ffn = nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout)
            )
            self.v2t_ffn_norm = nn.LayerNorm(d_model)

        if fusion_type in ['bidirectional', 't2v']:
            # Tactile queries Vision (触觉查询视觉)
            self.t2v_attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=True
            )
            self.t2v_norm = nn.LayerNorm(d_model)
            self.t2v_ffn = nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout)
            )
            self.t2v_ffn_norm = nn.LayerNorm(d_model)

        if fusion_type == 'concat':
            # 简单拼接后投影
            self.concat_proj = nn.Linear(d_model * 2, d_model)

    def forward(
        self,
        vision_tokens: torch.Tensor,
        tactile_tokens: torch.Tensor,
        vision_mask: Optional[torch.Tensor] = None,
        tactile_mask: Optional[torch.Tensor] = None
    ):
        """
        Args:
            vision_tokens: [B, N_v, D] 视觉token序列
            tactile_tokens: [B, N_t, D] 触觉token序列
            vision_mask: [B, N_v] 视觉mask (optional)
            tactile_mask: [B, N_t] 触觉mask (optional)

        Returns:
            fused_vision: [B, N_v, D] 融合后的视觉特征
            fused_tactile: [B, N_t, D] 融合后的触觉特征
        """

        if self.fusion_type == 'concat':
            # 简单拼接融合
            # 需要两者token数量相同或进行池化
            if vision_tokens.shape[1] != tactile_tokens.shape[1]:
                # 池化到相同长度
                min_len = min(vision_tokens.shape[1], tactile_tokens.shape[1])
                vision_tokens = F.adaptive_avg_pool1d(
                    vision_tokens.transpose(1, 2), min_len
                ).transpose(1, 2)
                tactile_tokens = F.adaptive_avg_pool1d(
                    tactile_tokens.transpose(1, 2), min_len
                ).transpose(1, 2)

            concatenated = torch.cat([vision_tokens, tactile_tokens], dim=-1)
            fused = self.concat_proj(concatenated)
            return fused, fused

        # 交叉注意力融合
        fused_vision = vision_tokens
        fused_tactile = tactile_tokens

        if self.fusion_type in ['bidirectional', 'v2t']:
            # Vision queries Tactile
            # Query: vision, Key/Value: tactile
            attn_output, _ = self.v2t_attention(
                query=fused_vision,
                key=tactile_tokens,
                value=tactile_tokens,
                key_padding_mask=tactile_mask
            )
            fused_vision = self.v2t_norm(fused_vision + attn_output)

            # FFN
            ffn_output = self.v2t_ffn(fused_vision)
            fused_vision = self.v2t_ffn_norm(fused_vision + ffn_output)

        if self.fusion_type in ['bidirectional', 't2v']:
            # Tactile queries Vision
            # Query: tactile, Key/Value: vision
            attn_output, _ = self.t2v_attention(
                query=fused_tactile,
                key=vision_tokens,
                value=vision_tokens,
                key_padding_mask=vision_mask
            )
            fused_tactile = self.t2v_norm(fused_tactile + attn_output)

            # FFN
            ffn_output = self.t2v_ffn(fused_tactile)
            fused_tactile = self.t2v_ffn_norm(fused_tactile + ffn_output)

        return fused_vision, fused_tactile


class BiDirectionalCrossAttention(nn.Module):
    """
    双向交叉注意力模块（多层堆叠版本）
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            CrossModalFusion(
                d_model=d_model,
                nhead=nhead,
                dropout=dropout,
                fusion_type='bidirectional'
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        vision_tokens: torch.Tensor,
        tactile_tokens: torch.Tensor,
        vision_mask: Optional[torch.Tensor] = None,
        tactile_mask: Optional[torch.Tensor] = None
    ):
        """
        多层双向交叉注意力

        Returns:
            fused_vision: [B, N_v, D]
            fused_tactile: [B, N_t, D]
        """
        for layer in self.layers:
            vision_tokens, tactile_tokens = layer(
                vision_tokens,
                tactile_tokens,
                vision_mask,
                tactile_mask
            )

        return vision_tokens, tactile_tokens


class AdaptiveFusion(nn.Module):
    """
    自适应融合模块
    根据当前状态动态调整视觉和触觉的权重
    """

    def __init__(self, d_model: int = 512):
        super().__init__()

        # 门控机制：学习视觉和触觉的相对重要性
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
            nn.Softmax(dim=-1)
        )

    def forward(
        self,
        vision_features: torch.Tensor,
        tactile_features: torch.Tensor
    ):
        """
        Args:
            vision_features: [B, D] 全局视觉特征
            tactile_features: [B, D] 全局触觉特征

        Returns:
            fused_features: [B, D] 自适应融合后的特征
            weights: [B, 2] 融合权重 [w_vision, w_tactile]
        """
        # 拼接特征
        concatenated = torch.cat([vision_features, tactile_features], dim=-1)

        # 计算自适应权重
        weights = self.gate_net(concatenated)  # [B, 2]

        # 加权融合
        w_v = weights[:, 0:1]  # [B, 1]
        w_t = weights[:, 1:2]  # [B, 1]

        fused = w_v * vision_features + w_t * tactile_features

        return fused, weights


if __name__ == '__main__':
    # 测试代码
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    B, N_v, N_t, D = 2, 100, 49, 512

    # 模拟视觉和触觉tokens
    vision_tokens = torch.randn(B, N_v, D).to(device)
    tactile_tokens = torch.randn(B, N_t, D).to(device)

    # 测试交叉注意力融合
    fusion = CrossModalFusion(
        d_model=D,
        nhead=8,
        fusion_type='bidirectional'
    ).to(device)

    fused_v, fused_t = fusion(vision_tokens, tactile_tokens)
    print(f"Fused vision tokens: {fused_v.shape}")
    print(f"Fused tactile tokens: {fused_t.shape}")

    # 测试多层双向交叉注意力
    bi_fusion = BiDirectionalCrossAttention(
        d_model=D,
        nhead=8,
        num_layers=2
    ).to(device)

    fused_v, fused_t = bi_fusion(vision_tokens, tactile_tokens)
    print(f"\nMulti-layer fused vision: {fused_v.shape}")
    print(f"Multi-layer fused tactile: {fused_t.shape}")

    # 测试自适应融合
    vision_global = torch.randn(B, D).to(device)
    tactile_global = torch.randn(B, D).to(device)

    adaptive_fusion = AdaptiveFusion(d_model=D).to(device)
    fused_global, weights = adaptive_fusion(vision_global, tactile_global)
    print(f"\nAdaptive fused features: {fused_global.shape}")
    print(f"Fusion weights: {weights}")
