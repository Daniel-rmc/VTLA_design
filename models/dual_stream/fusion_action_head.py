"""
Fusion Action Head
融合动作头：智能地融合视觉和触觉特征以生成最终动作
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple


class FusionActionHead(nn.Module):
    """
    融合动作头：将视觉流和触觉流的输出融合为最终动作

    支持多种融合策略：
    - concat: 简单拼接后MLP
    - gated: 学习自适应权重
    - cross_attn: 交叉注意力融合
    - moe: 混合专家模型
    """

    def __init__(
        self,
        vision_dim: int = 512,
        tactile_dim: int = 512,
        action_dim: int = 14,
        hidden_dim: int = 512,
        fusion_type: str = 'gated',
        dropout: float = 0.1,
        predict_pad: bool = True,
    ):
        """
        Args:
            vision_dim: 视觉特征维度
            tactile_dim: 触觉特征维度
            action_dim: 动作空间维度
            hidden_dim: 隐藏层维度
            fusion_type: 融合类型 ('concat', 'gated', 'cross_attn', 'moe')
            dropout: dropout率
            predict_pad: 是否预测padding
        """
        super().__init__()

        self.fusion_type = fusion_type
        self.predict_pad = predict_pad

        if fusion_type == 'concat':
            self.fusion = ConcatFusion(
                vision_dim, tactile_dim, action_dim, hidden_dim, dropout
            )
        elif fusion_type == 'gated':
            self.fusion = GatedFusion(
                vision_dim, tactile_dim, action_dim, hidden_dim, dropout
            )
        elif fusion_type == 'cross_attn':
            self.fusion = CrossAttentionFusion(
                vision_dim, tactile_dim, action_dim, hidden_dim, dropout
            )
        elif fusion_type == 'moe':
            self.fusion = MoEFusion(
                vision_dim, tactile_dim, action_dim, hidden_dim, dropout
            )
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

        # Padding预测头
        if predict_pad:
            self.is_pad_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        vision_features: torch.Tensor,
        tactile_features: torch.Tensor,
        return_components: bool = False,
    ):
        """
        Args:
            vision_features: [B, T, D_v] 视觉流特征
            tactile_features: [B, T, D_t] 触觉流特征
            return_components: 是否返回中间组件

        Returns:
            actions: [B, T, action_dim] 预测动作
            is_pad_logits: [B, T, 1] padding预测（如果enabled）
            components: Dict 中间输出（如果return_components=True）
        """
        # 融合
        fused_features, fusion_info = self.fusion(vision_features, tactile_features)

        # 预测动作
        actions = fusion_info.get('actions')

        # 预测padding
        if self.predict_pad:
            is_pad_logits = self.is_pad_head(fused_features)
        else:
            is_pad_logits = None

        if return_components:
            components = {
                'fused_features': fused_features,
                'vision_features': vision_features,
                'tactile_features': tactile_features,
                **fusion_info
            }
            return actions, is_pad_logits, components
        else:
            return actions, is_pad_logits


class ConcatFusion(nn.Module):
    """简单的拼接融合"""

    def __init__(self, vision_dim, tactile_dim, action_dim, hidden_dim, dropout):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(vision_dim + tactile_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, vision_feat, tactile_feat):
        # 拼接
        concat = torch.cat([vision_feat, tactile_feat], dim=-1)
        fused = self.proj(concat)
        actions = self.action_head(fused)

        info = {
            'actions': actions,
            'fusion_weights': None,
        }
        return fused, info


class GatedFusion(nn.Module):
    """门控融合：学习自适应权重"""

    def __init__(self, vision_dim, tactile_dim, action_dim, hidden_dim, dropout):
        super().__init__()

        # 投影到统一维度
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.tactile_proj = nn.Linear(tactile_dim, hidden_dim)

        # 门控网络：根据上下文学习模态权重
        self.gate_net = nn.Sequential(
            nn.Linear(vision_dim + tactile_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1)  # [w_vision, w_tactile]
        )

        # 融合后的处理
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # 动作头
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, vision_feat, tactile_feat):
        # 投影
        vision_proj = self.vision_proj(vision_feat)  # [B, T, D]
        tactile_proj = self.tactile_proj(tactile_feat)  # [B, T, D]

        # 计算自适应权重
        concat_for_gate = torch.cat([vision_feat, tactile_feat], dim=-1)
        weights = self.gate_net(concat_for_gate)  # [B, T, 2]

        w_v = weights[..., 0:1]  # [B, T, 1]
        w_t = weights[..., 1:2]  # [B, T, 1]

        # 加权融合
        fused = w_v * vision_proj + w_t * tactile_proj

        # 后续处理
        fused = self.fusion_mlp(fused)
        actions = self.action_head(fused)

        info = {
            'actions': actions,
            'fusion_weights': weights,
            'vision_weight': w_v,
            'tactile_weight': w_t,
        }
        return fused, info


class CrossAttentionFusion(nn.Module):
    """交叉注意力融合：让视觉和触觉互相attend"""

    def __init__(self, vision_dim, tactile_dim, action_dim, hidden_dim, dropout):
        super().__init__()

        # 投影到统一维度
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.tactile_proj = nn.Linear(tactile_dim, hidden_dim)

        # 双向交叉注意力
        self.v2t_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        self.t2v_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        self.norm_v = nn.LayerNorm(hidden_dim)
        self.norm_t = nn.LayerNorm(hidden_dim)

        # 融合
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # 动作头
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, vision_feat, tactile_feat):
        # 投影
        vision_proj = self.vision_proj(vision_feat)
        tactile_proj = self.tactile_proj(tactile_feat)

        # Vision attends to Tactile
        v2t, _ = self.v2t_attn(
            query=vision_proj,
            key=tactile_proj,
            value=tactile_proj
        )
        vision_enhanced = self.norm_v(vision_proj + v2t)

        # Tactile attends to Vision
        t2v, _ = self.t2v_attn(
            query=tactile_proj,
            key=vision_proj,
            value=vision_proj
        )
        tactile_enhanced = self.norm_t(tactile_proj + t2v)

        # 拼接增强后的特征
        concat = torch.cat([vision_enhanced, tactile_enhanced], dim=-1)
        fused = self.fusion_mlp(concat)
        actions = self.action_head(fused)

        info = {
            'actions': actions,
            'fusion_weights': None,
            'vision_enhanced': vision_enhanced,
            'tactile_enhanced': tactile_enhanced,
        }
        return fused, info


class MoEFusion(nn.Module):
    """混合专家融合：路由到视觉专家或触觉专家"""

    def __init__(self, vision_dim, tactile_dim, action_dim, hidden_dim, dropout):
        super().__init__()

        # 视觉专家
        self.vision_expert = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim)
        )

        # 触觉专家
        self.tactile_expert = nn.Sequential(
            nn.Linear(tactile_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim)
        )

        # 融合专家
        self.fusion_expert = nn.Sequential(
            nn.Linear(vision_dim + tactile_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim)
        )

        # 路由网络：决定使用哪个专家
        self.router = nn.Sequential(
            nn.Linear(vision_dim + tactile_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # 3个专家
            nn.Softmax(dim=-1)
        )

        # 用于返回fused_features
        self.feature_proj = nn.Linear(vision_dim + tactile_dim, hidden_dim)

    def forward(self, vision_feat, tactile_feat):
        # 路由权重
        concat = torch.cat([vision_feat, tactile_feat], dim=-1)
        router_weights = self.router(concat)  # [B, T, 3]

        # 各专家输出
        vision_out = self.vision_expert(vision_feat)
        tactile_out = self.tactile_expert(tactile_feat)
        fusion_out = self.fusion_expert(concat)

        # 加权组合
        w_v = router_weights[..., 0:1]
        w_t = router_weights[..., 1:2]
        w_f = router_weights[..., 2:3]

        actions = w_v * vision_out + w_t * tactile_out + w_f * fusion_out

        fused = self.feature_proj(concat)

        info = {
            'actions': actions,
            'fusion_weights': router_weights,
            'vision_actions': vision_out,
            'tactile_actions': tactile_out,
            'fusion_actions': fusion_out,
        }
        return fused, info


class ContactAwareRouting(nn.Module):
    """
    接触感知路由：根据接触状态动态调整视觉和触觉的权重

    非接触阶段：视觉主导（粗粒度运动规划）
    接触阶段：触觉主导（精细力控）
    """

    def __init__(
        self,
        tactile_dim: int = 512,
        hidden_dim: int = 128,
    ):
        super().__init__()

        # 接触检测器
        self.contact_detector = nn.Sequential(
            nn.Linear(tactile_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # 权重调节器：将接触概率映射到[w_vision, w_tactile]
        # 接触概率高 → 触觉权重高
        # 接触概率低 → 视觉权重高

    def forward(self, tactile_features: torch.Tensor):
        """
        Args:
            tactile_features: [B, T, D_t]

        Returns:
            contact_prob: [B, T, 1] 接触概率
            modality_weights: [B, T, 2] [w_vision, w_tactile]
        """
        contact_prob = self.contact_detector(tactile_features)

        # 根据接触概率计算模态权重
        # contact_prob=0 → [1.0, 0.0] (纯视觉)
        # contact_prob=1 → [0.0, 1.0] (纯触觉)
        w_tactile = contact_prob
        w_vision = 1.0 - contact_prob

        modality_weights = torch.cat([w_vision, w_tactile], dim=-1)

        return contact_prob, modality_weights


if __name__ == '__main__':
    # 测试融合动作头
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    B, T, D_v, D_t, A = 2, 10, 512, 512, 14

    vision_feat = torch.randn(B, T, D_v).to(device)
    tactile_feat = torch.randn(B, T, D_t).to(device)

    print("=== Testing Fusion Action Heads ===\n")

    # 测试各种融合策略
    for fusion_type in ['concat', 'gated', 'cross_attn', 'moe']:
        print(f"--- {fusion_type.upper()} Fusion ---")

        head = FusionActionHead(
            vision_dim=D_v,
            tactile_dim=D_t,
            action_dim=A,
            fusion_type=fusion_type
        ).to(device)

        actions, is_pad, components = head(
            vision_feat,
            tactile_feat,
            return_components=True
        )

        print(f"  Actions: {actions.shape}")
        print(f"  Is_pad: {is_pad.shape if is_pad is not None else None}")

        if components['fusion_weights'] is not None:
            weights = components['fusion_weights']
            print(f"  Fusion weights shape: {weights.shape}")
            if fusion_type == 'gated':
                print(f"    Vision weight mean: {components['vision_weight'].mean():.3f}")
                print(f"    Tactile weight mean: {components['tactile_weight'].mean():.3f}")

        n_params = sum(p.numel() for p in head.parameters())
        print(f"  Parameters: {n_params / 1e3:.1f}K\n")

    # 测试接触感知路由
    print("--- Contact-Aware Routing ---")
    router = ContactAwareRouting(tactile_dim=D_t).to(device)
    contact_prob, modality_weights = router(tactile_feat)
    print(f"  Contact prob: {contact_prob.shape}, mean={contact_prob.mean():.3f}")
    print(f"  Modality weights: {modality_weights.shape}")
    print(f"    Vision weight mean: {modality_weights[..., 0].mean():.3f}")
    print(f"    Tactile weight mean: {modality_weights[..., 1].mean():.3f}")
