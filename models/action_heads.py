"""
Action Heads Module
动作生成头：主动作头 + 触觉微调头 + 接触检测
"""
import torch
import torch.nn as nn
from typing import Optional


class MainActionHead(nn.Module):
    """
    主动作生成头
    基于融合后的视触觉特征生成动作序列
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        action_dim: int = 14,
        dropout: float = 0.1
    ):
        """
        Args:
            hidden_dim: 输入特征维度
            action_dim: 动作空间维度
            dropout: dropout率
        """
        super().__init__()

        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim)
        )

        # Pad预测头（预测是否为padding）
        self.is_pad_head = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor):
        """
        Args:
            features: [B, T, D] transformer输出特征

        Returns:
            actions: [B, T, action_dim]
            is_pad_logits: [B, T, 1]
        """
        actions = self.action_head(features)
        is_pad_logits = self.is_pad_head(features)

        return actions, is_pad_logits


class TactileRefineHead(nn.Module):
    """
    触觉微调头
    基于纯触觉特征生成动作残差，用于接触后的精细调整
    """

    def __init__(
        self,
        tactile_dim: int = 512,
        action_dim: int = 14,
        hidden_dim: int = 256,
        use_contact_gating: bool = True
    ):
        """
        Args:
            tactile_dim: 触觉特征维度
            action_dim: 动作空间维度
            hidden_dim: 隐藏层维度
            use_contact_gating: 是否使用接触门控
        """
        super().__init__()

        self.use_contact_gating = use_contact_gating

        # 触觉特征处理
        self.tactile_processor = nn.Sequential(
            nn.Linear(tactile_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # 动作残差生成
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # 限制残差范围在[-1, 1]
        )

        if use_contact_gating:
            # 接触强度门控（自适应权重）
            self.contact_gate = nn.Sequential(
                nn.Linear(tactile_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()  # 输出[0, 1]的权重
            )

    def forward(
        self,
        tactile_features: torch.Tensor,
        contact_strength: Optional[torch.Tensor] = None
    ):
        """
        Args:
            tactile_features: [B, T, D_t] 纯触觉特征序列
            contact_strength: [B, T, 1] 接触强度（可选，用于门控）

        Returns:
            action_residual: [B, T, action_dim] 动作残差
            gate_weight: [B, T, 1] 门控权重（如果使用）
        """
        # 处理触觉特征
        processed = self.tactile_processor(tactile_features)

        # 生成动作残差
        residual = self.residual_head(processed)

        # 计算门控权重
        if self.use_contact_gating:
            if contact_strength is not None:
                # 使用外部提供的接触强度
                gate_weight = contact_strength
            else:
                # 从触觉特征学习接触强度
                gate_weight = self.contact_gate(tactile_features)

            # 应用门控：接触越强，残差权重越大
            gated_residual = residual * gate_weight
            return gated_residual, gate_weight
        else:
            return residual, None


class ContactDetector(nn.Module):
    """
    接触检测模块
    从触觉特征中检测是否发生接触
    """

    def __init__(
        self,
        tactile_dim: int = 512,
        hidden_dim: int = 128
    ):
        super().__init__()

        self.detector = nn.Sequential(
            nn.Linear(tactile_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # 输出接触概率
        )

    def forward(self, tactile_features: torch.Tensor):
        """
        Args:
            tactile_features: [B, T, D_t] 或 [B, D_t]

        Returns:
            contact_prob: [B, T, 1] 或 [B, 1] 接触概率
        """
        return self.detector(tactile_features)


class DualPathActionHead(nn.Module):
    """
    双路动作生成头
    整合主动作头和触觉微调头
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        tactile_dim: int = 512,
        action_dim: int = 14,
        refine_scale: float = 0.1,
        adaptive_scale: bool = True
    ):
        """
        Args:
            hidden_dim: 主路特征维度
            tactile_dim: 触觉特征维度
            action_dim: 动作空间维度
            refine_scale: 触觉残差的缩放系数
            adaptive_scale: 是否使用自适应缩放
        """
        super().__init__()

        self.refine_scale = refine_scale
        self.adaptive_scale = adaptive_scale

        # 主动作头
        self.main_head = MainActionHead(
            hidden_dim=hidden_dim,
            action_dim=action_dim
        )

        # 触觉微调头
        self.refine_head = TactileRefineHead(
            tactile_dim=tactile_dim,
            action_dim=action_dim,
            use_contact_gating=True
        )

        # 接触检测器
        self.contact_detector = ContactDetector(tactile_dim=tactile_dim)

        if adaptive_scale:
            # 学习自适应缩放因子
            self.scale_predictor = nn.Sequential(
                nn.Linear(tactile_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()  # 输出[0, 1]
            )

    def forward(
        self,
        main_features: torch.Tensor,
        tactile_features: torch.Tensor,
        return_components: bool = False
    ):
        """
        Args:
            main_features: [B, T, D] 融合特征（视觉+触觉）
            tactile_features: [B, T, D_t] 纯触觉特征
            return_components: 是否返回各组件的输出

        Returns:
            final_actions: [B, T, action_dim] 最终动作
            is_pad_logits: [B, T, 1] padding预测
            components: dict (可选) 包含各组件输出
        """
        # 主动作预测
        main_actions, is_pad_logits = self.main_head(main_features)

        # 接触检测
        contact_prob = self.contact_detector(tactile_features)

        # 触觉微调残差
        action_residual, gate_weight = self.refine_head(
            tactile_features,
            contact_strength=contact_prob
        )

        # 计算最终动作
        if self.adaptive_scale:
            # 自适应缩放
            scale = self.scale_predictor(tactile_features)
            scaled_residual = action_residual * scale
        else:
            # 固定缩放
            scaled_residual = action_residual * self.refine_scale

        # 融合主动作和触觉残差
        final_actions = main_actions + scaled_residual

        if return_components:
            components = {
                'main_actions': main_actions,
                'action_residual': action_residual,
                'scaled_residual': scaled_residual,
                'contact_prob': contact_prob,
                'gate_weight': gate_weight,
                'adaptive_scale': scale if self.adaptive_scale else None
            }
            return final_actions, is_pad_logits, components
        else:
            return final_actions, is_pad_logits


if __name__ == '__main__':
    # 测试代码
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    B, T, D, D_t, A = 2, 10, 512, 512, 14

    # 测试主动作头
    main_head = MainActionHead(hidden_dim=D, action_dim=A).to(device)
    features = torch.randn(B, T, D).to(device)
    actions, is_pad = main_head(features)
    print(f"Main actions: {actions.shape}, is_pad: {is_pad.shape}")

    # 测试触觉微调头
    refine_head = TactileRefineHead(
        tactile_dim=D_t,
        action_dim=A,
        use_contact_gating=True
    ).to(device)
    tactile_features = torch.randn(B, T, D_t).to(device)
    residual, gate = refine_head(tactile_features)
    print(f"Tactile residual: {residual.shape}, gate: {gate.shape}")

    # 测试接触检测
    contact_detector = ContactDetector(tactile_dim=D_t).to(device)
    contact_prob = contact_detector(tactile_features)
    print(f"Contact probability: {contact_prob.shape}, mean: {contact_prob.mean():.3f}")

    # 测试双路动作头
    dual_head = DualPathActionHead(
        hidden_dim=D,
        tactile_dim=D_t,
        action_dim=A,
        adaptive_scale=True
    ).to(device)

    final_actions, is_pad, components = dual_head(
        features,
        tactile_features,
        return_components=True
    )

    print(f"\nDual-path outputs:")
    print(f"  Final actions: {final_actions.shape}")
    print(f"  Contact prob mean: {components['contact_prob'].mean():.3f}")
    print(f"  Gate weight mean: {components['gate_weight'].mean():.3f}")
    if components['adaptive_scale'] is not None:
        print(f"  Adaptive scale mean: {components['adaptive_scale'].mean():.3f}")
