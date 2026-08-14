"""
Dual-Stream VTLA Policy
双流视触觉动作模型：保持模态独立性的新架构
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
import sys
from pathlib import Path

# 导入自定义模块
from ..shared import TactileEncoder
from .transformer import DualStreamTransformer
from .fusion_action_head import FusionActionHead, ContactAwareRouting

# 添加UniVTAC路径
univtac_base = Path(__file__).resolve().parents[3] / 'UniVTAC'
if univtac_base.is_dir():
    univtac_path = str(univtac_base)
    if univtac_path not in sys.path:
        sys.path.append(univtac_path)

try:
    from policy.ACT.detr.models.backbone import build_backbone
    UNIVTAC_AVAILABLE = True
except ImportError:
    try:
        from models.backbone import build_backbone
        UNIVTAC_AVAILABLE = True
    except ImportError:
        print("Warning: Could not import UniVTAC modules. Make sure UniVTAC is in the path.")
        UNIVTAC_AVAILABLE = False
        def build_backbone(args):
            raise ImportError("UniVTAC modules not available")


def reparametrize(mu, logvar):
    """CVAE重参数化技巧"""
    std = logvar.div(2).exp()
    eps = torch.randn_like(std)
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    """生成正弦位置编码"""
    def get_position_angle_vec(position):
        return [position / (10000 ** (2 * (hid_j // 2) / d_hid)) for hid_j in range(d_hid)]

    import numpy as np
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


def get_2d_sinusoid_encoding(height, width, d_model, device, dtype):
    """生成2D正弦位置编码"""
    if d_model % 4 != 0:
        raise ValueError(f"d_model must be divisible by 4, got {d_model}")
    quarter_dim = d_model // 4
    omega = torch.arange(quarter_dim, device=device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / max(quarter_dim - 1, 1)))
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing='ij',
    )
    x = x.reshape(-1, 1) * omega.reshape(1, -1)
    y = y.reshape(-1, 1) * omega.reshape(1, -1)
    encoding = torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=1)
    return encoding.to(dtype=dtype).transpose(0, 1).reshape(1, d_model, height, width)


class DualStreamVTLAModel(nn.Module):
    """
    双流VTLA模型：保持视觉和触觉的独立性

    架构流程：
    1. 独立特征提取：Vision Backbone + Tactile Encoder
    2. 添加位置编码和模态编码
    3. 双流Transformer：独立的encoder-decoder路径
    4. 融合动作头：晚期融合生成最终动作
    """

    def __init__(
        self,
        vision_backbone,
        tactile_encoder: TactileEncoder,
        dual_stream_transformer: DualStreamTransformer,
        state_dim: int,
        num_queries: int,
        camera_names: list,
        tactile_names: list,
        hidden_dim: int = 512,
        fusion_type: str = 'gated',
        use_contact_routing: bool = False,
        use_cvae: bool = True,
        latent_dim: int = 32,
    ):
        super().__init__()

        self.num_queries = num_queries
        self.camera_names = camera_names
        self.tactile_names = tactile_names
        self.hidden_dim = hidden_dim
        self.use_contact_routing = use_contact_routing
        self.use_cvae = use_cvae
        self.latent_dim = latent_dim

        # 1. 特征提取模块
        self.vision_backbone = vision_backbone
        self.tactile_encoder = tactile_encoder

        # 2. 特征投影层
        self.vision_input_proj = nn.Conv2d(
            vision_backbone.num_channels, hidden_dim, kernel_size=1
        )
        self.tactile_input_proj = nn.Conv2d(
            tactile_encoder.latent_dim,
            hidden_dim,
            kernel_size=1
        )

        # 本体感觉投影
        self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)

        # 模态source embedding
        self.vision_source_embed = nn.Embedding(len(camera_names), hidden_dim)
        self.tactile_source_embed = nn.Embedding(len(tactile_names), hidden_dim)

        # 3. 双流Transformer
        self.dual_stream_transformer = dual_stream_transformer

        # Query embedding
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # 4. CVAE编码器（可选，用于行为多样性）
        if use_cvae:
            self.cls_embed = nn.Embedding(1, hidden_dim)
            self.encoder_action_proj = nn.Linear(state_dim, hidden_dim)
            self.encoder_joint_proj = nn.Linear(state_dim, hidden_dim)

            # CVAE encoder (简单的transformer encoder)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                activation='relu',
                batch_first=True
            )
            self.cvae_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

            self.latent_proj = nn.Linear(hidden_dim, latent_dim * 2)  # mu + logvar
            self.latent_out_proj = nn.Linear(latent_dim, hidden_dim)

            # 位置编码
            self.register_buffer(
                'pos_table',
                get_sinusoid_encoding_table(1 + 1 + num_queries, hidden_dim)
            )

        # 5. 融合动作头
        self.fusion_action_head = FusionActionHead(
            vision_dim=hidden_dim,
            tactile_dim=hidden_dim,
            action_dim=state_dim,
            hidden_dim=hidden_dim,
            fusion_type=fusion_type,
            predict_pad=True
        )

        # 6. 接触感知路由（可选）
        if use_contact_routing:
            self.contact_router = ContactAwareRouting(tactile_dim=hidden_dim)

    def forward(
        self,
        qpos: torch.Tensor,
        cam_image: torch.Tensor,
        tac_image: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        is_pad: Optional[torch.Tensor] = None,
        return_components: bool = False,
        deterministic_latent: bool = False,
    ):
        """
        Args:
            qpos: [B, state_dim] 机器人状态
            cam_image: [B, N_cam, C, H, W] 相机图像
            tac_image: [B, N_tac, C, H, W] 触觉图像
            actions: [B, T, state_dim] 动作序列（训练时）
            is_pad: [B, T] padding mask（训练时）
            return_components: 是否返回中间组件
            deterministic_latent: 是否使用确定性latent

        Returns:
            训练时: (actions_pred, is_pad_pred, (mu, logvar), components)
            推理时: (actions_pred, is_pad_pred)
        """
        is_training = actions is not None
        bs = qpos.shape[0]

        # ===== 1. CVAE编码（训练时，可选） =====
        mu = logvar = None
        if self.use_cvae and is_training:
            action_embed = self.encoder_action_proj(actions)
            qpos_embed = self.encoder_joint_proj(qpos).unsqueeze(1)
            cls_embed = self.cls_embed.weight.unsqueeze(0).repeat(bs, 1, 1)

            encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], dim=1)

            cls_joint_is_pad = torch.full((bs, 2), False, device=qpos.device)
            is_pad_full = torch.cat([cls_joint_is_pad, is_pad], dim=1)

            # CVAE encoder
            encoder_output = self.cvae_encoder(encoder_input)
            encoder_output = encoder_output[:, 0]  # CLS token

            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = mu if deterministic_latent else reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        elif self.use_cvae:
            # 推理时使用零latent
            latent_sample = torch.zeros(bs, self.latent_dim, device=qpos.device)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            latent_input = None

        # ===== 2. 视觉特征提取 =====
        vision_features_list = []

        for cam_id in range(len(self.camera_names)):
            features, pos = self.vision_backbone(cam_image[:, cam_id])
            features = features[0]  # 最后一层
            pos = pos[0]

            projected = self.vision_input_proj(features)
            source = self.vision_source_embed.weight[cam_id].view(1, -1, 1, 1)
            vision_features_list.append(projected + pos + source)

        # 拼接所有相机特征
        if len(vision_features_list) > 0:
            vision_features = torch.cat(
                [f.flatten(2) for f in vision_features_list], dim=2
            )  # [B, D, N_v]
            vision_tokens = vision_features.permute(0, 2, 1)  # [B, N_v, D]
        else:
            raise ValueError("At least one camera is required")

        # ===== 3. 触觉特征提取 =====
        tactile_features_list = []

        for tac_id in range(len(self.tactile_names)):
            tac_feat = self.tactile_encoder(
                tac_image[:, tac_id], return_tokens=True
            )  # [B, D, H', W']
            tac_feat = self.tactile_input_proj(tac_feat)

            # 2D正弦位置编码
            tactile_pos = get_2d_sinusoid_encoding(
                tac_feat.shape[-2], tac_feat.shape[-1], self.hidden_dim,
                tac_feat.device, tac_feat.dtype
            )

            source = self.tactile_source_embed.weight[tac_id].view(1, -1, 1, 1)
            tactile_features_list.append(tac_feat + tactile_pos + source)

        # 拼接所有触觉传感器特征
        if len(tactile_features_list) > 0:
            tactile_features = torch.cat(
                [f.flatten(2) for f in tactile_features_list], dim=2
            )  # [B, D, N_t]
            tactile_tokens = tactile_features.permute(0, 2, 1)  # [B, N_t, D]
        else:
            raise ValueError("At least one tactile sensor is required")

        # ===== 4. 添加latent和本体感觉（如果使用CVAE） =====
        if self.use_cvae and latent_input is not None:
            proprio_input = self.input_proj_robot_state(qpos)  # [B, D]

            # 作为额外的tokens添加到vision和tactile
            # 简化处理：添加到vision tokens前面
            additional_tokens = torch.stack([latent_input, proprio_input], dim=1)  # [B, 2, D]
            vision_tokens = torch.cat([additional_tokens, vision_tokens], dim=1)

        # ===== 5. 双流Transformer =====
        vision_output, tactile_output = self.dual_stream_transformer(
            vision_src=vision_tokens,
            tactile_src=tactile_tokens,
            query_embed=self.query_embed.weight,
        )  # [B, T, D], [B, T, D]

        # ===== 6. 接触感知路由（可选） =====
        contact_info = {}
        if self.use_contact_routing:
            # 从触觉decoder输出检测接触
            contact_prob, modality_weights = self.contact_router(tactile_output)
            contact_info = {
                'contact_prob': contact_prob,
                'modality_weights': modality_weights
            }

        # ===== 7. 融合动作头 =====
        if return_components:
            actions_pred, is_pad_pred, fusion_components = self.fusion_action_head(
                vision_output,
                tactile_output,
                return_components=True
            )
            components = {
                **fusion_components,
                **contact_info,
                'vision_decoder_output': vision_output,
                'tactile_decoder_output': tactile_output,
            }
        else:
            actions_pred, is_pad_pred = self.fusion_action_head(
                vision_output,
                tactile_output,
                return_components=False
            )
            components = None

        # ===== 8. 返回结果 =====
        if is_training:
            if return_components:
                return actions_pred, is_pad_pred, (mu, logvar), components
            else:
                return actions_pred, is_pad_pred, (mu, logvar)
        else:
            if return_components:
                return actions_pred, is_pad_pred, components
            else:
                return actions_pred, is_pad_pred


class DualStreamVTLAPolicy(nn.Module):
    """
    双流VTLA策略包装器：负责损失计算和训练/推理接口
    """

    def __init__(self, args_override):
        super().__init__()
        self.model = build_dual_stream_vtla_model(args_override)

        # 损失权重
        self.kl_weight = _cfg(args_override, 'kl_weight', 10.0)
        self.pad_weight = _cfg(args_override, 'pad_weight', 1.0)
        self.l1_reduction = _cfg(args_override, 'l1_reduction', 'valid_mean')

        # 辅助损失权重（可选）
        self.aux_vision_weight = _cfg(args_override, 'aux_vision_weight', 0.0)
        self.aux_tactile_weight = _cfg(args_override, 'aux_tactile_weight', 0.0)

    def forward(
        self,
        qpos,
        cam_image,
        tac_image,
        actions=None,
        is_pad=None,
        deterministic_latent=False,
    ):
        """
        前向传播并计算损失（训练）或返回预测动作（推理）
        """
        if actions is not None:  # 训练
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar), components = self.model(
                qpos,
                cam_image,
                tac_image,
                actions,
                is_pad,
                return_components=True,
                deterministic_latent=deterministic_latent,
            )

            loss_dict = self._compute_loss(
                a_hat, actions, is_pad, is_pad_hat, mu, logvar, components
            )
            return loss_dict
        else:  # 推理
            a_hat, is_pad_hat = self.model(qpos, cam_image, tac_image)
            return a_hat

    def _compute_loss(self, a_hat, actions, is_pad, is_pad_hat, mu, logvar, components):
        """计算各项损失"""
        from torch.nn import functional as F

        loss_dict = {}

        # 主动作L1损失
        all_l1 = F.l1_loss(a_hat, actions, reduction='none')
        valid = (~is_pad).unsqueeze(-1).expand_as(all_l1)

        if self.l1_reduction == 'official_mean':
            l1 = (all_l1 * valid).mean()
        elif self.l1_reduction == 'valid_mean':
            l1 = (all_l1 * valid).sum() / valid.sum().clamp_min(1)
        else:
            raise ValueError(f"Unsupported l1_reduction: {self.l1_reduction}")

        loss_dict['l1'] = l1

        # Padding损失
        if is_pad_hat is not None:
            pad_loss = F.binary_cross_entropy_with_logits(
                is_pad_hat.squeeze(-1), is_pad.float()
            )
            loss_dict['pad'] = pad_loss
        else:
            loss_dict['pad'] = torch.tensor(0.0, device=l1.device)

        # KL散度
        if mu is not None and logvar is not None:
            kl_loss = self._kl_divergence(mu, logvar)
            loss_dict['kl'] = kl_loss
        else:
            loss_dict['kl'] = torch.tensor(0.0, device=l1.device)

        # 辅助损失：单独监督vision和tactile分支（可选）
        # 这可以帮助每个流学到有用的表示
        if self.aux_vision_weight > 0 and 'vision_actions' in components:
            vision_l1 = F.l1_loss(components['vision_actions'], actions, reduction='none')
            vision_l1 = (vision_l1 * valid).sum() / valid.sum().clamp_min(1)
            loss_dict['aux_vision'] = vision_l1
        else:
            loss_dict['aux_vision'] = torch.tensor(0.0, device=l1.device)

        if self.aux_tactile_weight > 0 and 'tactile_actions' in components:
            tactile_l1 = F.l1_loss(components['tactile_actions'], actions, reduction='none')
            tactile_l1 = (tactile_l1 * valid).sum() / valid.sum().clamp_min(1)
            loss_dict['aux_tactile'] = tactile_l1
        else:
            loss_dict['aux_tactile'] = torch.tensor(0.0, device=l1.device)

        # 总损失
        loss_dict['loss'] = (
            loss_dict['l1'] +
            self.kl_weight * loss_dict['kl'] +
            self.pad_weight * loss_dict['pad'] +
            self.aux_vision_weight * loss_dict['aux_vision'] +
            self.aux_tactile_weight * loss_dict['aux_tactile']
        )

        return loss_dict

    @staticmethod
    def _kl_divergence(mu, logvar):
        """计算KL散度"""
        klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        total_kld = klds.sum(1).mean(0, True)
        return total_kld[0]


def build_dual_stream_vtla_model(args):
    """
    构建双流VTLA模型

    Args:
        args: 配置参数（dict或object）
    """
    # 如果args是dict，转换为object以兼容UniVTAC的build_backbone
    if isinstance(args, dict):
        class ArgsObject:
            def __init__(self, d):
                for key, value in d.items():
                    setattr(self, key, value)
        args = ArgsObject(args)

    # 构建视觉backbone
    vision_backbone = build_backbone(args)

    # 构建触觉编码器
    tactile_encoder = TactileEncoder(
        backbone=_cfg(args, 'tactile_backbone', 'resnet34'),
        latent_dim=_cfg(args, 'tactile_latent_dim', 512),
        pretrained=_cfg(args, 'pretrained_backbones', True),
        freeze_backbone=False
    )

    # 构建双流Transformer
    dual_stream_transformer = DualStreamTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=_cfg(args, 'enc_layers', 4),
        num_decoder_layers=_cfg(args, 'dec_layers', 6),
        dim_feedforward=_cfg(args, 'dim_feedforward', 2048),
        dropout=_cfg(args, 'dropout', 0.1),
        activation='relu',
        normalize_before=_cfg(args, 'pre_norm', False),
        shared_encoder=_cfg(args, 'shared_encoder', True),
        shared_decoder=_cfg(args, 'shared_decoder', False),
        enable_cross_stream=_cfg(args, 'enable_cross_stream', False),
        cross_stream_layers=_cfg(args, 'cross_stream_layers', []),
    )

    # 构建双流VTLA模型
    model = DualStreamVTLAModel(
        vision_backbone=vision_backbone,
        tactile_encoder=tactile_encoder,
        dual_stream_transformer=dual_stream_transformer,
        state_dim=args.state_dim,
        num_queries=args.chunk_size,
        camera_names=args.camera_names,
        tactile_names=args.tactile_names,
        hidden_dim=args.hidden_dim,
        fusion_type=_cfg(args, 'fusion_type', 'gated'),
        use_contact_routing=_cfg(args, 'use_contact_routing', False),
        use_cvae=_cfg(args, 'use_cvae', True),
        latent_dim=_cfg(args, 'latent_dim', 32),
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Dual-Stream VTLA Model - Total parameters: {n_parameters / 1e6:.2f}M")

    return model


def _cfg(args, key, default):
    """从args中获取配置值"""
    if isinstance(args, dict):
        return args.get(key, default)
    else:
        return getattr(args, key, default)


if __name__ == '__main__':
    print("Testing Dual-Stream VTLA Model...")

    # 模拟配置
    class Args:
        state_dim = 14
        chunk_size = 10
        camera_names = ['cam_high', 'cam_left']
        tactile_names = ['tac_left', 'tac_right']
        hidden_dim = 512
        nheads = 8
        dim_feedforward = 2048
        dropout = 0.1
        enc_layers = 4
        dec_layers = 6
        pre_norm = False
        tactile_backbone = 'resnet34'
        tactile_latent_dim = 512

        # 双流特定配置
        shared_encoder = True
        shared_decoder = False
        enable_cross_stream = False
        cross_stream_layers = []
        fusion_type = 'gated'
        use_contact_routing = False
        use_cvae = True
        latent_dim = 32

        # Backbone相关
        lr_backbone = 1e-5
        masks = False
        dilation = False
        position_embedding = 'sine'
        backbone = 'resnet18'

    args = Args()

    try:
        model = build_dual_stream_vtla_model(args)
        print("✓ Model built successfully")

        # 测试前向传播
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)

        B = 2
        qpos = torch.randn(B, args.state_dim).to(device)
        cam_image = torch.randn(B, len(args.camera_names), 3, 256, 256).to(device)
        tac_image = torch.randn(B, len(args.tactile_names), 3, 256, 256).to(device)

        # 测试推理
        with torch.no_grad():
            actions_pred, is_pad_pred = model(qpos, cam_image, tac_image)
            print(f"✓ Inference successful")
            print(f"  Actions shape: {actions_pred.shape}")
            print(f"  Is_pad shape: {is_pad_pred.shape}")

        # 测试训练
        actions_gt = torch.randn(B, args.chunk_size, args.state_dim).to(device)
        is_pad_gt = torch.zeros(B, args.chunk_size, dtype=torch.bool).to(device)

        output = model(qpos, cam_image, tac_image, actions_gt, is_pad_gt, return_components=True)
        actions_pred, is_pad_pred, (mu, logvar), components = output
        print(f"✓ Training forward successful")
        print(f"  Latent mu shape: {mu.shape}")
        print(f"  Vision decoder output: {components['vision_decoder_output'].shape}")
        print(f"  Tactile decoder output: {components['tactile_decoder_output'].shape}")
        if 'fusion_weights' in components and components['fusion_weights'] is not None:
            print(f"  Fusion weights shape: {components['fusion_weights'].shape}")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
