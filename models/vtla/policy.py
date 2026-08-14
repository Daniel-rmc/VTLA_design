"""
VTLA Policy - 完整的视触觉语言动作模型
整合所有模块：触觉编码器、视觉backbone、交叉注意力融合、VLA主干、双路动作头
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
import sys
from pathlib import Path

# 导入自定义模块
from ..shared import TactileEncoder
from .cross_modal_fusion import BiDirectionalCrossAttention
from .action_heads import DualPathActionHead

# 添加同一 workspace 下的 UniVTAC 路径。环境变量仍可覆盖模块搜索，
# 但独立评测脚本不应依赖训练启动器注入 PYTHONPATH。
univtac_base = Path(__file__).resolve().parents[3] / 'UniVTAC'
if univtac_base.is_dir():
    # Keep the local VTLA ``models`` package ahead of UniVTAC.  Prepending
    # UniVTAC's DETR directory makes multiprocessing workers resolve
    # ``models`` to the wrong package when they re-import this module.
    univtac_path = str(univtac_base)
    if univtac_path not in sys.path:
        sys.path.append(univtac_path)

try:
    from policy.ACT.detr.models.backbone import build_backbone
    from policy.ACT.detr.models.transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer
    UNIVTAC_AVAILABLE = True
except ImportError:
    try:
        from models.backbone import build_backbone
        from models.transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer
        UNIVTAC_AVAILABLE = True
    except ImportError:
        print("Warning: Could not import UniVTAC modules. Make sure UniVTAC is in the path.")
        UNIVTAC_AVAILABLE = False
        # 定义占位符以避免NameError
        def build_backbone(args):
            raise ImportError("UniVTAC modules not available")
        def build_transformer(args):
            raise ImportError("UniVTAC modules not available")
        TransformerEncoder = None
        TransformerEncoderLayer = None


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
    """Return a deterministic [1, D, H, W] 2-D positional encoding."""
    if d_model % 4 != 0:
        raise ValueError(f"hidden_dim must be divisible by 4, got {d_model}")
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


class VTLAModel(nn.Module):
    """
    VTLA主模型：Vision-Tactile-Language-Action

    架构流程：
    1. Vision Backbone提取视觉tokens
    2. Tactile Encoder提取触觉tokens
    3. Cross-Modal Fusion进行视触觉交叉注意力融合
    4. Transformer Encoder编码融合特征
    5. CVAE Encoder（训练时）学习动作潜在分布
    6. Transformer Decoder生成动作序列query
    7. Dual-Path Action Head生成最终动作（主路+触觉微调）
    """

    def __init__(
        self,
        vision_backbone,
        tactile_encoder: TactileEncoder,
        transformer,
        encoder,
        state_dim: int,
        num_queries: int,
        camera_names: list,
        tactile_names: list,
        hidden_dim: int = 512,
        nheads: int = 8,
        cross_attn_layers: int = 2,
        use_tactile_refine: bool = True,
        refine_scale: float = 0.1,
        tactile_position_embedding: str = 'sine',
    ):
        super().__init__()

        self.num_queries = num_queries
        self.camera_names = camera_names
        self.tactile_names = tactile_names
        self.hidden_dim = hidden_dim
        self.use_tactile_refine = use_tactile_refine
        self.tactile_position_embedding = tactile_position_embedding

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
        self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
        self.vision_source_embed = nn.Embedding(len(camera_names), hidden_dim)
        self.tactile_source_embed = nn.Embedding(len(tactile_names), hidden_dim)
        self.tactile_position_embed = (
            nn.Embedding(1024, hidden_dim)
            if tactile_position_embedding == 'learned' else None
        )
        if tactile_position_embedding not in {'sine', 'learned'}:
            raise ValueError(
                f"Unsupported tactile position embedding: {tactile_position_embedding}"
            )

        # 3. 视触觉交叉注意力融合
        self.cross_modal_fusion = BiDirectionalCrossAttention(
            d_model=hidden_dim,
            nhead=nheads,
            num_layers=cross_attn_layers,
            dropout=0.1
        )

        # 4. VLA主干
        self.transformer = transformer
        self.encoder = encoder
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # 5. CVAE编码器（用于训练）
        self.latent_dim = 32
        self.cls_embed = nn.Embedding(1, hidden_dim)
        self.encoder_action_proj = nn.Linear(state_dim, hidden_dim)
        self.encoder_joint_proj = nn.Linear(state_dim, hidden_dim)
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim * 2)
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)

        # 位置编码
        self.register_buffer(
            'pos_table',
            get_sinusoid_encoding_table(1 + 1 + num_queries, hidden_dim)
        )
        self.additional_pos_embed = nn.Embedding(2, hidden_dim)

        # 6. 双路动作生成头
        self.action_head = DualPathActionHead(
            hidden_dim=hidden_dim,
            tactile_dim=hidden_dim,
            action_dim=state_dim,
            refine_scale=refine_scale,
            adaptive_scale=True
        )

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

        Returns:
            训练时: (actions_pred, is_pad_pred, (mu, logvar), components)
            推理时: (actions_pred, is_pad_pred)
        """
        is_training = actions is not None
        bs = qpos.shape[0]

        # ===== 1. CVAE编码（训练时） =====
        if is_training:
            action_embed = self.encoder_action_proj(actions)
            qpos_embed = self.encoder_joint_proj(qpos).unsqueeze(1)
            cls_embed = self.cls_embed.weight.unsqueeze(0).repeat(bs, 1, 1)

            encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], dim=1)
            encoder_input = encoder_input.permute(1, 0, 2)

            cls_joint_is_pad = torch.full((bs, 2), False, device=qpos.device)
            is_pad_full = torch.cat([cls_joint_is_pad, is_pad], dim=1)

            pos_embed = self.pos_table.clone().detach().permute(1, 0, 2)

            encoder_output = self.encoder(
                encoder_input,
                pos=pos_embed,
                src_key_padding_mask=is_pad_full
            )
            encoder_output = encoder_output[0]  # CLS token output

            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = mu if deterministic_latent else reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros(bs, self.latent_dim, device=qpos.device)
            latent_input = self.latent_out_proj(latent_sample)

        # ===== 2. 视觉特征提取 =====
        all_vision_features = []

        for cam_id in range(len(self.camera_names)):
            features, pos = self.vision_backbone(cam_image[:, cam_id])
            features = features[0]  # 最后一层特征
            pos = pos[0]
            projected = self.vision_input_proj(features)
            if pos.shape[1] != self.hidden_dim:
                raise ValueError(
                    f"Vision position embedding has {pos.shape[1]} channels; "
                    f"expected hidden_dim={self.hidden_dim}"
                )
            source = self.vision_source_embed.weight[cam_id].view(1, -1, 1, 1)
            all_vision_features.append(projected + pos + source)

        # 拼接所有相机的视觉特征
        if len(all_vision_features) > 0:
            vision_features = torch.cat(
                [f.flatten(2) for f in all_vision_features], dim=2
            )  # [B, D, N_v]
            vision_tokens = vision_features.permute(0, 2, 1)  # [B, N_v, D]
        else:
            vision_tokens = None

        # ===== 3. 触觉特征提取 =====
        all_tactile_features = []

        for tac_id in range(len(self.tactile_names)):
            tac_feat = self.tactile_encoder(
                tac_image[:, tac_id], return_tokens=True
            )  # [B, D, H', W']
            tac_feat = self.tactile_input_proj(tac_feat)
            if self.tactile_position_embed is not None:
                tactile_token_count = tac_feat.shape[-2] * tac_feat.shape[-1]
                if tactile_token_count > self.tactile_position_embed.num_embeddings:
                    raise ValueError(
                        f"Tactile feature map has {tactile_token_count} tokens, exceeding "
                        f"the learned-position capacity of "
                        f"{self.tactile_position_embed.num_embeddings}"
                    )
                tactile_pos = self.tactile_position_embed.weight[:tactile_token_count]
                tactile_pos = tactile_pos.transpose(0, 1).reshape(
                    1, self.hidden_dim, tac_feat.shape[-2], tac_feat.shape[-1]
                )
            else:
                tactile_pos = get_2d_sinusoid_encoding(
                    tac_feat.shape[-2], tac_feat.shape[-1], self.hidden_dim,
                    tac_feat.device, tac_feat.dtype,
                )
            source = self.tactile_source_embed.weight[tac_id].view(1, -1, 1, 1)
            all_tactile_features.append(tac_feat + tactile_pos + source)

        # 拼接所有触觉传感器的特征
        if len(all_tactile_features) > 0:
            tactile_features = torch.cat(
                [f.flatten(2) for f in all_tactile_features], dim=2
            )  # [B, D, N_t]
            tactile_tokens = tactile_features.permute(0, 2, 1)  # [B, N_t, D]
        else:
            tactile_tokens = None

        # ===== 4. 视触觉交叉注意力融合 =====
        if vision_tokens is not None and tactile_tokens is not None:
            fused_vision, fused_tactile = self.cross_modal_fusion(
                vision_tokens, tactile_tokens
            )
            # 合并融合后的特征
            fused_tokens = torch.cat([fused_vision, fused_tactile], dim=1)  # [B, N_v+N_t, D]
        elif vision_tokens is not None:
            fused_tokens = vision_tokens
            fused_tactile = None
        elif tactile_tokens is not None:
            fused_tokens = tactile_tokens
            fused_tactile = tactile_tokens
        else:
            raise ValueError("At least one modality (vision or tactile) must be provided")

        # ===== 5. 准备transformer输入 =====
        # 将tokens转换为transformer格式
        src = fused_tokens.permute(1, 0, 2)  # [N, B, D]

        # 位置编码
        N_tokens = src.shape[0]
        pos_embed = torch.zeros(
            N_tokens, bs, self.hidden_dim, device=src.device, dtype=src.dtype
        )

        # 添加本体感觉和latent
        proprio_input = self.input_proj_robot_state(qpos)
        additional_pos_embed = self.additional_pos_embed.weight

        pos_embed_full = torch.cat(
            [additional_pos_embed.unsqueeze(1).repeat(1, bs, 1), pos_embed],
            dim=0
        )
        addition_input = torch.stack([latent_input, proprio_input], dim=0)
        src_full = torch.cat([addition_input, src], dim=0)

        # ===== 6. Transformer解码 =====
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        tgt = torch.zeros_like(query_embed)

        # Encoder-Decoder
        memory = self.transformer.encoder(
            src_full, src_key_padding_mask=None, pos=pos_embed_full
        )
        hs = self.transformer.decoder(
            tgt, memory,
            memory_key_padding_mask=None,
            pos=pos_embed_full,
            query_pos=query_embed
        )
        hs = hs.transpose(1, 2)  # [num_layers, B, T, D] -> [num_layers, T, B, D] -> process
        hs = hs[-1]  # 取最后一层 [B, T, D]

        # ===== 7. 双路动作生成 =====
        if self.use_tactile_refine and fused_tactile is not None:
            # 准备纯触觉特征（用于微调分支）
            # 对触觉tokens进行池化以匹配query数量
            B, N_t, D = fused_tactile.shape
            if N_t != self.num_queries:
                # 使用自适应池化
                tactile_for_refine = fused_tactile.permute(0, 2, 1)  # [B, D, N_t]
                tactile_for_refine = nn.functional.adaptive_avg_pool1d(
                    tactile_for_refine, self.num_queries
                )
                tactile_for_refine = tactile_for_refine.permute(0, 2, 1)  # [B, T, D]
            else:
                tactile_for_refine = fused_tactile

            # 双路动作头
            if return_components:
                actions_pred, is_pad_pred, components = self.action_head(
                    hs, tactile_for_refine, return_components=True
                )
            else:
                actions_pred, is_pad_pred = self.action_head(
                    hs, tactile_for_refine, return_components=False
                )
                components = None
        else:
            # 仅使用主动作头（不使用触觉微调）
            actions_pred, is_pad_pred = self.action_head.main_head(hs)
            components = None

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


class VTLAPolicy(nn.Module):
    """
    VTLA策略包装器：负责损失计算、分阶段参数冻结、训练/推理接口
    """

    def __init__(self, args_override, stage: str = 'full'):
        super().__init__()
        self.model = build_vtla_model(args_override)
        self.kl_weight = _cfg(args_override, 'kl_weight', 10.0)
        self.refine_weight = _cfg(args_override, 'refine_weight', 0.5)
        self.contact_weight = _cfg(args_override, 'contact_weight', 0.1)
        self.pad_weight = _cfg(args_override, 'pad_weight', 1.0)
        self.l1_reduction = _cfg(args_override, 'l1_reduction', 'valid_mean')
        self.freeze_tactile_batchnorm = _cfg(
            args_override, 'freeze_tactile_batchnorm', False
        )
        self.set_stage(stage)
        if self.freeze_tactile_batchnorm:
            self._freeze_tactile_batchnorm()

    def set_stage(self, stage: str):
        """
        设置训练阶段并冻结/解冻对应参数

        stage2: 训练主干（视觉+触觉编码器+融合+VLA），关闭触觉微调分支
        stage3: 冻结主干，仅训练触觉微调分支（refine head / contact / scale）
        full:   全部参数可训练
        """
        self.stage = stage
        head = self.model.action_head

        if stage == 'stage2':
            for p in self.model.parameters():
                p.requires_grad = True
            for p in head.refine_head.parameters():
                p.requires_grad = False
            for p in head.contact_detector.parameters():
                p.requires_grad = False
            if head.adaptive_scale:
                for p in head.scale_predictor.parameters():
                    p.requires_grad = False
            self.model.use_tactile_refine = False
        elif stage == 'stage3':
            for p in self.model.parameters():
                p.requires_grad = False
            trainable = [head.refine_head, head.contact_detector]
            if head.adaptive_scale:
                trainable.append(head.scale_predictor)
            for module in trainable:
                for p in module.parameters():
                    p.requires_grad = True
            self.model.use_tactile_refine = True
        else:  # 'full'
            for p in self.model.parameters():
                p.requires_grad = True
            self.model.use_tactile_refine = True

    def train(self, mode: bool = True):
        """Keep the frozen Stage-3 backbone, BatchNorm and dropout in eval mode."""
        super().train(mode)
        if mode and self.stage == 'stage3':
            self.model.eval()
            head = self.model.action_head
            head.refine_head.train()
            head.contact_detector.train()
            if head.adaptive_scale:
                head.scale_predictor.train()
        if mode and self.freeze_tactile_batchnorm:
            for module in self.model.tactile_encoder.backbone.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def _freeze_tactile_batchnorm(self):
        """Match the fixed normalization used by UniVTAC's tactile ResNet."""
        for module in self.model.tactile_encoder.backbone.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
                for parameter in module.parameters():
                    parameter.requires_grad = False

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
                deterministic_latent=(deterministic_latent or self.stage == 'stage3'),
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
            # Matches UniVTAC ACT: padded elements contribute zero to the
            # numerator but remain in the tensor-wide mean denominator.
            l1 = (all_l1 * valid).mean()
        elif self.l1_reduction == 'valid_mean':
            l1 = (all_l1 * valid).sum() / valid.sum().clamp_min(1)
        else:
            raise ValueError(f"Unsupported l1_reduction: {self.l1_reduction}")
        loss_dict['l1'] = l1

        pad_loss = F.binary_cross_entropy_with_logits(
            is_pad_hat.squeeze(-1), is_pad.float()
        )
        loss_dict['pad'] = pad_loss

        # KL散度
        total_kld = self._kl_divergence(mu, logvar)
        loss_dict['kl'] = total_kld

        # 触觉微调损失（仅在stage3或full）
        if self.stage in ['stage3', 'full'] and components is not None:
            # 使用触觉残差改进的动作与GT的损失
            # a_hat已经包含了残差，这是最终动作
            # 我们可以额外监督主动作和残差的分解
            if 'main_actions' in components:
                main_actions = components['main_actions']
                # 主动作也应该接近GT
                main_l1 = F.l1_loss(main_actions, actions, reduction='none')
                main_l1 = (main_l1 * valid).sum() / valid.sum().clamp_min(1)
                loss_dict['main_l1'] = main_l1

            # 数据集中没有可信接触标签。检测器仍可通过动作残差路径学习，
            # 但不能用“轨迹后半段=接触”这种伪标签进行监督。
            loss_dict['contact'] = torch.tensor(0.0, device=l1.device)
        else:
            loss_dict['main_l1'] = torch.tensor(0.0, device=l1.device)
            loss_dict['contact'] = torch.tensor(0.0, device=l1.device)

        # 总损失
        loss_dict['loss'] = (
            loss_dict['l1'] +
            (0.0 if self.stage == 'stage3' else self.kl_weight * loss_dict['kl']) +
            (0.0 if self.stage == 'stage3' else self.pad_weight * loss_dict['pad']) +
            self.contact_weight * loss_dict['contact']
        )

        return loss_dict

    @staticmethod
    def _kl_divergence(mu, logvar):
        """计算KL散度"""
        batch_size = mu.size(0)
        if mu.ndimension() == 4:
            mu = mu.view(mu.size(0), mu.size(1))
        if logvar.ndimension() == 4:
            logvar = logvar.view(logvar.size(0), logvar.size(1))

        klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        total_kld = klds.sum(1).mean(0, True)
        return total_kld[0]


def build_vtla_model(args):
    """
    构建VTLA模型

    Args:
        args: 配置参数，包含：
            - state_dim: 状态维度
            - chunk_size: 动作序列长度
            - camera_names: 相机列表
            - tactile_names: 触觉传感器列表
            - hidden_dim: 隐藏层维度
            - tactile_backbone: 触觉编码器backbone
            - tactile_latent_dim: 触觉编码器输出维度
            - cross_attn_layers: 交叉注意力层数
            - use_tactile_refine: 是否使用触觉微调分支
            - refine_scale: 触觉残差缩放系数
    """
    # 构建视觉backbone
    vision_backbone = build_backbone(args)

    # 构建触觉编码器
    tactile_encoder = TactileEncoder(
        backbone=_cfg(args, 'tactile_backbone', 'resnet34'),
        latent_dim=_cfg(args, 'tactile_latent_dim', 512),
        pretrained=_cfg(args, 'pretrained_backbones', True),
        freeze_backbone=False
    )

    # 构建transformer
    transformer = build_transformer(args)

    # 构建CVAE encoder
    encoder_layer = TransformerEncoderLayer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation='relu',
        normalize_before=args.pre_norm
    )
    encoder_norm = nn.LayerNorm(args.hidden_dim) if args.pre_norm else None
    encoder = TransformerEncoder(encoder_layer, args.enc_layers, encoder_norm)

    # 构建VTLA模型
    model = VTLAModel(
        vision_backbone=vision_backbone,
        tactile_encoder=tactile_encoder,
        transformer=transformer,
        encoder=encoder,
        state_dim=args.state_dim,
        num_queries=args.chunk_size,
        camera_names=args.camera_names,
        tactile_names=args.tactile_names,
        hidden_dim=args.hidden_dim,
        nheads=args.nheads,
        cross_attn_layers=_cfg(args, 'cross_attn_layers', 2),
        use_tactile_refine=_cfg(args, 'use_tactile_refine', True),
        refine_scale=_cfg(args, 'refine_scale', 0.1),
        tactile_position_embedding=_cfg(args, 'tactile_position_embedding', 'sine'),
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"VTLA Model - Total parameters: {n_parameters / 1e6:.2f}M")

    return model


def _cfg(args, key, default):
    """从args中获取配置值（支持dict和object两种形式）"""
    if isinstance(args, dict):
        return args.get(key, default)
    else:
        return getattr(args, key, default)


if __name__ == '__main__':
    # 测试代码
    print("Testing VTLA Model...")

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
        cross_attn_layers = 2
        use_tactile_refine = True
        refine_scale = 0.1

        # Backbone相关（需要与UniVTAC兼容）
        lr_backbone = 1e-5
        masks = False
        dilation = False
        position_embedding = 'sine'
        backbone = 'resnet18'

    args = Args()

    try:
        model = build_vtla_model(args)
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
        print(f"  Contact prob mean: {components['contact_prob'].mean():.3f}")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
