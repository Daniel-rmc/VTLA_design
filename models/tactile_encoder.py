"""
Tactile Encoder Module
独立的触觉编码器，不与视觉共享参数
"""
import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, Sequence


class TactileEncoder(nn.Module):
    """
    独立的触觉图像编码器

    特点：
    1. 专门处理触觉图像（GelSight类型的视触觉传感器）
    2. 支持自监督预训练（marker重建、深度重建等）
    3. 输出高维触觉特征tokens
    """

    def __init__(
        self,
        backbone: str = 'resnet34',
        latent_dim: int = 512,
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        """
        Args:
            backbone: 骨干网络类型 ('resnet18', 'resnet34', 'resnet50')
            latent_dim: 输出特征维度
            pretrained: 是否使用ImageNet预训练权重
            freeze_backbone: 是否冻结骨干网络
        """
        super().__init__()

        self.latent_dim = latent_dim

        # 选择骨干网络
        if backbone == 'resnet18':
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            self.feature_dim = 512
        elif backbone == 'resnet34':
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet34(weights=weights)
            self.feature_dim = 512
        elif backbone == 'resnet50':
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # 移除最后的全连接层，保留特征提取部分
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        # 冻结骨干网络（如果需要）
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 特征投影层：将CNN特征投影到latent_dim
        self.feature_proj = nn.Sequential(
            nn.Conv2d(self.feature_dim, latent_dim, kernel_size=1),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True)
        )

        # 全局特征提取（可选）
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor, return_tokens: bool = True):
        """
        Args:
            x: 触觉图像 [B, C, H, W]
            return_tokens: 是否返回空间token序列

        Returns:
            如果return_tokens=True:
                tokens [B, latent_dim, H', W']
            否则:
                global_feature [B, latent_dim]
        """
        # 提取CNN特征
        features = self.backbone(x)  # [B, feature_dim, H', W']

        # 投影到目标维度
        features = self.feature_proj(features)  # [B, latent_dim, H', W']

        if return_tokens:
            return features
        else:
            # 返回全局特征
            global_feat = self.global_pool(features).flatten(1)
            return global_feat

    def get_tokens(self, x: torch.Tensor):
        """
        获取触觉token序列（用于交叉注意力）

        Returns:
            tokens: [B, N, D] 其中N=H'*W', D=latent_dim
        """
        features = self.forward(x, return_tokens=True)  # [B, D, H', W']
        B, D, H, W = features.shape
        tokens = features.flatten(2).permute(0, 2, 1)  # [B, H'*W', D]
        return tokens


class TactileEncoderWithRefine(nn.Module):
    """
    带自监督重建头的触觉编码器（用于预训练）
    """

    def __init__(
        self,
        backbone: str = 'resnet34',
        latent_dim: int = 512,
        supervise: Optional[Sequence[str]] = None,
        marker_nums: int = 63,
        pretrained: bool = True
    ):
        super().__init__()

        self.encoder = TactileEncoder(
            backbone=backbone,
            latent_dim=latent_dim,
            pretrained=pretrained,
            freeze_backbone=False
        )

        self.supervise = list(supervise or ['marker', 'rgb'])
        self.decoders = nn.ModuleDict()

        # Marker重建头
        if 'marker' in self.supervise:
            self.decoders['marker'] = MarkerDecoder(
                latent_dim=latent_dim,
                marker_nums=marker_nums
            )

        # RGB重建头
        if 'rgb' in self.supervise:
            self.decoders['rgb'] = RGBDecoder(
                latent_dim=latent_dim,
                output_channels=3
            )

        # 深度重建头
        if 'depth' in self.supervise:
            self.decoders['depth'] = RGBDecoder(
                latent_dim=latent_dim,
                output_channels=1
            )

        # Pose回归头
        if 'pose' in self.supervise:
            self.decoders['pose'] = PoseDecoder(
                latent_dim=latent_dim,
                pose_dims=7
            )

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[dict] = None,
        weights: Optional[dict] = None,
    ):
        """Encode an image, or compute pretraining losses through DDP."""
        if targets is not None:
            return self.compute_loss(x, targets, weights)
        return self.encoder(x, return_tokens=False)

    def reconstruct(self, x: torch.Tensor):
        """重建所有监督信号"""
        latent = self.forward(x)
        outputs = {}
        for key in self.supervise:
            outputs[key] = self.decoders[key](latent)
        return outputs

    def compute_loss(self, x: torch.Tensor, targets: dict, weights: Optional[dict] = None):
        """计算重建损失"""
        outputs = self.reconstruct(x)

        if weights is None:
            weights = {key: 1.0 for key in self.supervise}

        total_loss = 0.0
        loss_dict = {}

        for key in self.supervise:
            if key not in targets:
                raise KeyError(f"Missing tactile pretraining target: {key}")
            if key in ['rgb', 'depth']:
                criterion = nn.MSELoss()
                # 调整输出尺寸以匹配目标
                if outputs[key].shape != targets[key].shape:
                    resized = nn.functional.interpolate(
                        outputs[key],
                        size=targets[key].shape[2:],
                        mode='bilinear',
                        align_corners=False
                    )
                    loss = criterion(resized, targets[key])
                else:
                    loss = criterion(outputs[key], targets[key])
            elif key == 'marker':
                criterion = nn.MSELoss()
                loss = criterion(outputs[key], targets[key])
            elif key == 'pose':
                criterion = nn.MSELoss()
                loss = criterion(outputs[key], targets[key])
            else:
                continue

            weighted_loss = weights.get(key, 1.0) * loss
            total_loss += weighted_loss
            loss_dict[key] = loss.item()

        loss_dict['total'] = total_loss.item()
        return total_loss, loss_dict


# 重建解码器（复用UniVTAC的实现）
class RGBDecoder(nn.Module):
    """RGB/Depth图像重建解码器"""

    def __init__(self, latent_dim: int = 512, output_channels: int = 3):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 8 * 8)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 8->16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),  # 16->32
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),  # 32->64
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 64->128
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 128->256
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, output_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        x = self.fc(x)
        x = x.view(-1, 256, 8, 8)
        x = self.deconv(x)
        return x


class MarkerDecoder(nn.Module):
    """Marker位置回归解码器"""

    def __init__(self, latent_dim: int = 512, marker_nums: int = 63):
        super().__init__()
        self.marker_nums = marker_nums
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.GELU(),
            nn.Linear(256, marker_nums * 2)
        )

    def forward(self, x: torch.Tensor):
        x = self.ffn(x)
        x = x.view(-1, self.marker_nums, 2)
        return x


class PoseDecoder(nn.Module):
    """Pose回归解码器"""

    def __init__(self, latent_dim: int = 512, pose_dims: int = 7):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.GELU(),
            nn.Linear(256, pose_dims)
        )

    def forward(self, x: torch.Tensor):
        return self.ffn(x)


if __name__ == '__main__':
    # 测试代码
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 测试基础编码器
    encoder = TactileEncoder(backbone='resnet34', latent_dim=512).to(device)
    dummy_input = torch.randn(2, 3, 256, 256).to(device)

    # 测试token输出
    tokens = encoder.get_tokens(dummy_input)
    print(f"Tactile tokens shape: {tokens.shape}")  # [2, 49, 512]

    # 测试带重建的编码器
    encoder_with_refine = TactileEncoderWithRefine(
        backbone='resnet34',
        latent_dim=512,
        supervise=['marker', 'rgb']
    ).to(device)

    outputs = encoder_with_refine.reconstruct(dummy_input)
    print("Reconstruction outputs:")
    for key, val in outputs.items():
        print(f"  {key}: {val.shape}")
