"""
VTLA (Vision-Tactile-Language-Action) Model Package
"""

from .tactile_encoder import TactileEncoder, TactileEncoderWithRefine
from .cross_modal_fusion import CrossModalFusion, BiDirectionalCrossAttention
from .vtla_policy import VTLAPolicy, VTLAModel
from .action_heads import MainActionHead, TactileRefineHead, ContactDetector

__all__ = [
    'TactileEncoder',
    'TactileEncoderWithRefine',
    'CrossModalFusion',
    'BiDirectionalCrossAttention',
    'VTLAPolicy',
    'VTLAModel',
    'MainActionHead',
    'TactileRefineHead',
    'ContactDetector',
]
