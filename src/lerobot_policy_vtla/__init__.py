"""VTLA policy plugin for LeRobot."""

try:
    import lerobot as _lerobot  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised only in a broken environment
    raise ImportError(
        "LeRobot is required by lerobot_policy_vtla. Install the pinned third-party "
        "checkout with `pip install -e third_party/lerobot[training]`."
    ) from exc

from .configuration_vtla import VTLAConfig
from .modeling_vtla import VTLAPolicy
from .processor_vtla import make_vtla_pre_post_processors
from .scheduling_vtla import WarmupStableCosineDecaySchedulerConfig

__all__ = [
    "VTLAConfig",
    "VTLAPolicy",
    "WarmupStableCosineDecaySchedulerConfig",
    "make_vtla_pre_post_processors",
]
