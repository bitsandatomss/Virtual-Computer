"""VirtualComputer monorepo facade."""

from .computer import LayerResult, VirtualComputer, layer_available
from .levels import certify_unified, summarize_pipeline_report
from .paths import REPO_ROOT, ensure_layer_paths
from .vertical import run_vertical_slice

__all__ = ["LayerResult", "REPO_ROOT", "VirtualComputer",
           "certify_unified", "ensure_layer_paths", "layer_available",
           "run_vertical_slice", "summarize_pipeline_report"]
