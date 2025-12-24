"""Sequential editing module for analyzing continuous knowledge editing effects.

This module provides tools for running sequential knowledge editing experiments
using ROME and analyzing the resulting ripple effects, forgetting, and model
stability over time.
"""

from .config import SeqEditConfig, EditMethod
from .kg_utils import KG, Triple
from .runner import run_sequential_edits
from .analysis import (
    load_stats,
    load_ripple,
    load_acc,
    plot_time_series,
    plot_hop_degree_heatmaps,
    plot_failure_hist,
    run_all_plots,
)

__all__ = [
    # Configuration
    "SeqEditConfig",
    "EditMethod",
    # Data structures
    "KG",
    "Triple",
    # Main runner
    "run_sequential_edits",
    # Analysis functions
    "load_stats",
    "load_ripple",
    "load_acc",
    "plot_time_series",
    "plot_hop_degree_heatmaps",
    "plot_failure_hist",
    "run_all_plots",
]
