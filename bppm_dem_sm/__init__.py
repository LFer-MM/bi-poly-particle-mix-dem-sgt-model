"""bppm_dem_sm: bidisperse particle mixing DEM surrogate-model package.

All code lives in flat modules at the package root (config, data_io, training,
prediction, run_metrics, lacey_mixing_index, run_visualization, cell_grid,
animate_particles, csv_to_parquet, verify_particle_integrity, sim_functions,
simulation, pipeline).
"""

from .config import (
    MetricsOptions,
    PipelineConfig,
    PredictionOptions,
    TrainingOptions,
    VisualizationOptions,
)
from .pipeline import run_pipeline

__all__ = [
    "MetricsOptions",
    "PipelineConfig",
    "PredictionOptions",
    "TrainingOptions",
    "VisualizationOptions",
    "run_pipeline",
]
