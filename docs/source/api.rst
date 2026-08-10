Python API reference
====================

Autodoc for the ``bppm_dem_sm`` package. This is the **callable Python
interface** (importable modules, classes, and functions). The CLI
``bppm-pipeline`` is a thin wrapper around the same code—especially
:class:`~bppm_dem_sm.config.PipelineConfig` and
:func:`~bppm_dem_sm.pipeline.run_pipeline`.

Configuration and CLI
---------------------

.. automodule:: bppm_dem_sm.config
   :members:
   :show-inheritance:

.. automodule:: bppm_dem_sm.cli
   :members:
   :show-inheritance:

Pipeline orchestration
----------------------

.. automodule:: bppm_dem_sm.pipeline
   :members:
   :show-inheritance:

.. automodule:: bppm_dem_sm.tf_quiet
   :members:
   :show-inheritance:

Data I/O and processing
-----------------------

.. automodule:: bppm_dem_sm.data_io
   :members:
   :show-inheritance:

.. automodule:: bppm_dem_sm.csv_to_parquet
   :members:
   :show-inheritance:

.. automodule:: bppm_dem_sm.verify_particle_integrity
   :members:
   :show-inheritance:

Model training and prediction
-----------------------------

.. automodule:: bppm_dem_sm.training
   :members:
   :show-inheritance:

.. automodule:: bppm_dem_sm.prediction
   :members:
   :show-inheritance:

Metrics
-------

.. automodule:: bppm_dem_sm.lacey_mixing_index
   :members:
   :show-inheritance:

.. automodule:: bppm_dem_sm.run_metrics
   :members:
   :show-inheritance:

Visualization
-------------

.. automodule:: bppm_dem_sm.animate_particles
   :members:
   :show-inheritance:

.. automodule:: bppm_dem_sm.cell_grid
   :members:
   :show-inheritance:

.. automodule:: bppm_dem_sm.run_visualization
   :members:
   :show-inheritance:

DEM simulation (YADE)
---------------------

.. automodule:: bppm_dem_sm.simulation
   :members:
   :show-inheritance:

.. automodule:: bppm_dem_sm.sim_functions
   :members:
   :show-inheritance:
