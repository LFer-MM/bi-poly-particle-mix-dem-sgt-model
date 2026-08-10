Bidisperse Polyhedrical Particle Mixing DEM Surrogate Model
===========================================================

Surrogate modeling for **bidisperse particle mixing** in a DEM (YADE) SAG-mill
slice: train a GRU on particle frame sequences, roll out predictions, score
mixing with the Lacey index, and render animations / cell-grid views.

You typically drive the stack with the ``bppm-pipeline`` CLI (or by importing
``run_pipeline`` / ``PipelineConfig`` in Python). The pages below document
that **Python library surface**—not an HTTP/REST service.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   api
