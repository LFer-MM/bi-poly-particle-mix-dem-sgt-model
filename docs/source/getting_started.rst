Getting started
===============

Install
-------

From the repository root (Python 3.12):

.. code-block:: bash

   pip install -e ".[docs]"
   # or, for runtime work:
   pip install -r requirements.txt
   pip install -e .

This registers the ``bppm-pipeline`` console script.

Run the pipeline (CLI)
----------------------

Preferred: pass a JSON file whose keys match :class:`~bppm_dem_sm.config.PipelineConfig`
fields. When ``--config`` is set, other pipeline flags are ignored.

.. code-block:: bash

   bppm-pipeline --config configs/pipeline_example.json

Override individual fields without JSON (Boolean flags use ``--flag`` /
``--no-flag``):

.. code-block:: bash

   bppm-pipeline --do-train --no-do-predict --epochs 10

See ``bppm-pipeline --help`` for the full flag list.

Example config
--------------

Minimal shape (paths and stage toggles). A fuller example lives at
``configs/pipeline_example.json`` in the repo:

.. code-block:: json

   {
     "data_dir": "data/processed/frames_parquet",
     "train_data_dir": "data/processed/train_parquet",
     "model_path": "models/rnn_gru_model.keras",
     "frames_in": 15,
     "do_train": false,
     "do_predict": true,
     "do_metrics": true,
     "do_visualization": true,
     "save_figures": true,
     "show_plots": false
   }

Stages gated by ``do_train``, ``do_predict``, ``do_metrics``, and
``do_visualization`` run in that order inside
:func:`~bppm_dem_sm.pipeline.run_pipeline`.

Use from Python
---------------

.. code-block:: python

   from bppm_dem_sm import PipelineConfig, run_pipeline

   cfg = PipelineConfig.from_json("configs/pipeline_example.json")
   results = run_pipeline(cfg)
   # or: run_pipeline(do_train=True, do_predict=False)

Build these docs locally
------------------------

.. code-block:: bash

   python -m sphinx -b html docs/source docs/build/html
