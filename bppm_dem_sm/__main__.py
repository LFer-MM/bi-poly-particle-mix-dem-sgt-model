"""Allow ``python -m bppm_dem_sm`` to run the pipeline CLI."""

from .tf_quiet import silence_tensorflow

silence_tensorflow()

from .cli import main

raise SystemExit(main())
