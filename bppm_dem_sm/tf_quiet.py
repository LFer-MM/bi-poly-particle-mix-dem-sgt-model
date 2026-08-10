"""Quiet TensorFlow / oneDNN startup noise before the libraries load."""

from __future__ import annotations

import os


def silence_tensorflow() -> None:
    """Suppress TensorFlow C++ INFO/WARNING logs and the oneDNN banner.

    Must run before TensorFlow/Keras is imported. Uses ``setdefault`` so an
    explicit user/env setting still wins.

    Sets:

    - ``TF_CPP_MIN_LOG_LEVEL=3``
    - ``TF_ENABLE_ONEDNN_OPTS=0``
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
