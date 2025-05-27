# Minimal spinup __init__.py for PyTorch exercises only
# This avoids importing TensorFlow and other heavy dependencies

# Only import what's needed for exercises
try:
    # Version
    from spinup.version import __version__
except ImportError:
    __version__ = "unknown"

# Don't import algorithms or loggers that have TF dependencies
# Users can import them directly if needed:
# from spinup.algos.pytorch.vpg.vpg import vpg as vpg_pytorch