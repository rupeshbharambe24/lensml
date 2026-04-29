"""lensml.core — instrumentation primitives.

Per plan §3.2:
    hooks.py     forward/grad hook installers
    telemetry.py metric extraction (loss, grad norms, activation stats)
    transport.py ZMQ publisher
    signals.py   SIGUSR1 handler for pause/checkpoint
"""
