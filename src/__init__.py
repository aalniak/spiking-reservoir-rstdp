"""rstpd_spiking_reservoir_timeseries.

A CPU/GPU research prototype for reservoir-based spiking time-series processing
and R-STDP (reward-modulated STDP / three-factor) local learning, designed with
future Lava / Loihi-2 deployment in mind.

NOTE: This package is a CPU/GPU prototype. It is *not* an actual Loihi-2
implementation unless explicitly run on the Lava Loihi-2 backend (see
``src.lava_export``).
"""

__version__ = "0.1.0"
