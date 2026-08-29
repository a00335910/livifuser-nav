"""Closed-loop evaluation executables for LiViFuser-Nav.

This package exists so that the closed-loop runner and supervisor entry points
live outside `livifuser_sim`, whose `setup.py` is byte-frozen by
`docs/experiments/PREREGISTRATION_RECOLLECTION_FREEZE_SIM_V3.json`. The node
implementations themselves stay in `livifuser_sim`; only the console_scripts
bindings live here.
"""
