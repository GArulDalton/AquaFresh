"""AquaFresh: freshness- and energy-aware hybrid acoustic-optical data harvesting by cooperative AUVs."""
import os as _os, sys as _sys
_HERE=_os.path.dirname(_os.path.abspath(__file__))
if _HERE not in _sys.path: _sys.path.insert(0,_HERE)
from paths import ROOT, MODELS, RESULTS, FIGURES  # noqa: E402
__version__="1.0.0"
