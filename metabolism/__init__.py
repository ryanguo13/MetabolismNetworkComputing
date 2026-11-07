"""Reaction-structured ODE experiments."""

import os
from pathlib import Path
import sys

# ctypes on macOS does not search the project's pinned SUNDIALS installation.
# Select its registered v6 ABI before any native binding is imported.
if sys.platform == 'darwin':
    _native_lib = Path(__file__).resolve().parents[1] / '.runtime/sundials-6.4.1/lib'
    if (_native_lib / 'libsundials_cvode.dylib').is_file():
        _search = os.environ.get('DYLD_LIBRARY_PATH', '').split(os.pathsep)
        os.environ['DYLD_LIBRARY_PATH'] = os.pathsep.join(dict.fromkeys([str(_native_lib), *filter(None, _search)]))
