#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
runtime="$PWD/.runtime/sundials-6.4.1/lib"
if [[ ! -f "$runtime/libsundials_cvode.dylib" ]]; then
  echo 'Missing project SUNDIALS 6 runtime. Run bash scripts/setup-macos.sh.' >&2
  exit 1
fi
export DYLD_LIBRARY_PATH="$runtime${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec .venv/bin/python "$@"
