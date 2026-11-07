#!/usr/bin/env bash
# Keep the registered SUNDIALS 6 ABI; Homebrew's SUNDIALS 7 is incompatible.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null
command -v cmake >/dev/null
brew list --versions suite-sparse >/dev/null || brew install suite-sparse
uv sync --locked
mkdir -p results/scratch
archive=results/scratch/sundials-6.4.1.tar.gz
if [[ ! -f "$archive" ]]; then
  curl -fL --retry 2 --connect-timeout 20 -o "$archive" \
    https://github.com/LLNL/sundials/releases/download/v6.4.1/sundials-6.4.1.tar.gz
fi
python=.venv/bin/python
"$python" - "$archive" <<'PY'
import hashlib
import sys
from pathlib import Path
path = Path(sys.argv[1])
expected = '7bf10a8d2920591af3fba2db92548e91ad60eb7241ab23350a9b1bc51e05e8d0'
if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
    raise SystemExit('SUNDIALS archive checksum mismatch')
PY
tar -xzf "$archive" -C results/scratch
suite_prefix=$(brew --prefix suite-sparse)
sdk_path=$(xcrun --show-sdk-path)
# A selected SDK may be newer than the installed Apple linker understands.
# Honor SDKROOT; use the installed macOS 26 SDK on this registered machine.
if [[ -z "${SDKROOT:-}" && -d /Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk ]]; then
  sdk_path=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk
fi
export SDKROOT="${SDKROOT:-$sdk_path}"
cmake -S results/scratch/sundials-6.4.1 -B results/scratch/sundials-build \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_SYSROOT="${SDKROOT:-$sdk_path}" \
  -DCMAKE_INSTALL_PREFIX="$PWD/.runtime/sundials-6.4.1" \
  -DBUILD_ARKODE=OFF -DBUILD_CVODES=OFF -DBUILD_IDA=OFF -DBUILD_IDAS=OFF \
  -DBUILD_KINSOL=OFF -DEXAMPLES_ENABLE_C=OFF -DEXAMPLES_ENABLE_CXX=OFF \
  -DBUILD_FORTRAN_MODULE_INTERFACE=OFF -DSUNDIALS_INDEX_SIZE=64 \
  -DENABLE_KLU=ON -DKLU_INCLUDE_DIR="$suite_prefix/include/suitesparse" \
  -DKLU_LIBRARY_DIR="$suite_prefix/lib"
cmake --build results/scratch/sundials-build --parallel 6
cmake --install results/scratch/sundials-build
bash scripts/run-macos.sh -m metabolism.fetch_models
bash scripts/run-macos.sh -m unittest discover -s tests -v
