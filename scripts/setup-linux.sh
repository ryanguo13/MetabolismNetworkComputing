#!/usr/bin/env bash
set -eu
# Run in Ubuntu 24.04 (WSL is supported), using sudo/root for apt only.
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  octave python3-numpy python3-scipy libsundials-cvode6 \
  libsundials-nvecserial6 libsundials-sunlinsol3 libsundials-sunmatrix4
# Recorded run: Octave 8.4.0, NumPy 1.26.4, SciPy 1.11.4, SUNDIALS 6.4.1.
# Inspect results/m1-completion/linux-environment.json for actual runtime.
