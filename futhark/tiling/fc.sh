#!/bin/sh
# Compile a Futhark program to a standalone executable: fc.sh BACKEND SRC OUT
# Uses FUTHARK, else the Futhark checkout next to this repository (as
# build.py does; FUTHARK_SRC overrides its location), built first.
set -e
if [ -z "$FUTHARK" ]; then
  checkout=${FUTHARK_SRC:-$(dirname "$0")/../../../futhark}
  [ -f "$HOME/.ghcup/env" ] && . "$HOME/.ghcup/env"
  (cd "$checkout" && cabal -v0 build exe:futhark)
  FUTHARK=$(cd "$checkout" && cabal -v0 list-bin exe:futhark)
fi
export CPATH=/opt/cuda/include${CPATH:+:$CPATH}
export LIBRARY_PATH=/opt/cuda/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}
export LD_LIBRARY_PATH=/opt/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
exec "$FUTHARK" "$1" "$2" -o "$3"
