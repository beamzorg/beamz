#!/bin/sh
# Compile a Futhark program to a standalone executable: fc.sh BACKEND SRC OUT
export CPATH=/opt/cuda/include${CPATH:+:$CPATH}
export LIBRARY_PATH=/opt/cuda/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}
export LD_LIBRARY_PATH=/opt/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
exec futhark "$1" "$2" -o "$3"
