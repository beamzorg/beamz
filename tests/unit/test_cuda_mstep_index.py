"""Independent exact-division oracle for compact CUDA queue coordinates."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_bounded_reciprocal_division_matches_native_integer_division(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("A C++17 compiler is required for the portable index helper")
    root = Path(__file__).resolve().parents[2]
    source = tmp_path / "index_oracle.cc"
    source.write_text(
        r"""
#include "mstep_index.h"
#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>
int main() {
  const uint32_t maximum = std::numeric_limits<int32_t>::max();
  const std::array<uint32_t, 19> divisors = {
      1, 2, 3, 7, 8, 12, 16, 31, 32, 33, 127, 128, 129,
      136, 272, 384, 512, 65537, maximum};
  auto check = [&](uint32_t value, uint32_t divisor) {
    const uint32_t actual = MstepQuotient(value, MakeMstepDivisor(divisor));
    if (actual != value / divisor) {
      std::fprintf(stderr, "%u/%u: got %u, expected %u\n", value,
                   divisor, actual, value / divisor);
      return false;
    }
    return true;
  };
  // Exhaust a useful small domain, including exact multiples and neighbours.
  for (uint32_t divisor = 1; divisor < 256; ++divisor)
    for (uint32_t value = 0; value < 16384; ++value)
      if (!check(value, divisor)) return 1;
  // Probe large numerator/divisor extremes and quotient boundaries.
  for (uint32_t divisor : divisors) {
    for (uint32_t delta = 0; delta < 1024; ++delta)
      if (!check(maximum - delta, divisor)) return 1;
    for (uint64_t quotient = 0; quotient * divisor <= maximum; quotient = quotient * 2 + 1) {
      const uint32_t exact = quotient * divisor;
      if (!check(exact, divisor)) return 1;
      if (exact && !check(exact - 1, divisor)) return 1;
      if (exact < maximum && !check(exact + 1, divisor)) return 1;
    }
  }
  uint32_t random = 0xa39d317u;
  for (int iteration = 0; iteration < 2000000; ++iteration) {
    random = random * 1664525u + 1013904223u;
    const uint32_t value = random & maximum;
    random = random * 1664525u + 1013904223u;
    const uint32_t divisor = (random & maximum) + 1u;
    if (!check(value, divisor)) return 1;
  }
}
"""
    )
    executable = tmp_path / "index_oracle"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-I",
            str(root / "cuda/src"),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    subprocess.run(
        [str(executable)], check=True, capture_output=True, text=True, timeout=30
    )
