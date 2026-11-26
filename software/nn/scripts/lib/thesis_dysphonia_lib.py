"""The thesis's WORLD dysphonia C++ (VoiceDysphonia.cpp), compiled for Python through ctypes.

WHY ONE MODULE. Two scripts call the same C++ the thesis binary runs, and neither may
re-implement it: the Praat calibration (scripts/pipeline/thesis/05_thesis_dysphonia_calibration.py)
and the WORLD voicing check of the BIOMEX-DB base (scripts/data/world_voicing.py, run by
biomex_verify.py). This module holds the C wrapper, its build and its call; it needs g++ and
a configured max-performance build tree (for the vendored WORLD and nlohmann_json sources), and
no Praat, so the base's verifier and its CI test can use it.

    build_library(tmp) -> path of libdysphonia.so     (compile, once per process tree)
    load_library(so)                                   (once per worker process)
    call(x, fs, native, profile, params, seed, break_kind) -> (rc, out, message)
    split_request(profile, params) -> (numeric params, break_kind)   (a request as the
                                       calibration writes it, made ready for call())
"""
from __future__ import annotations

import ctypes
import glob
import os
import subprocess

import numpy as np

NN = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
LIB = os.path.join(NN, "src", "experiments", "thesis", "lib")
BUILD = os.path.join(NN, "out", "build", "max-performance")

# thesis::DysphoniaProfile / thesis::BreakKind, in their declaration order.
PROFILE_CODE = {"resynthesis_control": 0, "rough": 1, "tremor": 2, "voice_breaks": 3,
                "diplophonia": 4}
BREAK_CODE = {"pitch_jump": 0, "aphonic": 1}
PARAMS = ("jitter_pct", "shimmer_db", "f0_depth_pct", "rate_hz", "breaks_per_second",
          "break_ms", "period_alt_pct", "amp_alt_db")

WRAPPER = r"""
#include "VoiceDysphonia.hpp"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <vector>
// p: jitter_pct, shimmer_db, f0_depth_pct, rate_hz, breaks_per_second, break_ms,
//    period_alt_pct, amp_alt_db
extern "C" int dysphonia(const float* x, std::size_t n, double fs, double native, int profile,
                         int break_kind, const double* p, std::uint32_t seed, float* out,
                         char* err, std::size_t err_len)
{
    try {
        thesis::DysphoniaParams d;
        d.profile = static_cast<thesis::DysphoniaProfile>(profile);
        d.break_kind = static_cast<thesis::BreakKind>(break_kind);
        d.jitter_pct = p[0]; d.shimmer_db = p[1]; d.tremor_f0_depth_pct = p[2];
        d.tremor_rate_hz = p[3]; d.breaks_per_second = p[4]; d.break_ms = p[5];
        d.period_alt_pct = p[6]; d.amp_alt_db = p[7];
        thesis::check_dysphonia_params(d);
        const std::vector<float> in(x, x + n);
        const std::vector<float> y = thesis::degrade_voice_f0(in, fs, native, d, seed);
        if (y.size() != n) { std::strncpy(err, "length changed", err_len - 1); return 1; }
        std::memcpy(out, y.data(), n * sizeof(float));
        return 0;
    } catch (const std::exception& e) {
        std::strncpy(err, e.what(), err_len - 1);
        return 1;
    }
}
"""

_LIB = None  # per worker process


def build_library(tmp: str) -> str:
    world = os.path.join(BUILD, "_deps", "world-src", "src")
    json_inc = os.path.join(BUILD, "_deps", "nlohmann_json-src", "include")
    for need in (world, json_inc):
        if not os.path.isdir(need):
            raise SystemExit(f"[thesis-dysphonia-lib] {need} is missing: configure the "
                             f"max-performance preset first (cmake --preset=max-performance)")
    src = os.path.join(tmp, "wrap.cpp")
    with open(src, "w", encoding="utf-8") as f:
        f.write(WRAPPER)
    so = os.path.join(tmp, "libdysphonia.so")
    cmd = (["g++", "-std=c++20", "-O2", "-shared", "-fPIC", "-Wl,--no-undefined",
            "-I", os.path.join(LIB, "include"), "-I", os.path.join(NN, "include"),
            "-I", os.path.join(NN, "src"), "-I", json_inc, "-I", world, src,
            os.path.join(LIB, "src", "VoiceDysphonia.cpp"),
            os.path.join(NN, "src", "core", "utility", "resample.cpp")]
           + sorted(glob.glob(os.path.join(world, "*.cpp"))) + ["-o", so])
    subprocess.run(cmd, check=True)
    return so


def load_library(so: str) -> None:
    global _LIB
    lib = ctypes.CDLL(so)
    lib.dysphonia.restype = ctypes.c_int
    fp = ctypes.POINTER(ctypes.c_float)
    lib.dysphonia.argtypes = [fp, ctypes.c_size_t, ctypes.c_double, ctypes.c_double,
                              ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double),
                              ctypes.c_uint32, fp, ctypes.c_char_p, ctypes.c_size_t]
    _LIB = lib


def loaded() -> bool:
    return _LIB is not None


def split_request(profile: str, params: dict) -> tuple[dict, str]:
    """A dysphonia request as the calibration (and the profiles) write it, made ready for call().

    In a request the break kind of a voice_breaks level travels among the numeric parameters,
    {"break_kind": "aphonic", "breaks_per_second": 1.0, "break_ms": 120}, because it is part of
    the level's name. call() takes it as an argument of its own and refuses any key that is not
    a numeric parameter. Passing the request through unsplit is what stopped the calibration's
    verification pass from 2026-11-16 (when this module was split out) until 2026-11-26.

    -> (numeric parameters, break kind). Refused (ValueError): an unknown profile, a voice_breaks
    request without a break_kind or with an unknown one, a break_kind on any other profile, or
    any other non-numeric key. For a profile other than voice_breaks the kind passed to the C++
    is "pitch_jump", which degrade_voice_f0 reads only under voice_breaks.
    """
    if profile not in PROFILE_CODE:
        raise ValueError(f"thesis_dysphonia_lib: unknown profile {profile!r}")
    numeric = {k: v for k, v in params.items() if k != "break_kind"}
    if profile == "voice_breaks":
        if "break_kind" not in params:
            raise ValueError("thesis_dysphonia_lib: a voice_breaks request names no break_kind")
        kind = params["break_kind"]
        if kind not in BREAK_CODE:
            raise ValueError(f"thesis_dysphonia_lib: unknown break_kind {kind!r}")
    else:
        if "break_kind" in params:
            raise ValueError(f"thesis_dysphonia_lib: break_kind applies to voice_breaks only, "
                             f"not to {profile}")
        kind = "pitch_jump"
    unknown = set(numeric) - set(PARAMS)
    if unknown:
        raise ValueError(f"thesis_dysphonia_lib: unknown parameters {sorted(unknown)}")
    return numeric, kind


def call(x, fs: float, native: float, profile: str, params: dict, seed: int,
         break_kind: str = "pitch_jump") -> tuple[int, np.ndarray, str]:
    """thesis::degrade_voice_f0 on one voice: (0, output, "") or (1, undefined, its message)."""
    if _LIB is None:
        raise RuntimeError("thesis_dysphonia_lib: load_library() was not called in this process")
    unknown = set(params) - set(PARAMS)
    if unknown:
        raise ValueError(f"thesis_dysphonia_lib: unknown parameters {sorted(unknown)}")
    arr = (ctypes.c_double * len(PARAMS))(*[float(params.get(k, 0.0)) for k in PARAMS])
    x = np.ascontiguousarray(x, dtype=np.float32)
    out = np.empty_like(x)
    err = ctypes.create_string_buffer(512)
    fp = ctypes.POINTER(ctypes.c_float)
    rc = _LIB.dysphonia(x.ctypes.data_as(fp), x.size, float(fs), float(native),
                        PROFILE_CODE[profile], BREAK_CODE[break_kind], arr,
                        seed & 0xFFFFFFFF, out.ctypes.data_as(fp), err, 512)
    return rc, out, err.value.decode(errors="replace")
