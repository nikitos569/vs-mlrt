__version__ = "3.20.12"

__all__ = [
    "Backend", "BackendV2",
    "Waifu2x", "Waifu2xModel",
    "DPIR", "DPIRModel",
    "RealESRGAN", "RealESRGANModel",
    "RealESRGANv2", "RealESRGANv2Model",
    "CUGAN",
    "RIFE", "RIFEModel", "RIFEMerge",
    "SAFA", "SAFAModel", "SAFAAdaptiveMode",
    "SCUNet", "SCUNetModel",
    "SwinIR", "SwinIRModel",
    "inference"
]

import copy
from dataclasses import dataclass, field
import enum
from fractions import Fraction
import math
import os
import os.path
import platform
import subprocess
import sys
import tempfile
import time
import typing
import zlib

import vapoursynth as vs
from vapoursynth import core


def get_plugins_path() -> str:
    path = b""

    path = core.migx.Version()["path"]

    assert path != b""

    return os.path.dirname(path).decode()

plugins_path: str = get_plugins_path()
trtexec_path: str = os.path.join(plugins_path, "vsmlrt-cuda", "trtexec")
migraphx_driver_path: str = os.path.join(plugins_path, "vsmlrt-hip", "migraphx-driver")
models_path: str = os.path.join(plugins_path, "models")

print(plugins_path)
