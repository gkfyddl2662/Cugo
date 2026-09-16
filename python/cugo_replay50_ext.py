from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from cugo_torch50_ext import _ensure_windows_build_env


def load_extension(*, verbose: bool = False) -> Any:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is not available")

    _ensure_windows_build_env()
    from torch.utils.cpp_extension import load

    repo = Path(__file__).resolve().parents[1]
    source_dir = repo / "python" / "csrc"
    build_dir = repo / "build" / "torch_extensions" / "cugo_replay50_ext"
    build_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MAX_JOBS", "4")
    major, minor = torch.cuda.get_device_capability()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")

    if os.name == "nt":
        cxx_flags = [
            "/O2",
            "/std:c++20",
            "/EHsc",
            "/utf-8",
            "/Zc:preprocessor",
        ]
        cuda_flags = [
            "-O3",
            "-std=c++20",
            "-Xcompiler=/utf-8",
            "-Xcompiler=/Zc:preprocessor",
        ]
    else:
        cxx_flags = ["-O3", "-std=c++20"]
        cuda_flags = ["-O3", "-std=c++20"]

    return load(
        name="cugo_replay50_ext_v1",
        sources=[
            str(source_dir / "replay50_ext.cpp"),
            str(source_dir / "replay50_ext.cu"),
        ],
        extra_cflags=cxx_flags,
        extra_cuda_cflags=cuda_flags,
        build_directory=str(build_dir),
        with_cuda=True,
        verbose=verbose,
        keep_intermediates=True,
    )
