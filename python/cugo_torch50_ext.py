from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _prepend_path(path: Path) -> None:
    value = str(path)
    current = os.environ.get("PATH", "")
    normalized = os.path.normcase(os.path.abspath(value))
    for item in current.split(os.pathsep):
        if item and os.path.normcase(os.path.abspath(item.strip('"'))) == normalized:
            return
    os.environ["PATH"] = value + (os.pathsep + current if current else "")


def _find_vcvars64() -> Path | None:
    candidates: list[Path] = []

    vs_install_dir = os.environ.get("VSINSTALLDIR")
    if vs_install_dir:
        candidates.append(Path(vs_install_dir) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat")

    program_files_x86 = Path(
        os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    )
    vswhere = program_files_x86 / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if vswhere.is_file():
        result = subprocess.run(
            [
                str(vswhere),
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
            ],
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            candidates.append(
                Path(result.stdout.strip().splitlines()[-1])
                / "VC"
                / "Auxiliary"
                / "Build"
                / "vcvars64.bat"
            )

    visual_studio_root = program_files_x86 / "Microsoft Visual Studio"
    if visual_studio_root.is_dir():
        candidates.extend(
            sorted(
                visual_studio_root.glob("*/*/VC/Auxiliary/Build/vcvars64.bat"),
                reverse=True,
            )
        )

    seen: set[str] = set()
    for candidate in candidates:
        key = os.path.normcase(os.path.abspath(str(candidate)))
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return candidate
    return None


def _ensure_windows_build_env() -> None:
    if os.name != "nt":
        return

    # A Python executable inside a venv does not guarantee that its Scripts
    # directory is present in PATH when invoked by absolute path.  PyTorch's
    # JIT loader shells out to both ninja.exe and cl.exe, so keep Scripts first.
    scripts_dir = Path(sys.executable).resolve().parent
    _prepend_path(scripts_dir)

    if shutil.which("cl") is None:
        vcvars64 = _find_vcvars64()
        if vcvars64 is None:
            raise RuntimeError(
                "MSVC cl.exe is not on PATH and vcvars64.bat could not be found. "
                "Install the Visual Studio C++ x64 build tools."
            )

        result = subprocess.run(
            ["cmd.exe", "/d", "/s", "/c", f'call "{vcvars64}" >nul && set'],
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(
                f"vcvars64.bat failed with exit code {result.returncode}: {detail}"
            )

        for line in result.stdout.splitlines():
            key, separator, value = line.partition("=")
            if separator and key:
                os.environ[key] = value

        # vcvars64 replaces PATH. Put the active venv's Scripts directory back
        # in front so the pip-installed Ninja remains discoverable.
        _prepend_path(scripts_dir)

    cl = shutil.which("cl")
    if cl is None:
        raise RuntimeError("MSVC environment setup completed but cl.exe is still unavailable")
    if shutil.which("ninja") is None:
        raise RuntimeError(
            "ninja.exe is unavailable. Install it in the active environment with "
            "`python -m pip install ninja`."
        )


def load_extension(*, verbose: bool = False) -> Any:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is not available")

    _ensure_windows_build_env()
    from torch.utils.cpp_extension import load

    repo = Path(__file__).resolve().parents[1]
    source_dir = repo / "python" / "csrc"
    build_dir = repo / "build" / "torch_extensions" / "cugo_torch50_ext"
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
        name="cugo_torch50_ext_v1",
        sources=[
            str(source_dir / "torch50_ext.cpp"),
            str(source_dir / "torch50_ext.cu"),
        ],
        extra_include_paths=[str(repo / "include")],
        extra_cflags=cxx_flags,
        extra_cuda_cflags=cuda_flags,
        build_directory=str(build_dir),
        with_cuda=True,
        verbose=verbose,
        keep_intermediates=True,
    )
