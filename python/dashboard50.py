"""Local training dashboard: parses build/ logs, checkpoints and eval results.

Standard library only, so it runs from any Python without touching the GPU:

    python python/dashboard50.py            # http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO = Path(__file__).resolve().parents[1]
PAGE = Path(__file__).with_name("dashboard50.html")
KV = re.compile(r"([A-Za-z_][\w]*)=(\S+)")
HEADER_LINE = re.compile(r"^(?:\w+ )?\w+=\S+(?: \w+=\S+)*$")
ITER_FILE = re.compile(r"iter_(\d+)\.pt$")
MAX_POINTS = 1500
RUNNING_WINDOW_S = 90.0

_cache: dict[Path, tuple[tuple[float, int], dict]] = {}


def read_text(path: Path) -> str:
    raw = path.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    return raw.decode("utf-8-sig", errors="replace")


def number(value: str) -> float | str:
    try:
        return float(value)
    except ValueError:
        return value


def cached(path: Path, parse) -> dict:
    stat = path.stat()
    key = (stat.st_mtime, stat.st_size)
    hit = _cache.get(path)
    if hit and hit[0] == key:
        return hit[1]
    result = parse(path, stat.st_mtime)
    _cache[path] = (key, result)
    return result


def parse_training_log(path: Path, mtime: float) -> dict:
    config: dict[str, float | str] = {}
    rows: dict[int, dict[str, float]] = {}
    checkpoint_dirs: list[str] = []
    finished = failed = False
    final: dict[str, float | str] = {}

    for line in read_text(path).splitlines():
        line = line.strip()
        if line.startswith("iter="):
            pairs = KV.findall(line)
            iteration = int(pairs[0][1])
            # Logs are appended across invocations. A run resumed from an earlier
            # checkpoint replaces the iterations after it, and progress after a
            # crash means the crash belonged to an earlier invocation.
            if rows and iteration < max(rows):
                rows = {i: r for i, r in rows.items() if i < iteration}
            failed = finished = False
            row = rows.setdefault(iteration, {})
            for key, value in pairs[1:]:
                parsed = number(value)
                if isinstance(parsed, float):
                    row[key] = parsed
        elif line.startswith("checkpoint="):
            parent = str(Path(line.split("=", 1)[1]).parent)
            if parent not in checkpoint_dirs:
                checkpoint_dirs.append(parent)
        elif "training PASS" in line:
            finished = True
            final = {k: number(v) for k, v in KV.findall(line)}
        elif line.startswith("Traceback") or "Error:" in line:
            failed = True
        elif not rows and line.startswith("device="):
            config["device"] = line[len("device="):].split(" capability=")[0]
        elif not rows and HEADER_LINE.match(line):
            config.update({k: number(v) for k, v in KV.findall(line)})

    if not rows:
        return {"is_run": False}

    iterations = sorted(rows)
    keys = sorted({key for row in rows.values() for key in row})
    stride = max(1, len(iterations) // MAX_POINTS)
    sampled = iterations[::stride]
    if sampled[-1] != iterations[-1]:
        sampled.append(iterations[-1])
    if stride > 1:  # never drop rows that carry sparse metrics such as periodic evals
        sparse = {k for k in keys if sum(k in row for row in rows.values()) * 2 < len(rows)}
        extra = {i for i in iterations if sparse & rows[i].keys()}
        sampled = sorted(set(sampled) | extra)

    timed_keys = ("selfplay_s", "buffer_add_s", "br_train_s", "average_train_s", "train_s")
    measured_s = sum(row.get(k, 0.0) for row in rows.values() for k in timed_keys)
    if any(k.startswith("ppo_") for k in keys):
        kind = "ppo"
    elif any(k.startswith("br_") for k in keys):
        kind = "nfsp"
    else:
        kind = "train50"

    return {
        "is_run": True,
        "kind": kind,
        "outcome": "done" if finished else "failed" if failed else None,
        "mtime": mtime,
        "config": config,
        "final": final,
        "iterations": len(iterations),
        "last_iteration": iterations[-1],
        "total_games": sum(row.get("games", 0.0) for row in rows.values()),
        "measured_seconds": measured_s,
        "checkpoint_dirs": checkpoint_dirs,
        "latest": {k: v for i in iterations for k, v in rows[i].items()},
        "series": {
            "iter": sampled,
            **{key: [rows[i].get(key) for i in sampled] for key in keys},
        },
    }


def live_status(info: dict, idle: str = "stopped") -> str:
    """Status depends on the clock, so it is computed per request, never cached."""
    if info.get("outcome"):
        return info["outcome"]
    return "running" if time.time() - info["mtime"] < RUNNING_WINDOW_S else idle


def parse_eval_log(path: Path, mtime: float) -> dict:
    sides: dict[str, dict] = {}
    record: dict = {"outcome": None}
    for line in read_text(path).splitlines():
        line = line.strip()
        head = line.split(" ", 1)[0]
        values = {k: number(v) for k, v in KV.findall(line)}
        if head in ("A", "B"):
            checkpoint = Path(str(values.get("checkpoint", "")))
            sides[head] = {
                "run": checkpoint.parent.name,
                "file": checkpoint.name,
                "policy": values.get("policy"),
                "iteration": values.get("iteration"),
            }
        elif head in ("RESULT", "ENGINE"):
            record[head.lower()] = values
        elif head == "PAIRS":
            ci = re.search(r"ci95=\[([-\d.]+),([-\d.]+)\]", line)
            record["pairs"] = values | (
                {"ci_low": float(ci.group(1)), "ci_high": float(ci.group(2))} if ci else {}
            )
        elif line == "evaluation PASS":
            record["outcome"] = "done"
    record.update(name=path.stem, mtime=mtime, a=sides.get("A"), b=sides.get("B"))
    return record


def list_checkpoints(dirs: list[str]) -> list[dict]:
    found = []
    for name in dirs:
        directory = Path(name) if Path(name).is_absolute() else REPO / name
        for file in sorted(directory.glob("*.pt")):
            match = ITER_FILE.search(file.name)
            stat = file.stat()
            found.append(
                {
                    "file": str(file.relative_to(REPO)),
                    "iteration": int(match.group(1)) if match else None,
                    "bytes": stat.st_size,
                    "mtime": stat.st_mtime,
                }
            )
    return found


def training_logs(build: Path) -> list[Path]:
    logs = list(build.glob("*.log")) + list((build / "logs").glob("*.log"))
    return sorted(logs, key=lambda p: p.stat().st_mtime, reverse=True)


def run_id(build: Path, path: Path) -> str:
    return path.relative_to(build).as_posix()


def summarize_runs(build: Path) -> list[dict]:
    runs = []
    for path in training_logs(build):
        info = cached(path, parse_training_log)
        if not info["is_run"]:
            continue
        target = info["final"].get("iterations") or info["config"].get("iterations")
        if target is not None and target < info["last_iteration"]:
            target = None  # several trainer invocations appended to one log
        runs.append(
            {
                "id": run_id(build, path),
                "name": path.stem,
                "kind": info["kind"],
                "status": live_status(info),
                "mtime": info["mtime"],
                "last_iteration": info["last_iteration"],
                "target_iterations": target,
            }
        )
    return runs


_gpu_cache: tuple[float, dict | None] = (0.0, None)


def gpu_status() -> dict | None:
    global _gpu_cache
    if time.time() - _gpu_cache[0] < 2.0:
        return _gpu_cache[1]
    fields = "name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3, check=True,
        ).stdout.splitlines()[0]
        name, util, used, total, temp, power = [part.strip() for part in out.split(",")]
        status = {
            "name": name, "util": number(util), "mem_used": number(used),
            "mem_total": number(total), "temp": number(temp), "power": number(power),
        }
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        status = None
    _gpu_cache = (time.time(), status)
    return status


def make_handler(build: Path):
    class Handler(BaseHTTPRequestHandler):
        def send(self, body: bytes, content_type: str, code: int = 200) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def json(self, payload, code: int = 200) -> None:
            self.send(json.dumps(payload).encode(), "application/json", code)

        def do_GET(self) -> None:
            url = urlparse(self.path)
            if url.path == "/":
                self.send(PAGE.read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/state":
                evals = [(e := cached(p, parse_eval_log)) | {"status": live_status(e, "failed")}
                         for p in (build / "evals").glob("*.log")]
                evals.sort(key=lambda e: e["mtime"], reverse=True)
                self.json({"runs": summarize_runs(build), "evals": evals, "gpu": gpu_status(),
                           "now": time.time()})
            elif url.path == "/api/run":
                wanted = parse_qs(url.query).get("id", [""])[0]
                path = (build / wanted).resolve()
                if build.resolve() not in path.parents or not path.is_file():
                    self.json({"error": "unknown run"}, 404)
                    return
                info = cached(path, parse_training_log)
                self.json(info | {"id": wanted, "name": path.stem, "status": live_status(info),
                                  "checkpoints": list_checkpoints(info.get("checkpoint_dirs", []))})
            else:
                self.json({"error": "not found"}, 404)

        def log_message(self, *args) -> None:
            pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Cugo training dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--build", type=Path, default=REPO / "build")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(args.build.resolve()))
    print(f"dashboard http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
